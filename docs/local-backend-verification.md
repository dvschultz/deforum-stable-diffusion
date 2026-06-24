# Local Z-Image backend — GPU verification runbook

**Read this into Claude on the GPU box to drive full validation of the local backend.**

## Validation results (complete)

Run on 2× NVIDIA RTX A5000 (24 GB, Ampere/sm_86), diffusers-from-source, torch 2.6.0+cu124.
**All steps pass**, and the feared diffusers `ZImage*` API drift did **not** materialize — the
kwarg contract holds. Merged to `main` via PR #1 and PR #2.

| Step | Result |
|---|---|
| 1. Pipeline loads | ✅ `ZImagePipeline`, cuda, bf16, ~11 s |
| 2. txt2img | ✅ 512², ~16 s — kwarg contract holds (no API drift) |
| 3. img2img + strength | ✅ faithful variation; Deforum 0.65 → diffusers 0.35 inversion correct |
| 4. inpaint | ✅ masked center changed ~10× vs edges (preserved) |
| 5. 2D animation (render loop) | ✅ coherent motion; fits 24 GB via component sharing (peak ~21 GB) |
| 6. true 16-bit (+ 32-bit) | ✅ 16-bit PNG (IHDR=16, max 65279); 32-bit `.exr` (float32) also confirmed |
| 7. per-step previews + threshold | ✅ after fix (below); 8 previews, thresholding stable |
| 8. interpolation | ✅ local = **semantic morph** (~70/255 from a pixel blend); fal = **pixel cross-dissolve** (~0.2) |
| 9. gradient guidance | ✅ model-free steers (blue +11 @8 steps, +27.6 @30); **CLIP wired & steers** (+0.13 cosine to a target text) |

### Bugs found on hardware and fixed
- **`zimage_local._make_step_callback`** (step 7) — per-step preview VAE-decode missed the dtype cast + `shift_factor` (fp32 latents vs bf16 VAE); error was swallowed → zero previews.
- **`zimage_local.encode_prompt`** (step 8) — ran the text encoder without `torch.no_grad()`, leaking the ~3.4 GB activation graph → OOM'd interpolation.
- **`zimage_local.txt2img_embeds`** (step 8) — passed CFG with only positive embeds, which the pipeline rejects; forced CFG-free.
- **`local_guidance.decode_fn`** (step 9) — same dtype+shift bug (differentiable variant).
- **`install_requirements.py`** — `scikit-image==0.19.3` is numpy-2-incompatible (→ `>=0.24`) and `scikit-learn` was undeclared. The box also needed a CUDA-12.4 torch build (cu124, **not** cu130) and `pytest`.

### Added during validation
- **CLIP/aesthetic guidance wiring** — `clip_scale`/`aesthetics_scale` now load `root.clip_model` / `root.aesthetics_model` (previously a silent no-op).
- **Component sharing** in `_load_pipe` — kinds reuse one copy of the weights, so animation fits 24 GB at full bf16 speed (~21 GB vs ~40 GB). See "VRAM" below.
- **Opt-in quantization** (`ZIMAGE_QUANTIZE=int8`/`nf4`) — ~11 GB / ~6.5 GB resident for memory-bound runs (a memory lever only on Ampere; int8 ~1.5× slower).

Mocked suite stays green throughout (71 passed). Nothing outstanding from this runbook.

## What this is

The local Z-Image backend (`backend='local'`) was built and **unit-tested with the
diffusers pipeline mocked** on a Mac with no GPU. That proves the wiring (routing,
knob-gating, callback composition, grad-math, the fal path staying torch-free) but
**not** that the real diffusers `ZImage*` pipeline accepts the kwargs `helpers/zimage_local.py`
passes, nor that any feature actually works on hardware. This runbook closes that gap.

Now merged to `main` (PR #1 + follow-up PR #2). The default `fal` backend is fully verified
and unaffected — everything here is the opt-in local path.

**Highest-likelihood failure: diffusers API drift.** The ZImage pipelines aren't in a
stable diffusers release, so a `__call__`/`encode_prompt` parameter name in
`helpers/zimage_local.py` may not match the installed version. If you hit
`TypeError: __call__() got an unexpected keyword argument '...'` or a missing attribute,
that's the fix target — see "If something breaks" at the end.

## Agent instructions

Work through the steps in order. After each: report the command, the actual result, and
PASS/FAIL. On FAIL, capture the full traceback, make the smallest fix in the named file,
re-run, and note what changed. Don't proceed past a failing foundational step (1–2) — later
steps build on it. Keep a running summary; at the end, list what passed, what you fixed,
and anything still broken.

## Prerequisites

```bash
cd <repo>
git checkout feat/z-image-turbo-backend && git pull
python install_requirements.py --with-local      # torch + diffusers-from-source (large)
hf download Tongyi-MAI/Z-Image-Turbo              # ~6B weights
export DEFORUM_BACKEND=local                      # or set backend in ModelSetup
# optional: export ZIMAGE_LOCAL_PATH=/path/to/Z-Image-Turbo   # skip re-download
# optional: export ZIMAGE_QUANTIZE=int8            # only if memory-bound; see "VRAM" below
python -m pytest tests/ -q                         # sanity: mocked suite should still pass
```

## Disk / model location (set FIRST if the main drive is tight)

The weights (~6B), CUDA torch wheels, and the conda env are multi-GB and default to the
home/system drive. If that drive is full, redirect them to an SSD **before** installing or
downloading — set these in the shell (or the conda env's activate script) so the install,
the download, AND every generation run all see the SSD:

```bash
export HF_HOME=/mnt/ssd/hf            # redirects ALL HuggingFace downloads (model + sub-components)
export PIP_CACHE_DIR=/mnt/ssd/pipcache
# conda env on the SSD too:
conda create -p /mnt/ssd/envs/dsd-local python=3.11 -y && conda activate /mnt/ssd/envs/dsd-local
```

- `HF_HOME` is the robust lever: with it set, `hf download Tongyi-MAI/Z-Image-Turbo` and the
  default `from_pretrained("Tongyi-MAI/Z-Image-Turbo")` both resolve from the SSD — no code
  or path arg needed. It also catches any auxiliary repos diffusers pulls (text encoder, etc.).
- Explicit alternative: `hf download Tongyi-MAI/Z-Image-Turbo --local-dir /mnt/ssd/Z-Image-Turbo`
  then `export ZIMAGE_LOCAL_PATH=/mnt/ssd/Z-Image-Turbo` (the loader honors it). Keep `HF_HOME`
  set as well in case a sub-component isn't bundled in that dir.
- Verify before pulling 6B: `df -h /mnt/ssd` (need ~20–30 GB headroom for weights + torch).

## VRAM (24GB cards)

Default is **full bf16** (~20GB resident, fastest). `_load_pipe` **shares components** across
kinds (txt2img/img2img/inpaint reuse one copy of the transformer/text-encoder/vae — same
module objects, no extra VRAM), so a 2D animation stays ~20GB instead of ~40GB —
full-precision animation fits a 24GB card at full speed. No need to quantize to run animations.

If you're still memory-bound (e.g. very tight cards, or stacking guidance + interpolation),
opt into bitsandbytes weight quantization of the transformer **and** text encoder:
`ZIMAGE_QUANTIZE=int8` (~11GB, ~bf16 quality) or `nf4` (~6.5GB, lower fidelity to bf16).
Caveat: on Ampere (A5000, sm_86) there's **no native int8/fp8 compute** — matmuls dequantize
to bf16, so quant is a *memory* lever, not a speed one. Measured per-image (512², 8 steps):
bf16 ~3.6s, **int8 ~5.6s (~1.5× slower, steady over a full batch)**, nf4 ~3.9s. So quantize
only when you actually need the memory.

## Test sequence

### 1. Pipeline loads (foundational)
```bash
python -c "from helpers import zimage_local as zl; p=zl._load_pipe('txt2img'); print(type(p).__name__)"
```
Expect a `ZImagePipeline`. FAIL here = install/weights/diffusers problem; fix before continuing.

### 2. txt2img round-trip (foundational — proves the kwarg contract)
```bash
python -c "from helpers import zimage_local as zl; zl.txt2img('a serene mountain lake, painterly', 512, 512, steps=8, guidance_scale=5.0)[0].save('loc_txt2img.png')"
```
Expect `loc_txt2img.png`. A `TypeError` on a kwarg = API drift → fix `_common_kwargs` / the
pipe call in `helpers/zimage_local.py`, re-run.

### 3. img2img + strength inversion
```bash
python -c "from helpers import zimage_local as zl; from PIL import Image; init=Image.open('loc_txt2img.png'); zl.img2img('the same lake, more mist', init, 0.65, 512, 512, steps=8)[0].save('loc_img2img.png')"
```
Expect a coherent **variation** of the input (Deforum strength 0.65 → diffusers 0.35 →
faithful). If it's unrelated to the input, the strength mapping is wrong for diffusers'
convention (check `to_fal_strength` usage).

### 4. inpaint
```bash
python -c "from helpers import zimage_local as zl; from PIL import Image; img=Image.open('loc_txt2img.png'); m=Image.new('L',(512,512),0); m.paste(255,(128,128,384,384)); zl.inpaint('a bright red boat',img,m,0.6,512,512)[0].save('loc_inpaint.png')"
```
Expect the center region changed, edges preserved.

### 5. Full pipeline: a short 2D animation on the local backend
Run a 3–5 frame 2D animation through `render_animation` with `backend='local'` (mirror
`scripts/smoke_test.py` style, or the notebook with backend=local, animation_mode='2D',
max_frames=5). Expect coherent motion — this proves `generate()` routes to local and the
render loop's sample contract holds with real outputs.

### 6. True 16-bit output
Run a single txt2img with `bit_depth_output=16` (local). Expect a 16-bit PNG; open it and
confirm 16-bit depth (`exiftool` / an EXR/PNG16 viewer). 32-bit → an `.exr`.

### 7. Per-step previews + thresholding
Set `save_sample_per_step=True` (and try `dynamic_threshold=99.0`). Expect per-step
`<timestring>_step_NNN.png` files. Confirm thresholding doesn't blow up the image.

### 8. Interpolation morph (semantic vs pixel)
Run `Interpolation` mode with 2 key prompts on `backend='local'`, then again on `fal`.
Local should **morph** between concepts (embedding slerp); fal should **cross-dissolve**
(pixel blend). If local errors on token-length mismatch, see `slerp_embeds` (min-length
alignment may need refinement for very different-length prompts).

### 9. Gradient guidance A/B (the experimental one — judge honestly)
Generate the same prompt/seed three ways and compare:
- `clip_scale=0` (off) — baseline
- `clip_scale=5000` (or similar), `steps=8`
- same scale, `steps=30`

**Expected reality:** guidance may do little at 8 steps. Report whether the image visibly
shifts toward the guidance target, and whether more steps help. If it does nothing even at
high steps, that confirms the plan's risk note — document it; don't force it.

`clip_scale`/`aesthetics_scale` load their models onto `root` automatically:
`local_guidance._ensure_guidance_models` pulls the vendored CLIP (and the aesthetic
predictor) when those scales are set, and defaults the CLIP target (`clip_prompt`) to the
generation prompt. Verified: `clip_scale=5000` toward a different target text moves the
image toward it in CLIP space (+0.13 cosine). **Memory:** CLIP guidance backprops through
both the VAE decode and CLIP, so it peaks near 24GB at 448² — drop the resolution or set
`ZIMAGE_QUANTIZE=int8` to free room.

## If something breaks

| Symptom | Fix target |
|---|---|
| `TypeError: ... unexpected keyword argument` on generate | `helpers/zimage_local.py` `_common_kwargs` + the `pipe(...)` calls — align kwarg names to the installed diffusers `ZImage*.__call__` |
| `encode_prompt` signature mismatch (interpolation) | `helpers/zimage_local.py` `encode_prompt` |
| 16-bit save errors | `helpers/generate.py` `_format_highbit_results` + `render.py` `save_8_16_or_32bpc_image` |
| per-step callback errors | `helpers/zimage_local.py` `_make_step_callback` (VAE decode line) |
| guidance errors / no effect | `helpers/local_guidance.py` (`decode_fn` scaling, `build_guidance_loss` signatures) |
| fal path regressed | should be impossible — `tests/test_torch_free.py` + `tests/test_backend_select.py` guard it; run them |

When reporting back to the author's session, paste: the failing step, the full traceback,
and the diff of any fix you made. The mocked unit tests (`python -m pytest tests/ -q`) must
stay green after any fix.
