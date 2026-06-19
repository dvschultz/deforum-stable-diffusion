# Local Z-Image backend — GPU verification runbook

**Read this into Claude on the GPU box to drive full validation of the local backend.**

## What this is

The local Z-Image backend (`backend='local'`) was built and **unit-tested with the
diffusers pipeline mocked** on a Mac with no GPU. That proves the wiring (routing,
knob-gating, callback composition, grad-math, the fal path staying torch-free) but
**not** that the real diffusers `ZImage*` pipeline accepts the kwargs `helpers/zimage_local.py`
passes, nor that any feature actually works on hardware. This runbook closes that gap.

Branch: `feat/z-image-turbo-backend` (PR #1 on the `dvschultz` fork). The default `fal`
backend is fully verified and unaffected — everything here is the opt-in local path.

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
hf download Tongyi-MAI/Z-Image-Turbo              # ~6B weights; note the local path
export DEFORUM_BACKEND=local                      # or set backend in ModelSetup
# optional: export ZIMAGE_LOCAL_PATH=/path/to/Z-Image-Turbo   # skip re-download
python -m pytest tests/ -q                         # sanity: mocked suite should still pass
```

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
high steps, that confirms the plan's risk note — document it; don't force it. CLIP/aesthetic
guidance also need their models loaded on `root` (`clip_scale>0` must load `root.clip_model`);
if guidance silently no-ops, check that wiring in the entry/predict setup.

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
