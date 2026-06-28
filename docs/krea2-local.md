# Krea 2 Turbo — model option (fal + local)

Krea 2 Turbo (`krea/Krea-2-Turbo`) is selectable alongside Z-Image Turbo. It's a 12.9B
distilled DiT (Qwen-Image VAE + Qwen3-VL-4B text encoder, flow-matching, CFG-free at 8
steps). This doc covers what works, the setup, and the experimental bits to validate on
GPU. The Z-Image runbook is `local-backend-verification.md`.

## Selecting it

Two independent axes — **model** (`z-image` default | `krea2`) and **backend/location**
(`fal` default | `local`):

| | `backend=fal` (hosted) | `backend=local` (your GPU) |
|---|---|---|
| `model=z-image` | txt2img, img2img, inpaint | txt2img, img2img, inpaint |
| `model=krea2` | **txt2img only** | txt2img, **experimental img2img** (no inpaint) |

Set it via the notebook `ModelSetup` cell (`model_name`), the `predict.py` `model` input,
or env: `DEFORUM_MODEL=krea2` (and `DEFORUM_BACKEND=local`). Precedence per axis:
`root` setting → env var → default.

## Why Krea 2 is text-to-image-first

Out of the box Krea 2 is **txt2img only**: diffusers ships just `Krea2Pipeline` (no
`Krea2Img2ImgPipeline`/`Krea2InpaintPipeline`), and fal exposes only `fal-ai/krea-2/turbo`
(no Krea 2 image-to-image endpoint). So:

- **fal + krea2**: single images and interpolation morphs work. `img2img` / `inpaint`
  raise a clear error pointing you to the local backend (img2img) or Z-Image (inpaint).
- **local + krea2**: txt2img + a **custom flow-match img2img** (built on `Krea2Pipeline`)
  so the Deforum 2D/3D animation loop runs. `inpaint` is unsupported (raises).

## Setup (local)

```bash
python install_requirements.py --with-local      # upgrades diffusers-from-source (needs Krea2Pipeline)
hf download krea/Krea-2-Turbo                     # ~26GB into $HF_HOME
# 12.9B won't fit one 24GB card in bf16 -> quantize:
export ZIMAGE_QUANTIZE=nf4                         # (or KREA2_QUANTIZE / DEFORUM_QUANTIZE)
export DEFORUM_BACKEND=local DEFORUM_MODEL=krea2
# optional, skip re-download: export KREA2_LOCAL_PATH=/path/to/Krea-2-Turbo
```

Verify the pipeline is present: `python -c "from diffusers import Krea2Pipeline"` and
that `transformers` exposes `Qwen3VLModel`. If either fails, re-run `--with-local` (it
force-reinstalls diffusers from source).

## VRAM (24GB cards)

bf16 transformer alone is ~26GB — it does **not** fit a single 24GB A5000. Options:

- **`ZIMAGE_QUANTIZE=nf4`** (recommended): ~quarter resident, near-bf16 speed, fits one
  card. nf4 quantizes the transformer + text encoder (the existing quant lever).
- **`int8`**: ~half resident, ~bf16 quality, but ~1.5x slower per image on Ampere (no
  native int8 compute) and tighter with the 4B text encoder.
- **Multi-GPU bf16** (`device_map`/accelerate across both A5000s): full fidelity, no
  quant. Not the default and not wired into the loader yet — a follow-up.

## Defaults / CFG

Distilled turbo checkpoint: `steps=8`, `guidance_scale=0.0` (CFG-free). All Krea 2 entry
points force guidance to 0.0 (a non-zero request prints a one-time note). The experimental
gradient-conditioning guidance (CLIP/aesthetic) is **not** wired for Krea 2 (its latents
are packed); use Z-Image for that.

## License

Krea 2 weights are under the **Krea 2 Community License** — review its terms before use
(it differs from Z-Image's license).

## GPU validation results (RTX A5000, nf4)

Validated on a single 24GB RTX A5000 with diffusers from source (commit with
`Krea2Pipeline`), `krea/Krea-2-Turbo` weights, `ZIMAGE_QUANTIZE=nf4`:

| Item | Result |
|---|---|
| txt2img 512², 8 steps | ✅ ~23s incl. load+nf4 quant; coherent image |
| Resident VRAM (nf4) | ✅ peak **11.9 GB** — fits one 24GB card with headroom |
| Custom img2img (strength 0.65) | ✅ ~3.2s; **structure preserved, prompt change applied** |
| VAE encode temporal + mean/std norm | ✅ no shape error |
| Packing → `transformer.in_channels` | ✅ assert passes (no mismatch) |
| Sigma alignment (the highest-risk item) | ✅ the pre-noised start sigma matches the tail schedule (`scale_noise` at `begin_index=t_start`); confirmed against `pipeline_krea2.py` and at runtime |
| 2D animation loop (6 frames, motion warp) | ✅ ~39s; frame 0 txt2img + custom img2img per frame; temporally coherent |
| Interpolation morph (2 keys, 4D embed slerp) | ✅ ~51s; semantic morph, endpoints match prompts (sharp midpoint when prompts differ greatly — use more frames / closer prompts to smooth) |

The custom img2img mirrors the proven Qwen-Image img2img (same VAE + packing family) and
matches `Krea2Pipeline`'s own `prepare_latents`/`_pack_latents`/`scale_noise` contracts.
All planned GPU-validation items pass.
