"""Local Krea 2 Turbo backend via the diffusers Krea2Pipeline.

Mirrors the helpers.zimage_local surface (txt2img / img2img / inpaint -> list of PIL
images) so helpers.generate routes uniformly when model="krea2", backend="local".
Shares the generic machinery (device/dtype, quantization, lazy loading, callbacks,
common kwargs) with Z-Image via helpers._diffusers_local; this module binds the Krea 2
:class:`ModelSpec` and adds the Krea-2-specific bits.

Two things are special about Krea 2:

* **txt2img only in diffusers.** There is no Krea2Img2ImgPipeline / Krea2InpaintPipeline,
  so img2img here is a *custom* flow-match implementation built on the txt2img pipeline
  (VAE-encode the init, blend noise at a strength-chosen sigma, pack, denoise only the
  remaining tail). EXPERIMENTAL -- the latent packing / VAE temporal handling / sigma
  alignment are GPU-validation items (see comments). inpaint is unsupported and raises.

* **12.9B + CFG-free.** bf16 transformer (~26GB) won't fit one 24GB GPU -- set
  ``ZIMAGE_QUANTIZE=nf4`` (or KREA2_QUANTIZE) to fit a single card. The distilled "turbo"
  checkpoint is CFG-free, so guidance_scale is forced to 0.0 everywhere.

EXPERIMENTAL / opt-in. Requires a CUDA GPU, a current diffusers-from-source (for
Krea2Pipeline), and the Krea-2-Turbo weights (`hf download krea/Krea-2-Turbo`, ~26GB).
Unit tests mock the pipeline; real generation needs the hardware.
"""
import os

import torch

from . import _diffusers_local as D
from ._diffusers_local import ModelSpec
from .zimage_client import to_fal_strength  # pure, torch-free; reused for strength inversion

# Weights: a HF id or a local path (set KREA2_LOCAL_PATH to a downloaded dir).
MODEL_ID = os.environ.get("KREA2_LOCAL_PATH", "krea/Krea-2-Turbo")

_PIPE_CLASSES = {
    "txt2img": "Krea2Pipeline",
    "img2img": None,   # built custom on the txt2img pipeline (see img2img below)
    "inpaint": None,   # unsupported (no Krea2InpaintPipeline)
}

SPEC = ModelSpec(
    name="krea2",
    model_id=MODEL_ID,
    pipe_classes=_PIPE_CLASSES,
    default_steps=8,
    default_guidance=0.0,        # distilled turbo checkpoint is CFG-free
    vae_norm="mean_std",         # Qwen video VAE (AutoencoderKLQwenImage)
    # Quant env precedence: model-specific, then a shared alias, then the Z-Image var
    # (so a single ZIMAGE_QUANTIZE=nf4 covers both backends).
    quant_env_vars=("KREA2_QUANTIZE", "DEFORUM_QUANTIZE", "ZIMAGE_QUANTIZE"),
)

_PIPES = {}  # kind -> loaded pipeline (cached); this model's own cache

_INPAINT_MSG = (
    "Inpaint is not supported for Krea 2 (no Krea2InpaintPipeline in diffusers). Use the "
    "Z-Image model for masked generation."
)

_cfg_free_noted = False


# Thin module-level wrappers over the shared core (kept patchable for tests).

def _load_pipe(kind):
    return D.load_pipe(SPEC, kind, _PIPES)


def _make_step_callback(**kw):
    return D.make_step_callback(SPEC, **kw)


def _note_cfg_free(guidance_scale):
    global _cfg_free_noted
    if guidance_scale and not _cfg_free_noted:
        _cfg_free_noted = True
        print("..Krea 2 Turbo is distilled / CFG-free; ignoring guidance_scale "
              f"(requested {guidance_scale}, using 0.0).")


def _kwargs(W, H, seed, steps, num_images, output_type, kw):
    """Common diffusers kwargs with guidance forced off and the Z-Image-tuned gradient
    guidance callback dropped (it assumes Z-Image's unpacked latents + shift/scale VAE;
    Krea 2 latents are packed, so wiring it would break)."""
    kw = dict(kw)
    if kw.pop("guidance_callback", None) is not None:
        # EXPERIMENTAL gradient guidance isn't supported on Krea 2's packed-latent path.
        pass
    return D.common_kwargs(SPEC, W, H, seed, steps, num_images, 0.0, output_type, kw)


def txt2img(prompt, W, H, seed=None, steps=8, num_images=1,
            guidance_scale=0.0, output_type="pil", **kw):
    """Text-to-image. Returns PIL images, or HWC float arrays when output_type='np'."""
    _note_cfg_free(guidance_scale)
    pipe = _load_pipe("txt2img")
    out = pipe(prompt=prompt, **_kwargs(W, H, seed, steps, num_images, output_type, kw))
    return list(out.images)


def _set_timesteps(scheduler, sigmas, device, mu):
    """Configure the flow-match scheduler over an explicit sigma schedule, passing the
    distilled fixed shift ``mu`` when the scheduler accepts it."""
    sig = [float(s) for s in sigmas]
    if mu is not None:
        try:
            scheduler.set_timesteps(sigmas=sig, mu=mu, device=device)
            return
        except TypeError:
            pass
    scheduler.set_timesteps(sigmas=sig, device=device)


def img2img(prompt, init_image, deforum_strength, W, H, seed=None, steps=8, num_images=1,
            guidance_scale=0.0, output_type="pil", **kw):
    """EXPERIMENTAL custom flow-match image-to-image on the txt2img Krea2Pipeline.

    diffusers ships no Krea2Img2ImgPipeline, so we do it by hand, mirroring the proven
    Qwen-Image img2img (same VAE + packing family): VAE-encode the init to a clean
    flow-match latent, blend in Gaussian noise at the sigma the requested strength maps
    to, pack to the transformer layout, and run the pipeline over only the remaining
    sigma tail (passed via ``latents=`` + ``sigmas=``; the pipeline uses provided latents
    verbatim and never re-noises them).

    GPU-VALIDATION ITEMS (untested without the weights): the VAE temporal-dim handling,
    that ``_pack_latents`` yields ``transformer.in_channels`` columns, and that the
    pre-noised start sigma aligns with the pipeline's tail schedule.
    """
    import numpy as np

    _note_cfg_free(guidance_scale)
    pipe = _load_pipe("txt2img")               # Krea 2 ships only the txt2img pipeline
    dev = pipe._execution_device
    dtype = pipe.transformer.dtype
    vae = pipe.vae
    p = int(pipe.patch_size)
    vsf = int(pipe.vae_scale_factor)
    nb = int(num_images)
    # Generator on the pipe's own device (so the blended-noise tensors match it).
    gen = None if seed is None else torch.Generator(device=dev).manual_seed(int(seed))
    strength = to_fal_strength(deforum_strength)   # Deforum -> diffusers (higher = more change)

    # ---- (1) VAE-encode the init image -> normalized clean flow-match latent x0 ----
    img = pipe.image_processor.preprocess(init_image, height=int(H), width=int(W))
    img = img.to(dev, vae.dtype).unsqueeze(2)              # [B,C,1,H,W] (Qwen video VAE temporal dim)
    with torch.no_grad():
        x0 = vae.encode(img).latent_dist.sample(generator=gen)   # [B,z,1,H',W']
    x0 = x0.squeeze(2)                                     # [B,z,H',W'] (drop T=1)
    z = int(vae.config.z_dim)
    lm = torch.tensor(vae.config.latents_mean, device=x0.device, dtype=x0.dtype).view(1, z, 1, 1)
    lstd = 1.0 / torch.tensor(vae.config.latents_std, device=x0.device, dtype=x0.dtype).view(1, z, 1, 1)
    x0 = (x0 - lm) * lstd                                  # normalized flow-match clean latent
    if nb > 1:
        x0 = x0.repeat(nb, 1, 1, 1)

    # latent grid + packed channel count -- exactly as Krea2Pipeline.prepare_latents:
    # latent_height = height // vae_scale_factor; num_channels_latents = in_channels // patch_size**2.
    h = int(H) // vsf
    w = int(W) // vsf
    C = int(pipe.transformer.config.in_channels) // (p * p)

    # ---- (2)+(3) full sigma schedule, truncate the tail by strength ----
    full = np.linspace(1.0, 1.0 / int(steps), int(steps))
    mu = 1.15 if getattr(pipe.config, "is_distilled", True) else None  # distilled: fixed shift
    _set_timesteps(pipe.scheduler, full, dev, mu)
    init_ts = min(int(steps) * strength, int(steps))
    t_start = int(max(int(steps) - init_ts, 0))
    order = int(getattr(pipe.scheduler, "order", 1) or 1)
    timesteps = pipe.scheduler.timesteps[t_start * order:]
    if len(timesteps) < 1:
        raise ValueError("strength leaves no denoising steps; lower the Deforum strength.")
    pipe.scheduler.set_begin_index(t_start * order)
    latent_t = timesteps[:1].repeat(x0.shape[0])

    # ---- (4) sample noise and flow-match blend at the start sigma ----
    noise = torch.randn((x0.shape[0], C, h, w), generator=gen, device=dev, dtype=x0.dtype)
    blended = pipe.scheduler.scale_noise(x0, latent_t, noise)   # sigma*noise + (1-sigma)*x0

    # ---- (5) pack to (B, seq, in_channels) and denoise only the remaining tail ----
    packed = pipe._pack_latents(blended, x0.shape[0], C, h, w).to(dtype)
    expected_ch = int(pipe.transformer.config.in_channels)
    if packed.shape[-1] != expected_ch:
        raise RuntimeError(
            f"Krea 2 img2img packing mismatch: packed latent has {packed.shape[-1]} channels "
            f"but the transformer expects {expected_ch}. (patch_size/vae_scale_factor mismatch.)"
        )
    tail = [float(s) for s in full[t_start:]]
    out = pipe(prompt=prompt, latents=packed, sigmas=tail,
               num_inference_steps=len(tail), guidance_scale=0.0,
               height=int(H), width=int(W), num_images_per_prompt=nb,
               generator=gen, output_type=output_type,
               **_callback_only(kw))
    return list(out.images)


def _callback_only(kw):
    """Just the per-step callback (thresholding/preview), since img2img sets size/steps/
    guidance explicitly. Drops the Z-Image gradient-guidance callback (packed-latent
    incompatible)."""
    kw = dict(kw)
    kw.pop("guidance_callback", None)
    cb = D.make_step_callback(SPEC, **kw)
    return {"callback_on_step_end": cb} if cb is not None else {}


def inpaint(*_args, **_kwargs):
    raise NotImplementedError(_INPAINT_MSG)


# --- Embedding-space interpolation (model-aware morph) -------------------

def encode_prompt(prompt):
    """Return Krea 2's text conditioning for a prompt as ``(prompt_embeds, mask)``.

    Krea 2 embeds are 4D ``(B, seq, num_text_layers, hidden)`` with a companion boolean
    mask -- different from Z-Image's list-of-token-tensors -- so the morph threads the
    pair through slerp_embeds/txt2img_embeds opaquely."""
    pipe = _load_pipe("txt2img")
    with torch.no_grad():
        embeds, mask = pipe.encode_prompt(
            prompt, device=pipe._execution_device, num_images_per_prompt=1)
    return (embeds, mask)


def slerp_embeds(e1, e2, t):
    """Slerp two Krea 2 prompt embeddings, aligning on the shorter sequence length.
    EXPERIMENTAL: 4D-embed slerp semantics are a GPU-validation item."""
    from .interpolation import interpolate
    emb1, mask1 = e1
    emb2, mask2 = e2
    n = min(emb1.shape[1], emb2.shape[1])      # align on the token/seq axis
    emb = interpolate(t, emb1[:, :n], emb2[:, :n], mode="slerp")
    mask = mask1[:, :n] if t < 0.5 else mask2[:, :n]
    return (emb, mask)


def txt2img_embeds(emb, W, H, seed=None, steps=8, guidance_scale=0.0,
                   output_type="pil", **kw):
    """Generate from precomputed (e.g. slerped) Krea 2 prompt embeddings + mask. CFG-free
    (the embeds path carries no negatives, and the distilled checkpoint is CFG-free)."""
    pipe = _load_pipe("txt2img")
    prompt_embeds, prompt_embeds_mask = emb
    out = pipe(prompt_embeds=prompt_embeds, prompt_embeds_mask=prompt_embeds_mask,
               **_kwargs(W, H, seed, steps, 1, output_type, kw))
    return list(out.images)
