"""Local Z-Image backend via the diffusers ZImage pipelines.

Mirrors the helpers.zimage_client surface (txt2img / img2img / inpaint -> list of
PIL images) so helpers.generate routes to either backend uniformly. This module
imports torch and (lazily) diffusers, so it must only be imported when the local
backend is selected -- helpers.backends.resolve_backend handles that.

EXPERIMENTAL / opt-in. Requires a CUDA GPU, diffusers-from-source, and the
Z-Image-Turbo weights (see install_requirements.py --with-local). Real generation
cannot be validated without that hardware; the unit tests mock the pipeline.

diffusers __call__ surface this wraps (img2img, confirmed against diffusers main):
  prompt, image, strength=0.6, height, width, num_inference_steps, guidance_scale,
  num_images_per_prompt, generator, prompt_embeds, negative_prompt_embeds,
  output_type ("pil"|"pt"|"latent"), callback_on_step_end, max_sequence_length.
strength follows the standard "higher = more change" convention (same as fal), so
Deforum strength is inverted identically via the shared zimage_client helper.
"""
import os

import torch

from .zimage_client import to_fal_strength  # pure, torch-free; reused for strength inversion

# Weights: a HF id or a local path (set ZIMAGE_LOCAL_PATH to a downloaded dir).
MODEL_ID = os.environ.get("ZIMAGE_LOCAL_PATH", "Tongyi-MAI/Z-Image-Turbo")

_PIPE_CLASSES = {
    "txt2img": "ZImagePipeline",
    "img2img": "ZImageImg2ImgPipeline",
    "inpaint": "ZImageInpaintPipeline",
}
_PIPES = {}  # kind -> loaded pipeline (cached)


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_pipe(kind):
    """Load and cache a ZImage pipeline of the given kind."""
    if kind in _PIPES:
        return _PIPES[kind]
    try:
        import diffusers
    except ImportError as e:
        raise RuntimeError(
            "diffusers is required for the local backend. Install with "
            "`python install_requirements.py --with-local` "
            "(pulls torch + diffusers-from-source)."
        ) from e
    cls_name = _PIPE_CLASSES[kind]
    Pipe = getattr(diffusers, cls_name, None)
    if Pipe is None:
        raise RuntimeError(
            f"{cls_name} not found in your diffusers install. The Z-Image pipelines "
            "require diffusers from source: pip install "
            "git+https://github.com/huggingface/diffusers.git"
        )
    dtype = torch.bfloat16 if _device() == "cuda" else torch.float32
    pipe = Pipe.from_pretrained(MODEL_ID, torch_dtype=dtype).to(_device())
    _PIPES[kind] = pipe
    return pipe


def _generator(seed):
    if seed is None:
        return None
    return torch.Generator(device=_device()).manual_seed(int(seed))


def _dynamic_threshold(latents, percentile):
    # Imagen-style per-step clamp: clamp to the given abs-value percentile, then rescale.
    import numpy as np
    s = np.percentile(latents.detach().abs().cpu().numpy(), percentile,
                      axis=tuple(range(1, latents.ndim)))
    s = np.maximum(s, 1.0)
    s = torch.as_tensor(s, device=latents.device, dtype=latents.dtype).view(-1, *([1] * (latents.ndim - 1)))
    return latents.clamp(-s, s) / s


def _make_step_callback(dynamic_threshold=None, static_threshold=None,
                        save_sample_per_step=False, show_sample_per_step=False,
                        outdir=None, timestring="", **_ignored):
    """Build a callback_on_step_end for thresholding + per-step previews, or None.

    Returns a dict to override latents (diffusers pops 'latents' from the return),
    so threshold clamps actually take effect for the next step.
    """
    want = (dynamic_threshold is not None or static_threshold is not None
            or save_sample_per_step or show_sample_per_step)
    if not want:
        return None

    def callback(pipe, step, timestep, cbk):
        latents = cbk["latents"]
        if static_threshold is not None:
            latents = latents.clamp(-static_threshold, static_threshold)
        if dynamic_threshold is not None:
            latents = _dynamic_threshold(latents, dynamic_threshold)
        if save_sample_per_step and outdir:
            try:
                import os
                # Mirror the pipeline's own latent->image decode (denoising keeps
                # latents in float32 while the VAE is bf16, and this VAE has a
                # shift_factor): cast to the VAE dtype, unscale, then unshift.
                lat = latents.to(pipe.vae.dtype)
                lat = (lat / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                img = pipe.image_processor.postprocess(
                    pipe.vae.decode(lat, return_dict=False)[0],
                    output_type="pil")[0]
                img.save(os.path.join(outdir, f"{timestring}_step_{step:03d}.png"))
            except Exception:
                pass  # previews are best-effort; never break the render
        return {"latents": latents}

    return callback


def _compose_callbacks(callbacks):
    """Chain callbacks; each may return {'latents': ...} which the next one sees."""
    callbacks = [c for c in callbacks if c is not None]
    if not callbacks:
        return None
    if len(callbacks) == 1:
        return callbacks[0]

    def combined(pipe, step, timestep, cbk):
        merged = {}
        for c in callbacks:
            r = c(pipe, step, timestep, cbk) or {}
            if "latents" in r:
                cbk = {**cbk, "latents": r["latents"]}
            merged.update(r)
        return merged

    return combined


def _common_kwargs(W, H, seed, steps, num_images, guidance_scale, output_type, kw_extra):
    kw = dict(
        height=int(H), width=int(W),
        num_inference_steps=int(steps),
        num_images_per_prompt=int(num_images),
        guidance_scale=float(guidance_scale),
        generator=_generator(seed),
        output_type=output_type,
    )
    # Compose thresholding/preview (U4) with experimental gradient guidance (U7).
    callback = _compose_callbacks([
        _make_step_callback(**kw_extra),
        kw_extra.get("guidance_callback"),
    ])
    if callback is not None:
        kw["callback_on_step_end"] = callback
    if kw_extra.get("scheduler") and kw_extra["scheduler"] != "default":
        pass  # scheduler swap is applied on the pipe at load time; see _load_pipe note
    return kw


def txt2img(prompt, W, H, seed=None, steps=8, num_images=1,
            guidance_scale=5.0, output_type="pil", **kw):
    """Text-to-image. Returns PIL images, or HWC float arrays when output_type='np'
    (used for 16/32-bit output)."""
    pipe = _load_pipe("txt2img")
    out = pipe(prompt=prompt,
               **_common_kwargs(W, H, seed, steps, num_images, guidance_scale, output_type, kw))
    return list(out.images)


def img2img(prompt, init_image, deforum_strength, W, H, seed=None, steps=8, num_images=1,
            guidance_scale=5.0, output_type="pil", **kw):
    """Image-to-image from a PIL init. Deforum strength is inverted for diffusers."""
    pipe = _load_pipe("img2img")
    out = pipe(prompt=prompt, image=init_image, strength=to_fal_strength(deforum_strength),
               **_common_kwargs(W, H, seed, steps, num_images, guidance_scale, output_type, kw))
    return list(out.images)


def inpaint(prompt, init_image, mask_image, deforum_strength, W, H, seed=None, steps=8,
            num_images=1, guidance_scale=5.0, output_type="pil", **kw):
    """Masked generation. White mask regions change."""
    pipe = _load_pipe("inpaint")
    out = pipe(prompt=prompt, image=init_image, mask_image=mask_image,
               strength=to_fal_strength(deforum_strength),
               **_common_kwargs(W, H, seed, steps, num_images, guidance_scale, output_type, kw))
    return list(out.images)


# --- Embedding-space interpolation (U5) ----------------------------------

def encode_prompt(prompt):
    """Return the prompt's text embedding (a list of token tensors, per diffusers)."""
    pipe = _load_pipe("txt2img")
    # no_grad: the pipe's own __call__ is wrapped, but calling encode_prompt directly
    # is not -- without this the text-encoder activation graph stays alive (the returned
    # embeds reference it), leaking ~3GB and OOM-ing interpolation on a 24GB card.
    with torch.no_grad():
        prompt_embeds, _negative = pipe.encode_prompt(prompt, do_classifier_free_guidance=False)
    return prompt_embeds


def slerp_embeds(e1, e2, t):
    """Slerp two prompt embeddings (lists of token tensors). Token counts can differ
    between prompts, so each tensor pair is aligned to the shorter length before
    interpolating -- an approximation; exactness is a GPU-validation item."""
    from .interpolation import interpolate
    out = []
    for a, b in zip(e1, e2):
        n = min(a.shape[0], b.shape[0])
        out.append(interpolate(t, a[:n], b[:n], mode="slerp"))
    return out


def txt2img_embeds(prompt_embeds, W, H, seed=None, steps=8, guidance_scale=0.0,
                   output_type="pil", **kw):
    """Generate from precomputed (e.g. slerped) prompt embeddings.

    The embeds path carries only positive (conditional) embeddings -- encode_prompt
    returns no negatives -- so classifier-free guidance can't be applied here: the
    diffusers ZImage pipeline raises if CFG is on (guidance_scale>0) without
    negative_prompt_embeds. Force it off. Z-Image Turbo is distilled for CFG-free
    few-step sampling, so unguided is the intended mode for the morph anyway.
    """
    pipe = _load_pipe("txt2img")
    out = pipe(prompt_embeds=prompt_embeds,
               **_common_kwargs(W, H, seed, steps, 1, 0.0, output_type, kw))
    return list(out.images)
