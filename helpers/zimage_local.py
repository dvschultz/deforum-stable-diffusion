"""Local Z-Image backend via the diffusers ZImage pipelines.

Mirrors the helpers.zimage_client surface (txt2img / img2img / inpaint -> list of
PIL images) so helpers.generate routes to either backend uniformly. This module
imports torch and (lazily) diffusers, so it must only be imported when the local
backend is selected -- helpers.backends.resolve_backend handles that.

The generic machinery (device/dtype, quantization, lazy loading with component
sharing, thresholding/preview callbacks, common kwargs) lives in
helpers._diffusers_local and is shared with the Krea 2 local backend; this module
is a thin adapter that binds the Z-Image :class:`ModelSpec` and keeps the public
function names the rest of the codebase (and the tests) reference.

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

from . import _diffusers_local as D
from ._diffusers_local import ModelSpec
from .zimage_client import to_fal_strength  # pure, torch-free; reused for strength inversion

# Weights: a HF id or a local path (set ZIMAGE_LOCAL_PATH to a downloaded dir).
MODEL_ID = os.environ.get("ZIMAGE_LOCAL_PATH", "Tongyi-MAI/Z-Image-Turbo")

_PIPE_CLASSES = {
    "txt2img": "ZImagePipeline",
    "img2img": "ZImageImg2ImgPipeline",
    "inpaint": "ZImageInpaintPipeline",
}

SPEC = ModelSpec(
    name="z-image",
    model_id=MODEL_ID,
    pipe_classes=_PIPE_CLASSES,
    default_steps=8,
    default_guidance=5.0,
    vae_norm="shift_scale",
    quant_env_vars=("ZIMAGE_QUANTIZE",),
)

_PIPES = {}  # kind -> loaded pipeline (cached); this model's own cache


# Thin module-level wrappers over the shared core. Kept as real module attributes
# (not bare re-exports) so tests can monkeypatch e.g. zimage_local._load_pipe.

def _load_pipe(kind):
    """Load and cache a ZImage pipeline of the given kind (with component sharing)."""
    return D.load_pipe(SPEC, kind, _PIPES)


def _make_step_callback(**kw):
    return D.make_step_callback(SPEC, **kw)


def _common_kwargs(W, H, seed, steps, num_images, guidance_scale, output_type, kw_extra):
    return D.common_kwargs(SPEC, W, H, seed, steps, num_images, guidance_scale, output_type, kw_extra)


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
