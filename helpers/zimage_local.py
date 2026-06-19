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


def _common_kwargs(W, H, seed, steps, num_images, guidance_scale, output_type, callback):
    kw = dict(
        height=int(H), width=int(W),
        num_inference_steps=int(steps),
        num_images_per_prompt=int(num_images),
        guidance_scale=float(guidance_scale),
        generator=_generator(seed),
        output_type=output_type,
    )
    if callback is not None:
        kw["callback_on_step_end"] = callback
    return kw


def txt2img(prompt, W, H, seed=None, steps=8, num_images=1,
            guidance_scale=5.0, output_type="pil", callback=None, **_ignored):
    """Text-to-image. Returns a list of PIL images (or float tensors if output_type='pt')."""
    pipe = _load_pipe("txt2img")
    out = pipe(prompt=prompt,
               **_common_kwargs(W, H, seed, steps, num_images, guidance_scale, output_type, callback))
    return list(out.images)


def img2img(prompt, init_image, deforum_strength, W, H, seed=None, steps=8, num_images=1,
            guidance_scale=5.0, output_type="pil", callback=None, **_ignored):
    """Image-to-image from a PIL init. Deforum strength is inverted for diffusers."""
    pipe = _load_pipe("img2img")
    out = pipe(prompt=prompt, image=init_image, strength=to_fal_strength(deforum_strength),
               **_common_kwargs(W, H, seed, steps, num_images, guidance_scale, output_type, callback))
    return list(out.images)


def inpaint(prompt, init_image, mask_image, deforum_strength, W, H, seed=None, steps=8,
            num_images=1, guidance_scale=5.0, output_type="pil", callback=None, **_ignored):
    """Masked generation. White mask regions change."""
    pipe = _load_pipe("inpaint")
    out = pipe(prompt=prompt, image=init_image, mask_image=mask_image,
               strength=to_fal_strength(deforum_strength),
               **_common_kwargs(W, H, seed, steps, num_images, guidance_scale, output_type, callback))
    return list(out.images)
