"""Krea 2 Turbo (fal.ai) client wrapper.

Hosted Krea 2 Turbo via fal.ai's ``fal-ai/krea-2/turbo`` endpoint. Same surface as
helpers.zimage_client (txt2img -> list of PIL images) so helpers.generate routes to
it uniformly when model="krea2", backend="fal".

Krea 2 Turbo is **text-to-image only** on fal -- there is no Krea 2 image-to-image or
inpaint endpoint (only the older FLUX-Krea has img2img). So img2img/inpaint raise with
an actionable message: use the local backend for Krea 2 animation, or the Z-Image model
for hosted img2img/inpaint.

Torch-free: reuses the model-agnostic plumbing (auth, retry, result download, step
clamp, image_size) from zimage_client; the fal path never imports torch.

Auth is via the FAL_KEY environment variable (or a local .env), same as Z-Image.
"""
from .zimage_client import (
    clamp_steps,
    image_size_arg,
    resolve_fal_key,  # noqa: F401  (kept importable for parity; _submit resolves it too)
    _result_to_images,
    _submit,
)

# fal.ai endpoint identifier (text-to-image; the distilled "turbo" checkpoint).
ENDPOINT_TXT2IMG = "fal-ai/krea-2/turbo"

# Krea 2 Turbo is distilled to the same small step budget as Z-Image Turbo.
MAX_STEPS = 8
MIN_STEPS = 1

_IMG2IMG_MSG = (
    "Krea 2 Turbo on fal.ai (fal-ai/krea-2/turbo) is text-to-image only -- fal has no "
    "Krea 2 image-to-image endpoint. For Krea 2 image-to-image / animation use the local "
    "backend (backend='local'), or use the Z-Image model for hosted img2img."
)
_INPAINT_MSG = (
    "Krea 2 Turbo on fal.ai is text-to-image only (no inpaint endpoint). Use the Z-Image "
    "model for hosted masked generation."
)


def _arguments(prompt, W, H, seed, steps, num_images):
    # NOTE: built independently of zimage_client._base_arguments, which injects the
    # Z-Image-only `acceleration` param. The fal Krea 2 turbo input schema (image_size vs
    # width/height, step range) should be confirmed at
    # https://fal.ai/models/fal-ai/krea-2/turbo/api -- this mirrors the z-image shape.
    args = {
        "prompt": prompt,
        "image_size": image_size_arg(W, H),
        "num_inference_steps": clamp_steps(steps),
        "num_images": int(num_images),
        "output_format": "png",
    }
    if seed is not None:
        args["seed"] = int(seed)
    return args


def txt2img(prompt, W, H, seed=None, steps=MAX_STEPS, num_images=1, **_ignored):
    """Text-to-image. Returns a list of PIL images (length == num_images).

    Krea 2 Turbo is CFG-free; no guidance_scale is sent. Accepts and ignores
    local/Z-Image-only kwargs (guidance_scale, acceleration, ...) so generate() can
    pass one common kwarg set to either backend."""
    arguments = _arguments(prompt, W, H, seed, steps, num_images)
    return _result_to_images(_submit(ENDPOINT_TXT2IMG, arguments))


def img2img(*_args, **_kwargs):
    raise NotImplementedError(_IMG2IMG_MSG)


def inpaint(*_args, **_kwargs):
    raise NotImplementedError(_INPAINT_MSG)
