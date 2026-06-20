"""Frame generation adapter.

Historically this ran the in-process Stable Diffusion pipeline (encode -> sample
-> decode). It now routes to the hosted Z-Image Turbo model via
``helpers.zimage_client``. The function signature and return-value contract are
preserved so the animation engine above it (``helpers/render.py``) is unchanged:

  - returns a ``results`` list of PIL images (one per ``n_samples``)
  - when ``return_sample=True``, prepends a ``[-1,1]`` sample tensor that the
    render loop warps into the next frame's init image

The render loop hands us a pixel init image (a warped, color-matched, noised
previous frame) plus a Deforum ``strength``; we choose the endpoint by request
shape and call the client. There is no latent space and no per-step callback.
"""

# Standard library imports
import os

# Related third-party imports
import numpy as np
from PIL import Image, ImageOps, ImageFilter

# Local application/library specific imports
from .animation import sample_from_cv2, sample_to_cv2
from . import zimage_client as zc
from .backends import resolve_backend, resolve_backend_name


def add_noise(sample: np.ndarray, noise_amt: float) -> np.ndarray:
    return sample + np.random.randn(*sample.shape).astype(sample.dtype) * noise_amt


def _sample_to_pil(sample: np.ndarray) -> Image.Image:
    """Convert a ``[-1,1]`` sample array ([1,3,H,W] or [3,H,W]) to a PIL RGB image."""
    arr = sample_to_cv2(sample, type=np.uint8)  # HWC uint8 RGB
    return Image.fromarray(arr)


def _pil_to_sample(pil: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to a ``[-1,1]`` float32 array ([1,3,H,W]).

    Inverse of ``_sample_to_pil``; the "sample" is the pixel buffer the render loop
    warps into the next frame's init image (no longer a latent/tensor).
    """
    return sample_from_cv2(np.array(pil.convert("RGB")))


def _load_init_pil(args) -> Image.Image:
    """Return a PIL init image (resized to W x H) from init_sample or init_image, else None."""
    if getattr(args, "init_sample", None) is not None:
        return _sample_to_pil(args.init_sample)
    if args.use_init and getattr(args, "init_image", None):
        from .load_images import load_img
        img_t, _ = load_img(
            args.init_image,
            shape=(args.W, args.H),
            use_alpha_as_mask=args.use_alpha_as_mask,
        )
        return _sample_to_pil(img_t)
    return None


def _load_mask_pil(args) -> Image.Image:
    """Return a PIL 'L' mask (white = regions that change), resized to W x H, else None.

    Prefers ``args.mask_sample`` when present: the render loop warps the mask each
    frame so the changeable region tracks 2D/3D motion. Falls back to the static
    ``args.mask_file`` otherwise.
    """
    if getattr(args, "mask_sample", None) is not None:
        mask = _sample_to_pil(args.mask_sample).convert("L")
    else:
        mask_src = getattr(args, "mask_file", None)
        if not mask_src:
            return None
        from .load_images import load_mask_latent
        mask = load_mask_latent(mask_src, (1, 1, args.H, args.W)).convert("L")
    if mask.size != (args.W, args.H):
        mask = mask.resize((args.W, args.H), Image.LANCZOS)
    if getattr(args, "invert_mask", False):
        mask = ImageOps.invert(mask)
    return mask


def _apply_overlay_mask(generated: Image.Image, init_pil: Image.Image,
                        mask_pil: Image.Image, args) -> Image.Image:
    """Composite the original init back into the unmasked regions (pixel-space).

    Mirrors the prior latent overlay so masked areas (white) take the generated
    result and the rest is preserved, without degrading the unchanged regions.
    """
    blur = getattr(args, "mask_overlay_blur", 0) or 0
    soft_mask = mask_pil.filter(ImageFilter.GaussianBlur(blur)) if blur > 0 else mask_pil
    return Image.composite(generated.convert("RGB"), init_pil.convert("RGB"), soft_mask)


def generate(args, root, frame=0, return_latent=False, return_sample=False, return_c=False):
    if return_latent or return_c:
        raise NotImplementedError(
            "return_latent / return_c are unavailable with the Z-Image Turbo backend "
            "(no access to latents or text embeddings). Interpolation now uses "
            "pixel-space blending instead -- see helpers/render.py:render_interpolation."
        )

    os.makedirs(args.outdir, exist_ok=True)

    prompt = args.cond_prompt
    assert prompt is not None, "generate requires args.cond_prompt"

    n_samples = int(getattr(args, "n_samples", 1) or 1)
    steps = getattr(args, "steps", zc.MAX_STEPS)
    seed = getattr(args, "seed", None)

    # Backend resolution (fal default, local opt-in). Each backend takes the common
    # kwargs and ignores the ones it doesn't use (acceleration is fal-only;
    # guidance_scale / thresholds / per-step previews are local-only).
    backend = resolve_backend(root)
    is_local = resolve_backend_name(root) == "local"
    bit = int(getattr(args, "bit_depth_output", 8) or 8)
    highbit = is_local and bit in (16, 32)  # true 16/32-bit only on local (VAE decode)

    common = dict(
        seed=seed, steps=steps, num_images=n_samples,
        acceleration=getattr(args, "acceleration", "regular"),
        guidance_scale=getattr(args, "guidance_scale", 5.0),
        output_type=("np" if highbit else "pil"),
    )
    if is_local:
        common.update(
            dynamic_threshold=getattr(args, "dynamic_threshold", None),
            static_threshold=getattr(args, "static_threshold", None),
            save_sample_per_step=getattr(args, "save_sample_per_step", False),
            show_sample_per_step=getattr(args, "show_sample_per_step", False),
            outdir=args.outdir, timestring=getattr(args, "timestring", ""),
            scheduler=getattr(args, "scheduler", "default"),
        )
        # EXPERIMENTAL gradient conditioning guidance (U7), only when a scale is set.
        _GUIDANCE_SCALES = ("clip_scale", "aesthetics_scale", "colormatch_scale",
                            "init_mse_scale", "blue_scale", "mean_scale", "var_scale",
                            "exposure_scale")
        if any(getattr(args, s, 0) for s in _GUIDANCE_SCALES):
            from .local_guidance import make_guidance_callback  # lazy: pulls torch + CLIP
            common["guidance_callback"] = make_guidance_callback(args, root)

    init_pil = _load_init_pil(args)
    has_init = init_pil is not None

    # No init image, but strength > 0: auto-zero strength (mirrors prior behavior).
    if not has_init and args.strength > 0 and args.strength_0_no_init:
        args.strength = 0

    mask_pil = _load_mask_pil(args) if (getattr(args, "use_mask", False) and has_init) else None

    # Route by request shape (KTD-3).
    if mask_pil is not None:
        images = backend.inpaint(prompt, init_pil, mask_pil, args.strength, args.W, args.H, **common)
    elif has_init and args.strength > 0:
        images = backend.img2img(prompt, init_pil, args.strength, args.W, args.H, **common)
    else:
        images = backend.txt2img(prompt, args.W, args.H, **common)

    # Fail loudly if the backend returned nothing (quota/content-filter/schema change),
    # rather than crashing later in np.concatenate or the render loop's tuple unpack.
    if not images:
        raise RuntimeError(
            f"Z-Image returned no images (prompt={prompt!r}). "
            "Check your backend (fal.ai quota, or local pipeline/weights)."
        )

    if highbit:
        return _format_highbit_results(images, args, bit, return_sample)

    # image_size is a request hint, not a guarantee: normalize every result to the
    # requested canvas so frame warps, batching (np.concatenate), and ffmpeg
    # assembly never see drifting dimensions.
    images = [im if im.size == (args.W, args.H) else im.resize((args.W, args.H), Image.LANCZOS)
              for im in images]

    # Optional pixel-space overlay: preserve unmasked regions of the init.
    if mask_pil is not None and getattr(args, "overlay_mask", False):
        images = [_apply_overlay_mask(im, init_pil, mask_pil, args) for im in images]

    results = []
    if return_sample:
        # Batched [B,3,H,W] sample array for the render loop's next-frame warp.
        results.append(np.concatenate([_pil_to_sample(im) for im in images], axis=0))
    results.extend(images)
    return results


def _format_highbit_results(images, args, bit, return_sample):
    """Local backend, 16/32-bit: backend returns HWC float [0,1] arrays. Produce the
    bit-depth arrays render's save_8_16_or_32bpc_image expects, plus an 8-bit-derived
    sample buffer for the warp loop (warping doesn't need the extra bit depth)."""
    norm = []
    for a in images:
        a = np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0)
        if a.shape[:2] != (args.H, args.W):
            pil = Image.fromarray((a * 255).astype(np.uint8)).resize((args.W, args.H), Image.LANCZOS)
            a = np.asarray(pil, dtype=np.float32) / 255.0
        norm.append(a)

    results = []
    if return_sample:
        samples = [_pil_to_sample(Image.fromarray((a * 255).astype(np.uint8))) for a in norm]
        results.append(np.concatenate(samples, axis=0))
    for a in norm:
        if bit == 16:
            results.append((a * 65535.0).astype(np.uint16))
        else:  # 32-bit float (EXR)
            results.append(a.astype(np.float32))
    return results
