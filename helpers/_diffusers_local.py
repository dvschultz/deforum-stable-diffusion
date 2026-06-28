"""Shared core for the local diffusers backends (Z-Image, Krea 2).

The generic, model-agnostic machinery that the per-model adapters
(``helpers.zimage_local``, ``helpers.krea2_local``) bind to via a small
:class:`ModelSpec`: device/dtype selection, optional bitsandbytes weight
quantization, lazy pipeline loading with cross-kind component sharing, per-step
thresholding/preview callbacks, and the common ``__call__`` kwargs.

Only the bits that genuinely differ per model live in the adapters: the model id
and pipeline classes, the default step/guidance budget, the VAE latent
normalization family, and the text-embedding / interpolation surface (plus Krea 2's
custom img2img). Everything here is shared so the two backends don't duplicate the
quant + component-sharing + callback code (the riskiest, most subtle part).

Imports torch (and lazily diffusers); imported ONLY by the local adapters, which
are themselves imported lazily by helpers.backends.resolve_backend. The fal path
never touches this module, preserving its torch-free property.
"""
import os
from dataclasses import dataclass

import torch


@dataclass
class ModelSpec:
    """Per-model configuration the shared core needs.

    name:            short id ("z-image" | "krea2"), for messages and cache keys.
    model_id:        HF repo id or local path (already resolved from the env override).
    pipe_classes:    {"txt2img":..., "img2img":..., "inpaint":...} diffusers class names;
                     a None value marks that kind unsupported for this model.
    default_steps:   distilled step budget (8).
    default_guidance:default CFG scale (z-image 5.0; krea2 0.0 / CFG-free).
    vae_norm:        latent<->image normalization family for per-step previews:
                     "shift_scale" (Z-Image: lat/scaling + shift) or
                     "mean_std" (Qwen video VAE: packed latents, not previewed yet).
    quant_env_vars:  env vars consulted (in order) for ZIMAGE_QUANTIZE-style quant.
    """
    name: str
    model_id: str
    pipe_classes: dict
    default_steps: int = 8
    default_guidance: float = 0.0
    vae_norm: str = "shift_scale"
    quant_env_vars: tuple = ("ZIMAGE_QUANTIZE",)


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def generator(seed):
    if seed is None:
        return None
    return torch.Generator(device=device()).manual_seed(int(seed))


def _quant_mode(spec):
    """Return (mode, var_name) for the first set quant env var, or (None, None)."""
    for var in spec.quant_env_vars:
        val = os.environ.get(var, "").strip().lower()
        if val:
            return val, var
    return None, None


def quant_config(spec):
    """Optional in-loader weight quantization, OFF by default (full bf16, fastest). Opt in
    with the model's quant env var (e.g. ``ZIMAGE_QUANTIZE``): ``int8`` (~half resident,
    ~bf16 quality, but ~1.5x slower per image on Ampere) or ``nf4`` (~quarter resident,
    near-bf16 speed, lower fidelity to bf16). Quantizes the transformer AND the large text
    encoder via bitsandbytes. CUDA only.

    On Ampere (A5000, sm_86) there's no native int8/fp8 *compute*, so matmuls dequantize to
    bf16 -- quantization is a memory lever, not a speed one. For a 6B model (Z-Image) it's
    rarely needed; for a 12.9B model (Krea 2) nf4 is effectively required to fit a single
    24GB card. Falls back to bf16 (with a note) if bitsandbytes is missing."""
    mode, var = _quant_mode(spec)
    if mode in (None, "", "none", "off", "no", "bf16", "0"):
        return None
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        print(f"..{var}={mode} but bitsandbytes is unavailable; loading in bf16. "
              f"Install bitsandbytes (bundled in --with-local) or set {var}=bf16.")
        return None
    from diffusers import PipelineQuantizationConfig
    comps = ["transformer", "text_encoder"]
    if mode in ("int8", "8bit", "8"):
        return PipelineQuantizationConfig(
            quant_backend="bitsandbytes_8bit",
            quant_kwargs={"load_in_8bit": True},
            components_to_quantize=comps)
    if mode in ("int4", "nf4", "4bit", "4"):
        return PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                          "bnb_4bit_compute_dtype": torch.bfloat16},
            components_to_quantize=comps)
    raise ValueError(f"{var}={mode!r} not recognized (use 'int8', 'nf4', or unset).")


def load_pipe(spec, kind, cache):
    """Load and cache a diffusers pipeline of the given kind for ``spec``.

    ``cache`` is the adapter's own dict (keyed by kind) -- each model keeps a separate
    cache, so two models loaded in one process never alias each other's pipelines.
    """
    if kind in cache:
        return cache[kind]
    cls_name = spec.pipe_classes.get(kind)
    if cls_name is None:
        raise NotImplementedError(
            f"{kind} is not supported for the {spec.name} model "
            "(no corresponding diffusers pipeline)."
        )
    try:
        import diffusers
    except ImportError as e:
        raise RuntimeError(
            "diffusers is required for the local backend. Install with "
            "`python install_requirements.py --with-local` "
            "(pulls torch + diffusers-from-source)."
        ) from e
    Pipe = getattr(diffusers, cls_name, None)
    if Pipe is None:
        raise RuntimeError(
            f"{cls_name} not found in your diffusers install. The {spec.name} pipelines "
            "require a current diffusers from source: pip install "
            "git+https://github.com/huggingface/diffusers.git"
        )
    # Share weights across kinds: txt2img/img2img/inpaint are separate pipeline classes but
    # use the same transformer/text_encoder/vae, so build later kinds from an already-loaded
    # one's components -- same module objects, no extra VRAM. Without this a 2D animation
    # (txt2img frame 0 + img2img rest) would hold two large copies and OOM on 24GB.
    # (We construct directly rather than via from_pipe, which re-runs .to(dtype) over all
    # the modules and spikes memory enough to OOM at the 24GB edge.)
    if cache:
        import inspect
        base = next(iter(cache.values()))
        expected = set(inspect.signature(Pipe.__init__).parameters) - {"self"}
        pipe = Pipe(**{k: v for k, v in base.components.items() if k in expected})
        cache[kind] = pipe
        return pipe
    dtype = torch.bfloat16 if device() == "cuda" else torch.float32
    quant = quant_config(spec) if device() == "cuda" else None
    if quant is not None:
        # bitsandbytes places the quantized weights on the GPU itself, and calling .to() on
        # a quantized model raises -- so move only the non-quantized components (the VAE) to
        # the device. Cast them to the compute dtype too: otherwise the VAE stays fp32 and
        # img2img/inpaint's vae.encode(bf16 image) hits a dtype mismatch.
        pipe = Pipe.from_pretrained(spec.model_id, torch_dtype=dtype, quantization_config=quant)
        for comp in pipe.components.values():
            if isinstance(comp, torch.nn.Module) and not (
                    getattr(comp, "is_loaded_in_8bit", False)
                    or getattr(comp, "is_loaded_in_4bit", False)):
                comp.to(device(), dtype=dtype)
    else:
        pipe = Pipe.from_pretrained(spec.model_id, torch_dtype=dtype).to(device())
    cache[kind] = pipe
    return pipe


def apply_dynamic_threshold(latents, percentile):
    # Imagen-style per-step clamp: clamp to the given abs-value percentile, then rescale.
    import numpy as np
    s = np.percentile(latents.detach().abs().cpu().numpy(), percentile,
                      axis=tuple(range(1, latents.ndim)))
    s = np.maximum(s, 1.0)
    s = torch.as_tensor(s, device=latents.device, dtype=latents.dtype).view(-1, *([1] * (latents.ndim - 1)))
    return latents.clamp(-s, s) / s


def _decode_preview(spec, pipe, latents):
    """Decode current latents to a PIL preview, mirroring the pipeline's own normalization.

    "shift_scale" (Z-Image): latents are (B,C,H,W); cast to VAE dtype, unscale, unshift.
    "mean_std" (Krea 2 / Qwen video VAE): latents are packed (B, seq, in_channels) and the
    VAE is 5D -- per-step previews aren't wired for that layout yet, so return None (the
    caller treats previews as best-effort).
    """
    if spec.vae_norm != "shift_scale":
        return None
    lat = latents.to(pipe.vae.dtype)
    lat = (lat / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    return pipe.image_processor.postprocess(
        pipe.vae.decode(lat, return_dict=False)[0], output_type="pil")[0]


def make_step_callback(spec, dynamic_threshold=None, static_threshold=None,
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
            latents = apply_dynamic_threshold(latents, dynamic_threshold)
        if save_sample_per_step and outdir:
            try:
                img = _decode_preview(spec, pipe, latents)
                if img is not None:
                    img.save(os.path.join(outdir, f"{timestring}_step_{step:03d}.png"))
            except Exception:
                pass  # previews are best-effort; never break the render
        return {"latents": latents}

    return callback


def compose_callbacks(callbacks):
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


def common_kwargs(spec, W, H, seed, steps, num_images, guidance_scale, output_type, kw_extra):
    """Assemble the shared diffusers ``__call__`` kwargs (size, steps, guidance, generator,
    output_type) plus the composed per-step callback (thresholding/preview + optional
    experimental gradient guidance)."""
    kw = dict(
        height=int(H), width=int(W),
        num_inference_steps=int(steps),
        num_images_per_prompt=int(num_images),
        guidance_scale=float(guidance_scale),
        generator=generator(seed),
        output_type=output_type,
    )
    callback = compose_callbacks([
        make_step_callback(spec, **kw_extra),
        kw_extra.get("guidance_callback"),
    ])
    if callback is not None:
        kw["callback_on_step_end"] = callback
    return kw
