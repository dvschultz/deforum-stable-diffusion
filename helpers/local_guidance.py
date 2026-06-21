"""EXPERIMENTAL gradient conditioning guidance for the local Z-Image (DiT) backend.

Reimplements the old SD CFGDenoiserWithGrad idea against flow-matching latents:
inside the diffusers callback_on_step_end, differentiably decode the current
latent, compute the reused conditioning losses (helpers.conditioning), backprop to
a latent gradient, and nudge the latent toward lower loss.

EXPERIMENTAL: backprop guidance on an 8-step distilled DiT may barely move the
image (it may need many more steps to matter). All conditioning scales default to
0 (off). The core nudge math (`apply_latent_guidance`) is unit-tested; whether it
meaningfully steers Z-Image Turbo is a GPU-machine validation item.

Only imported on the local backend (pulls torch + helpers.conditioning, which
pulls CLIP). The fal path never touches this module.
"""
import torch


def apply_latent_guidance(latents, decode_fn, loss_fn, scale=1.0, clamp=None):
    """Return latents nudged to reduce ``loss_fn(decode_fn(latents))``.

    The decoupled, testable core: with a mock identity ``decode_fn`` and a simple
    ``loss_fn`` (e.g. mean), the returned latents move in the loss-reducing
    direction. ``clamp`` bounds the per-element gradient (mirrors the old grad clamp).
    """
    work = latents.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        image = decode_fn(work)
        loss = loss_fn(image) * float(scale)
        grad = torch.autograd.grad(loss, work)[0]
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    if clamp:
        grad = grad.clamp(-float(clamp), float(clamp))
    return latents.detach() - grad


def _ensure_guidance_models(args, root):
    """Load the models CLIP/aesthetic guidance needs onto ``root`` -- only when those scales
    are set, and only once. Without this ``root.clip_model`` is never populated on the local
    path, so build_guidance_loss silently drops the CLIP/aesthetic losses (they no-op). The
    model-free losses (blue/mean/var/exposure) need nothing here."""
    wants_clip = bool(getattr(args, "clip_scale", 0)) or bool(getattr(args, "aesthetics_scale", 0))
    if wants_clip and getattr(root, "clip_model", None) is None:
        import clip  # vendored (src/clip); also imported by helpers.conditioning
        name = getattr(args, "clip_name", "ViT-L/14")
        root.clip_model = clip.load(name, jit=False, device=root.device)[0].eval().requires_grad_(False)
    if getattr(args, "aesthetics_scale", 0) and getattr(root, "aesthetics_model", None) is None:
        from .aesthetics import load_aesthetics_model
        root.aesthetics_model = load_aesthetics_model(args, root).eval().requires_grad_(False)
    # CLIP guidance target text: default to the generation prompt when none is given.
    if getattr(args, "clip_scale", 0) and not getattr(args, "clip_prompt", None):
        args.clip_prompt = [getattr(args, "cond_prompt", "") or ""]


def build_guidance_loss(args, root):
    """Combine the active conditioning losses into one image->scalar loss, or None.

    Reuses helpers.conditioning loss functions (the same ones the SD path used).
    Model-free losses (blue/mean/var/exposure) always work; CLIP/aesthetics losses
    are included only when their models are present on ``root`` (loaded when those
    scales are enabled). EXPERIMENTAL.
    """
    from . import conditioning as C

    pairs = []  # (loss_fn(image, sigma), scale)
    if getattr(args, "blue_scale", 0):
        pairs.append((C.blue_loss_fn, args.blue_scale))
    if getattr(args, "mean_scale", 0):
        pairs.append((C.mean_loss_fn, args.mean_scale))
    if getattr(args, "var_scale", 0):
        pairs.append((C.var_loss_fn, args.var_scale))
    if getattr(args, "exposure_scale", 0):
        pairs.append((C.exposure_loss(args.exposure_target), args.exposure_scale))
    if getattr(args, "clip_scale", 0) and getattr(root, "clip_model", None) is not None:
        pairs.append((C.make_clip_loss_fn(root, args), args.clip_scale))
    if getattr(args, "aesthetics_scale", 0) and getattr(root, "aesthetics_model", None) is not None:
        pairs.append((C.make_aesthetics_loss_fn(root, args), args.aesthetics_scale))

    if not pairs:
        return None

    def loss_fn(image):
        total = image.new_zeros(())
        for fn, scale in pairs:
            total = total + fn(image, None) * float(scale)
        return total

    return loss_fn


def make_guidance_callback(args, root):
    """Build a callback_on_step_end that applies gradient guidance, or None when no
    conditioning scale is set. The callback uses the pipe (its first arg) to decode
    latents, so it needs nothing from generate() beyond args/root."""
    _ensure_guidance_models(args, root)  # load CLIP/aesthetics onto root if those scales are set
    loss_fn = build_guidance_loss(args, root)
    if loss_fn is None:
        return None

    clamp = getattr(args, "clamp_grad_threshold", None)
    timing = getattr(args, "grad_inject_timing", None)  # 1-based steps to inject on

    def callback(pipe, step, timestep, cbk):
        if timing and (step + 1) not in timing:
            return {}
        latents = cbk["latents"]

        def decode_fn(lat):
            vae = pipe.vae
            cfg = getattr(vae, "config", None)
            sf = getattr(cfg, "scaling_factor", 1.0) if cfg is not None else 1.0
            shift = getattr(cfg, "shift_factor", 0.0) if cfg is not None else 0.0
            # Denoising keeps latents in fp32 while the VAE is bf16; the differentiable
            # .to(vae.dtype) cast keeps `lat` (requires_grad) in the autograd graph. Also
            # apply shift_factor so the decode matches the pipeline's own normalization.
            lat = lat.to(vae.dtype)
            return vae.decode(lat / sf + shift, return_dict=False)[0]

        return {"latents": apply_latent_guidance(latents, decode_fn, loss_fn, scale=1.0, clamp=clamp)}

    return callback
