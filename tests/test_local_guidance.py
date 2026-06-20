"""U7: experimental gradient guidance. The grad-math core is unit-tested with a
mock decode + simple loss; real CLIP/aesthetic guidance on the DiT is GPU-validated."""
from types import SimpleNamespace

import torch

from helpers import local_guidance as lg


def test_apply_latent_guidance_reduces_loss():
    # identity decode + mean loss -> nudged latents must have lower mean
    latents = torch.randn(1, 4, 8, 8)
    decode = lambda x: x
    loss = lambda img: img.mean()
    nudged = lg.apply_latent_guidance(latents, decode, loss, scale=1.0)
    assert nudged.mean() < latents.mean()


def test_apply_latent_guidance_clamps_gradient():
    latents = torch.randn(1, 4, 4, 4)
    decode = lambda x: x * 1000.0  # large -> large grad
    loss = lambda img: img.sum()
    nudged = lg.apply_latent_guidance(latents, decode, loss, scale=1.0, clamp=0.01)
    delta = (latents - nudged).abs()
    assert float(delta.max()) <= 0.01 + 1e-6  # gradient clamp bounds the step


def test_apply_latent_guidance_nan_safe():
    latents = torch.zeros(1, 4, 4, 4, requires_grad=False)
    decode = lambda x: x
    loss = lambda img: img.sum() * float("nan")
    nudged = lg.apply_latent_guidance(latents, decode, loss, scale=1.0)
    assert torch.isfinite(nudged).all()  # NaN grad zeroed


def test_build_guidance_loss_none_when_all_scales_zero():
    args = SimpleNamespace(blue_scale=0, mean_scale=0, var_scale=0, exposure_scale=0,
                           clip_scale=0, aesthetics_scale=0)
    assert lg.build_guidance_loss(args, SimpleNamespace()) is None


def test_make_guidance_callback_none_when_off():
    args = SimpleNamespace(blue_scale=0, mean_scale=0, var_scale=0, exposure_scale=0,
                           clip_scale=0, aesthetics_scale=0)
    assert lg.make_guidance_callback(args, SimpleNamespace()) is None


def test_build_guidance_loss_active_with_model_free_scale():
    # a model-free loss (mean) with scale>0 yields a usable combined loss
    args = SimpleNamespace(blue_scale=0, mean_scale=1.0, var_scale=0, exposure_scale=0,
                           clip_scale=0, aesthetics_scale=0)
    loss_fn = lg.build_guidance_loss(args, SimpleNamespace())
    assert loss_fn is not None
    val = loss_fn(torch.ones(1, 3, 8, 8))
    assert torch.is_tensor(val)
