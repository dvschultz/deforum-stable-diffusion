"""U3 (16/32-bit) + U4 (per-step callback/thresholding) for the local backend.
diffusers is mocked; torch is available in the dev env."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from helpers import generate as gen
from helpers import zimage_local as zl


def _args(tmp_path, **over):
    d = dict(cond_prompt="x", outdir=str(tmp_path), use_init=False, strength=0.0,
             strength_0_no_init=True, n_samples=1, steps=8, seed=1, acceleration="regular",
             guidance_scale=5.0, backend="local", bit_depth_output=8,
             init_sample=None, init_image=None, use_alpha_as_mask=False, use_mask=False,
             mask_file=None, invert_mask=False, overlay_mask=False, mask_overlay_blur=0,
             W=8, H=8, timestring="t")
    d.update(over)
    return SimpleNamespace(**d)


# --- U3: 16/32-bit -------------------------------------------------------

def test_generate_16bit_returns_uint16(tmp_path, monkeypatch):
    # local backend returns HWC float [0,1] np arrays when output_type='np'
    monkeypatch.setattr(zl, "txt2img", lambda *a, **k: [np.ones((8, 8, 3), dtype=np.float32)])
    out = gen.generate(_args(tmp_path, bit_depth_output=16), root=SimpleNamespace(backend="local"))
    assert out[0].dtype == np.uint16
    assert out[0].max() == 65535


def test_generate_32bit_returns_float32(tmp_path, monkeypatch):
    monkeypatch.setattr(zl, "txt2img", lambda *a, **k: [np.full((8, 8, 3), 0.5, dtype=np.float32)])
    out = gen.generate(_args(tmp_path, bit_depth_output=32), root=SimpleNamespace(backend="local"))
    assert out[0].dtype == np.float32 and out[0].max() <= 1.0


def test_generate_16bit_return_sample_is_minus1_to_1(tmp_path, monkeypatch):
    monkeypatch.setattr(zl, "txt2img", lambda *a, **k: [np.ones((8, 8, 3), dtype=np.float32)])
    out = gen.generate(_args(tmp_path, bit_depth_output=16),
                       root=SimpleNamespace(backend="local"), return_sample=True)
    sample, image = out
    assert sample.shape == (1, 3, 8, 8) and sample.dtype == np.float32
    assert image.dtype == np.uint16


def test_fal_ignores_16bit_request(tmp_path, monkeypatch):
    # backend=fal must NOT produce high-bit output (no local VAE); stays 8-bit PIL
    from helpers import zimage_client
    monkeypatch.setattr(zimage_client, "txt2img", lambda *a, **k: [Image.new("RGB", (8, 8))])
    out = gen.generate(_args(tmp_path, backend="fal", bit_depth_output=16), root=SimpleNamespace(backend="fal"))
    assert isinstance(out[0], Image.Image)  # fal path unaffected


# --- U4: per-step callback / thresholding --------------------------------

def test_static_threshold_clamps_latents():
    cb = zl._make_step_callback(static_threshold=0.5)
    assert cb is not None
    lat = torch.tensor([[-2.0, 0.1, 2.0]])
    out = cb(pipe=None, step=0, timestep=0, cbk={"latents": lat})
    assert float(out["latents"].max()) <= 0.5 and float(out["latents"].min()) >= -0.5


def test_dynamic_threshold_returns_latents():
    cb = zl._make_step_callback(dynamic_threshold=90.0)
    out = cb(pipe=None, step=0, timestep=0, cbk={"latents": torch.randn(1, 4, 4, 4)})
    assert "latents" in out


def test_no_callback_when_no_knobs():
    assert zl._make_step_callback() is None
    assert zl._make_step_callback(dynamic_threshold=None, static_threshold=None) is None


def test_local_passes_callback_when_thresholding(tmp_path, monkeypatch):
    captured = {}

    def fake_pipe(**kw):
        captured.update(kw)
        return SimpleNamespace(images=[Image.new("RGB", (8, 8))])

    monkeypatch.setattr(zl, "_load_pipe", lambda kind: fake_pipe)
    zl.txt2img("p", 8, 8, static_threshold=0.5)
    assert "callback_on_step_end" in captured  # callback wired into the pipe call


# --- U5: embedding-slerp interpolation ----------------------------------

def test_slerp_embeds_aligns_to_min_token_length():
    e1 = [torch.randn(3, 4)]
    e2 = [torch.randn(5, 4)]
    out = zl.slerp_embeds(e1, e2, 0.5)
    assert out[0].shape == (3, 4)  # aligned to the shorter prompt


def test_slerp_embeds_endpoints():
    e1 = [torch.ones(2, 4)]
    e2 = [torch.ones(2, 4) * 3]
    near0 = zl.slerp_embeds(e1, e2, 0.0)[0]
    near1 = zl.slerp_embeds(e1, e2, 1.0)[0]
    assert torch.allclose(near0, e1[0], atol=1e-4)
    assert torch.allclose(near1, e2[0], atol=1e-4)


def test_txt2img_embeds_passes_prompt_embeds(monkeypatch):
    captured = {}

    def fake_pipe(**kw):
        captured.update(kw)
        return SimpleNamespace(images=[Image.new("RGB", (8, 8))])

    monkeypatch.setattr(zl, "_load_pipe", lambda kind: fake_pipe)
    embeds = [torch.randn(2, 4)]
    out = zl.txt2img_embeds(embeds, 8, 8, seed=1)
    assert captured["prompt_embeds"] is embeds and len(out) == 1
