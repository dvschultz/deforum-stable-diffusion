"""Unit tests for the generate() adapter.

The zimage_client endpoint functions are mocked, so these assert routing and the
(sample, image) return contract without any network access.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from helpers import generate as gen


def make_args(tmp_path, **over):
    d = dict(
        cond_prompt="a cat", uncond_prompt="", outdir=str(tmp_path),
        use_init=False, strength=0.65, strength_0_no_init=True,
        n_samples=1, steps=50, seed=7, acceleration="regular",
        init_sample=None, init_image=None, use_alpha_as_mask=False,
        use_mask=False, mask_file=None, invert_mask=False,
        overlay_mask=False, mask_overlay_blur=0, W=8, H=8,
    )
    d.update(over)
    return SimpleNamespace(**d)


def _pil():
    return Image.new("RGB", (8, 8), (10, 20, 30))


def _init_sample():
    # [-1,1] array, [1,3,H,W]
    return np.zeros((1, 3, 8, 8), dtype=np.float32)


@pytest.fixture
def capture(monkeypatch):
    calls = {}

    def mk(name):
        def fn(*a, **k):
            calls[name] = {"args": a, "kwargs": k}
            return [_pil()]
        return fn

    monkeypatch.setattr(gen.zc, "txt2img", mk("txt2img"))
    monkeypatch.setattr(gen.zc, "img2img", mk("img2img"))
    monkeypatch.setattr(gen.zc, "inpaint", mk("inpaint"))
    return calls


def test_txt2img_when_no_init(tmp_path, capture):
    args = make_args(tmp_path)
    out = gen.generate(args, root=None)
    assert "txt2img" in capture and "img2img" not in capture
    assert len(out) == 1 and isinstance(out[0], Image.Image)


def test_img2img_when_init_and_strength(tmp_path, capture):
    # render loop always sets use_init=True alongside init_sample
    args = make_args(tmp_path, use_init=True, init_sample=_init_sample(), strength=0.65)
    gen.generate(args, root=None)
    assert "img2img" in capture and "txt2img" not in capture
    # generate passes Deforum-convention strength through; inversion is the client's job
    assert capture["img2img"]["args"][2] == pytest.approx(0.65)


def test_return_sample_contract(tmp_path, capture):
    args = make_args(tmp_path, use_init=True, init_sample=_init_sample(), strength=0.5)
    out = gen.generate(args, root=None, return_sample=True)
    assert "img2img" in capture and "txt2img" not in capture  # routing, not just shape
    assert len(out) == 2
    sample, image = out
    assert isinstance(sample, np.ndarray)
    assert sample.shape == (1, 3, 8, 8)
    assert sample.dtype == np.float32
    assert sample.min() >= -1.0 and sample.max() <= 1.0
    assert isinstance(image, Image.Image)


def test_empty_images_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(gen.zc, "txt2img", lambda *a, **k: [])
    args = make_args(tmp_path)
    with pytest.raises(RuntimeError, match="no images"):
        gen.generate(args, root=None)


def test_result_resized_to_canvas(tmp_path, monkeypatch):
    # API may return a different size than requested; generate must normalize it.
    monkeypatch.setattr(gen.zc, "txt2img", lambda *a, **k: [Image.new("RGB", (640, 480), (1, 2, 3))])
    args = make_args(tmp_path, W=8, H=8)
    out = gen.generate(args, root=None)
    assert out[0].size == (8, 8)


def test_mask_sample_routes_to_inpaint_via_real_loader(tmp_path, capture):
    # With a warped per-frame mask present, _load_mask_pil (real, not mocked) should
    # build the mask from mask_sample and route to inpaint.
    args = make_args(tmp_path, init_sample=_init_sample(), use_mask=True,
                     mask_sample=_init_sample(), strength=0.6)
    gen.generate(args, root=None)
    assert "inpaint" in capture


def test_strength_zero_no_init_falls_back_to_txt2img(tmp_path, capture):
    # use_init False but strength > 0 -> auto-zeroed -> txt2img, not img2img
    args = make_args(tmp_path, use_init=False, strength=0.65, strength_0_no_init=True)
    gen.generate(args, root=None)
    assert "txt2img" in capture and "img2img" not in capture
    assert args.strength == 0


def test_mask_routes_to_inpaint(tmp_path, capture, monkeypatch):
    monkeypatch.setattr(gen, "_load_mask_pil", lambda args: Image.new("L", (8, 8), 255))
    args = make_args(tmp_path, init_sample=_init_sample(), use_mask=True, strength=0.6)
    gen.generate(args, root=None)
    assert "inpaint" in capture
    assert "img2img" not in capture and "txt2img" not in capture


def test_overlay_mask_composites(tmp_path, capture, monkeypatch):
    monkeypatch.setattr(gen, "_load_mask_pil", lambda args: Image.new("L", (8, 8), 255))
    args = make_args(tmp_path, init_sample=_init_sample(), use_mask=True,
                     overlay_mask=True, strength=0.6)
    out = gen.generate(args, root=None)
    assert isinstance(out[0], Image.Image)  # composite still yields an image


@pytest.mark.parametrize("kw", [{"return_latent": True}, {"return_c": True}])
def test_unsupported_returns_raise(tmp_path, capture, kw):
    args = make_args(tmp_path)
    with pytest.raises(NotImplementedError):
        gen.generate(args, root=None, **kw)
