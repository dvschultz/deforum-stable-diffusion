"""Unit tests for the Z-Image Turbo client wrapper.

All fal.ai network interaction (subscribe / upload / download) is mocked, so these
run without a FAL_KEY or network access.
"""
import pytest
from PIL import Image

from helpers import zimage_client as zc


def _img(w=8, h=8):
    return Image.new("RGB", (w, h), (128, 128, 128))


# --- resolve_fal_key -------------------------------------------------------

def test_resolve_fal_key_returns_value(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "abc123")
    assert zc.resolve_fal_key() == "abc123"


def test_resolve_fal_key_raises_when_missing(monkeypatch):
    monkeypatch.delenv("FAL_KEY", raising=False)
    with pytest.raises(RuntimeError) as exc:
        zc.resolve_fal_key()
    assert "FAL_KEY" in str(exc.value)


def test_resolve_fal_key_raises_when_blank(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "   ")
    with pytest.raises(RuntimeError):
        zc.resolve_fal_key()


# --- to_fal_strength (the inversion that keeps animation coherent) ---------

def test_to_fal_strength_inverts():
    assert zc.to_fal_strength(0.65) == pytest.approx(0.35)
    assert zc.to_fal_strength(0.0) == pytest.approx(1.0)
    assert zc.to_fal_strength(1.0) == pytest.approx(0.0)


def test_to_fal_strength_clamps():
    assert zc.to_fal_strength(1.5) == 0.0
    assert zc.to_fal_strength(-0.5) == 1.0


# --- clamp_steps -----------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [(50, 8), (0, 1), (4, 4), (-3, 1), (8, 8), (None, 8)])
def test_clamp_steps(raw, expected):
    assert zc.clamp_steps(raw) == expected


# --- image_size_arg --------------------------------------------------------

def test_image_size_arg():
    assert zc.image_size_arg(512, 512) == {"width": 512, "height": 512}
    assert zc.image_size_arg(640, 384) == {"width": 640, "height": 384}


# --- _is_auth_error --------------------------------------------------------

@pytest.mark.parametrize("msg,is_auth", [
    ("401 Unauthorized", True),
    ("403 Forbidden", True),
    ("Invalid API key", True),
    ("Connection reset by peer", False),
    ("503 Service Unavailable", False),
])
def test_is_auth_error(msg, is_auth):
    assert zc._is_auth_error(Exception(msg)) is is_auth


# --- _submit retry behaviour ----------------------------------------------

def test_submit_retries_transient_then_succeeds(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "k")
    calls = {"n": 0}

    def flaky(endpoint, arguments=None, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("503 Service Unavailable")
        return {"images": [{"url": "http://x/img.png"}]}

    monkeypatch.setattr(zc.fal_client, "subscribe", flaky)
    result = zc._submit("ep", {"prompt": "p"}, base_delay=0, sleep=lambda *_: None)
    assert calls["n"] == 3
    assert result["images"][0]["url"] == "http://x/img.png"


def test_submit_fails_fast_on_auth(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "k")
    calls = {"n": 0}

    def auth_fail(endpoint, arguments=None, **kwargs):
        calls["n"] += 1
        raise RuntimeError("401 Unauthorized")

    monkeypatch.setattr(zc.fal_client, "subscribe", auth_fail)
    with pytest.raises(RuntimeError):
        zc._submit("ep", {}, sleep=lambda *_: None)
    assert calls["n"] == 1  # no retries on auth failure


def test_submit_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "k")

    def always_fail(endpoint, arguments=None, **kwargs):
        raise RuntimeError("timeout")

    monkeypatch.setattr(zc.fal_client, "subscribe", always_fail)
    with pytest.raises(RuntimeError) as exc:
        zc._submit("ep", {}, max_retries=2, base_delay=0, sleep=lambda *_: None)
    assert "failed after 2 attempts" in str(exc.value)


# --- endpoint functions (routing + argument shaping) -----------------------

def test_txt2img_happy_path(monkeypatch):
    captured = {}

    def fake_submit(endpoint, arguments, **kw):
        captured["endpoint"] = endpoint
        captured["arguments"] = arguments
        return {"images": [{"url": "u"}]}

    monkeypatch.setattr(zc, "_submit", fake_submit)
    monkeypatch.setattr(zc, "_download_image", lambda url: _img())

    out = zc.txt2img("a cat", 512, 512, seed=7, steps=50)
    assert len(out) == 1 and isinstance(out[0], Image.Image)
    assert captured["endpoint"] == zc.ENDPOINT_TXT2IMG
    assert captured["arguments"]["num_inference_steps"] == 8  # clamped
    assert captured["arguments"]["seed"] == 7
    assert "image_url" not in captured["arguments"]


def test_img2img_inverts_strength_and_uploads(monkeypatch):
    captured = {}
    monkeypatch.setattr(zc, "_upload", lambda img: "http://uploaded/init.png")

    def fake_submit(endpoint, arguments, **kw):
        captured["endpoint"] = endpoint
        captured["arguments"] = arguments
        return {"images": [{"url": "u"}]}

    monkeypatch.setattr(zc, "_submit", fake_submit)
    monkeypatch.setattr(zc, "_download_image", lambda url: _img())

    out = zc.img2img("a dog", _img(), 0.65, 512, 512, seed=1, steps=8)
    assert len(out) == 1
    assert captured["endpoint"] == zc.ENDPOINT_IMG2IMG
    assert captured["arguments"]["image_url"] == "http://uploaded/init.png"
    assert captured["arguments"]["strength"] == pytest.approx(0.35)  # inverted


def test_acceleration_none_is_omitted(monkeypatch):
    captured = {}

    def fake_submit(endpoint, arguments, **kw):
        captured["arguments"] = arguments
        return {"images": [{"url": "u"}]}

    monkeypatch.setattr(zc, "_submit", fake_submit)
    monkeypatch.setattr(zc, "_download_image", lambda url: _img())

    zc.txt2img("p", 512, 512, acceleration="none")
    assert "acceleration" not in captured["arguments"]  # 'none' -> field omitted

    zc.txt2img("p", 512, 512, acceleration="high")
    assert captured["arguments"]["acceleration"] == "high"


def test_inpaint_uploads_image_and_mask(monkeypatch):
    captured = {}
    uploads = []
    monkeypatch.setattr(zc, "_upload", lambda img: uploads.append(img) or f"http://u/{len(uploads)}")

    def fake_submit(endpoint, arguments, **kw):
        captured["endpoint"] = endpoint
        captured["arguments"] = arguments
        return {"images": [{"url": "u"}]}

    monkeypatch.setattr(zc, "_submit", fake_submit)
    monkeypatch.setattr(zc, "_download_image", lambda url: _img())

    out = zc.inpaint("fill", _img(), _img(), 0.5, 512, 512)
    assert len(out) == 1
    assert captured["endpoint"] == zc.ENDPOINT_INPAINT
    assert "image_url" in captured["arguments"]
    assert "mask_url" in captured["arguments"]
    assert len(uploads) == 2  # init + mask
