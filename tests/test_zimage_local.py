"""Local diffusers backend wrapper. The diffusers pipeline is mocked, so these run
without a GPU or the model weights (torch itself is available in the dev env)."""
from types import SimpleNamespace

import pytest
from PIL import Image

from helpers import zimage_local as zl


class FakePipe:
    def __init__(self):
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        n = kw.get("num_images_per_prompt", 1)
        return SimpleNamespace(images=[Image.new("RGB", (kw.get("width", 8), kw.get("height", 8)))] * n)


@pytest.fixture
def fake(monkeypatch):
    pipe = FakePipe()
    monkeypatch.setattr(zl, "_load_pipe", lambda kind: pipe)
    return pipe


def test_txt2img_passes_core_kwargs(fake):
    out = zl.txt2img("a cat", 512, 512, seed=7, steps=12, guidance_scale=4.0)
    assert len(out) == 1 and isinstance(out[0], Image.Image)
    kw = fake.calls[-1]
    assert kw["prompt"] == "a cat"
    assert kw["height"] == 512 and kw["width"] == 512
    assert kw["num_inference_steps"] == 12          # not clamped to 8 locally
    assert kw["guidance_scale"] == 4.0
    assert kw["generator"] is not None              # seeded


def test_img2img_inverts_strength_and_passes_image(fake):
    init = Image.new("RGB", (8, 8))
    zl.img2img("a dog", init, 0.65, 8, 8, seed=1, steps=8)
    kw = fake.calls[-1]
    assert kw["image"] is init
    assert kw["strength"] == pytest.approx(0.35)    # Deforum 0.65 -> diffusers 0.35
    assert "mask_image" not in kw


def test_inpaint_passes_image_and_mask(fake):
    init, mask = Image.new("RGB", (8, 8)), Image.new("L", (8, 8), 255)
    zl.inpaint("fill", init, mask, 0.6, 8, 8)
    kw = fake.calls[-1]
    assert kw["image"] is init and kw["mask_image"] is mask


def test_ignores_local_irrelevant_kwargs(fake):
    # acceleration is a fal-only kwarg; the local wrapper must tolerate it.
    zl.txt2img("p", 8, 8, acceleration="high")
    assert "acceleration" not in fake.calls[-1]


def test_num_images_batches(fake):
    out = zl.txt2img("p", 8, 8, num_images=3)
    assert len(out) == 3
