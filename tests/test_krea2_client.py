"""Krea 2 fal client: txt2img args to fal-ai/krea-2/turbo; img2img/inpaint unsupported."""
import pytest
from PIL import Image

from helpers import krea2_client as kc


@pytest.fixture
def capture(monkeypatch):
    seen = {}

    def fake_submit(endpoint, arguments, **_):
        seen["endpoint"] = endpoint
        seen["arguments"] = arguments
        return {"images": [{"url": "x"}]}

    monkeypatch.setattr(kc, "_submit", fake_submit)
    monkeypatch.setattr(kc, "_result_to_images", lambda result: [Image.new("RGB", (8, 8))])
    return seen


def test_txt2img_hits_krea2_turbo_endpoint(capture):
    out = kc.txt2img("a fox in the snow", 512, 512, seed=7, steps=8, num_images=2)
    assert len(out) == 1 and isinstance(out[0], Image.Image)
    assert capture["endpoint"] == "fal-ai/krea-2/turbo"
    args = capture["arguments"]
    assert args["prompt"] == "a fox in the snow"
    assert args["image_size"] == {"width": 512, "height": 512}
    assert args["num_inference_steps"] == 8
    assert args["num_images"] == 2
    assert args["output_format"] == "png"
    assert args["seed"] == 7


def test_txt2img_clamps_steps_and_omits_seed(capture):
    kc.txt2img("p", 64, 64, seed=None, steps=20)
    args = capture["arguments"]
    assert args["num_inference_steps"] == 8       # clamped to the 1-8 turbo budget
    assert "seed" not in args


def test_txt2img_ignores_zimage_only_kwargs(capture):
    # guidance_scale / acceleration are not part of the Krea 2 turbo schedule.
    kc.txt2img("p", 64, 64, guidance_scale=5.0, acceleration="high")
    args = capture["arguments"]
    assert "guidance_scale" not in args
    assert "acceleration" not in args


def test_img2img_and_inpaint_unsupported():
    with pytest.raises(NotImplementedError):
        kc.img2img("p", Image.new("RGB", (8, 8)), 0.6, 8, 8)
    with pytest.raises(NotImplementedError):
        kc.inpaint("p", Image.new("RGB", (8, 8)), Image.new("L", (8, 8)), 0.6, 8, 8)
