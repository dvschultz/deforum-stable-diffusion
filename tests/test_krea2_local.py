"""Krea 2 local backend. The diffusers Krea2Pipeline is mocked, so these run without a GPU
or weights (torch itself is available in the dev env). The custom flow-match img2img is
checked for call *shape* (packed latents, CFG off, truncated sigma tail), not numerics --
numeric correctness is a GPU-validation item."""
from types import SimpleNamespace as NS

import pytest
import torch
from PIL import Image

from helpers import krea2_local as kl


# --- a tiny capture pipe for txt2img / embeds ----------------------------

class CapturePipe:
    def __init__(self):
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        n = kw.get("num_images_per_prompt", 1)
        return NS(images=[Image.new("RGB", (kw.get("width", 8), kw.get("height", 8)))] * n)

    def encode_prompt(self, prompt, device=None, num_images_per_prompt=1):
        return torch.zeros(1, 5, 12, 8), torch.ones(1, 5, dtype=torch.bool)


@pytest.fixture
def cap(monkeypatch):
    pipe = CapturePipe()
    monkeypatch.setattr(kl, "_load_pipe", lambda kind: pipe)
    return pipe


def test_txt2img_defaults_steps8_cfg_free(cap):
    out = kl.txt2img("a fox", 512, 512, seed=3, steps=8, guidance_scale=5.0)
    assert len(out) == 1
    kw = cap.calls[-1]
    assert kw["prompt"] == "a fox"
    assert kw["num_inference_steps"] == 8
    assert kw["guidance_scale"] == 0.0          # forced CFG-free regardless of request
    assert kw["height"] == 512 and kw["width"] == 512


def test_txt2img_embeds_forces_cfg_off_and_passes_mask(cap):
    emb = (torch.zeros(1, 5, 12, 8), torch.ones(1, 5, dtype=torch.bool))
    kl.txt2img_embeds(emb, 8, 8, seed=1, guidance_scale=4.5)
    kw = cap.calls[-1]
    assert kw["prompt_embeds"] is emb[0]
    assert kw["prompt_embeds_mask"] is emb[1]
    assert kw["guidance_scale"] == 0.0


def test_slerp_embeds_aligns_seq_and_endpoints(cap):
    e1 = (torch.ones(1, 3, 12, 8), torch.ones(1, 3, dtype=torch.bool))
    e2 = (torch.ones(1, 5, 12, 8) * 3, torch.ones(1, 5, dtype=torch.bool))
    emb, mask = kl.slerp_embeds(e1, e2, 0.5)
    assert emb.shape == (1, 3, 12, 8)           # aligned to the shorter prompt
    assert mask.shape == (1, 3)


def test_inpaint_unsupported():
    with pytest.raises(NotImplementedError):
        kl.inpaint("p", Image.new("RGB", (8, 8)), Image.new("L", (8, 8)), 0.6, 8, 8)


# --- the custom flow-match img2img ---------------------------------------

class FakeLatentDist:
    def __init__(self, t):
        self._t = t

    def sample(self, generator=None):
        return self._t


class FakeVAE:
    dtype = torch.float32
    config = NS(z_dim=16, latents_mean=[0.0] * 16, latents_std=[1.0] * 16)

    def encode(self, img):
        B, H, W = img.shape[0], img.shape[-2], img.shape[-1]
        return NS(latent_dist=FakeLatentDist(torch.zeros(B, 16, 1, H // 8, W // 8)))


class FakeScheduler:
    order = 1

    def __init__(self):
        self.timesteps = None
        self.begin = None
        self.scale_calls = []

    def set_timesteps(self, sigmas=None, mu=None, device=None, **kw):
        self.timesteps = torch.arange(len(sigmas), 0, -1)   # length == len(sigmas)

    def set_begin_index(self, i):
        self.begin = i

    def scale_noise(self, sample, timestep, noise):
        self.scale_calls.append((tuple(sample.shape), tuple(noise.shape)))
        return 0.5 * sample + 0.5 * noise


class FakeImageProcessor:
    def preprocess(self, image, height=None, width=None):
        return torch.zeros(1, 3, int(height), int(width))


class FakeKrea2Pipe:
    def __init__(self):
        self._execution_device = "cpu"
        self.transformer = NS(dtype=torch.float32, config=NS(in_channels=64))
        self.vae = FakeVAE()
        self.patch_size = 2
        self.vae_scale_factor = 8
        self.image_processor = FakeImageProcessor()
        self.config = NS(is_distilled=True)
        self.scheduler = FakeScheduler()
        self.calls = []

    def _pack_latents(self, latents, B, C, h, w):
        p = self.patch_size
        return (latents.view(B, C, h // p, p, w // p, p)
                .permute(0, 2, 4, 1, 3, 5)
                .reshape(B, (h // p) * (w // p), C * p * p))

    def __call__(self, **kw):
        self.calls.append(kw)
        return NS(images=[Image.new("RGB", (kw.get("width", 8), kw.get("height", 8)))])


def test_custom_img2img_packs_latents_truncates_sigmas_cfg_off(monkeypatch):
    pipe = FakeKrea2Pipe()
    monkeypatch.setattr(kl, "_load_pipe", lambda kind: pipe)

    out = kl.img2img("a fox", Image.new("RGB", (64, 64)), 0.65, 64, 64, seed=1, steps=8)
    assert len(out) == 1

    kw = pipe.calls[-1]
    # latents are packed to (B, seq, in_channels): seq=(8/2)*(8/2)=16, in_channels=64
    assert tuple(kw["latents"].shape) == (1, 16, 64)
    assert kw["latents"].shape[-1] == pipe.transformer.config.in_channels
    assert kw["guidance_scale"] == 0.0
    assert kw["prompt"] == "a fox"
    # strength 0.65 -> diffusers 0.35 -> init_ts=min(8*0.35,8)=2.8 -> t_start=5 -> tail len 3
    assert len(kw["sigmas"]) == 3
    assert kw["num_inference_steps"] == 3
    # the init was blended with noise via scale_noise exactly once
    assert len(pipe.scheduler.scale_calls) == 1
    assert pipe.scheduler.begin == 5


def test_custom_img2img_packing_mismatch_raises(monkeypatch):
    pipe = FakeKrea2Pipe()
    # in_channels not a multiple of patch_size**2 (66): C = 66//4 = 16 (matches z_dim, so
    # the blend is consistent) but C*p*p = 64 != 66, so the packing guard must fire.
    pipe.transformer.config.in_channels = 66
    monkeypatch.setattr(kl, "_load_pipe", lambda kind: pipe)
    with pytest.raises(RuntimeError):
        kl.img2img("p", Image.new("RGB", (64, 64)), 0.65, 64, 64, seed=1, steps=8)
