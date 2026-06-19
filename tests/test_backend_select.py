"""Backend selection: fal default, local opt-in, generate() routes uniformly."""
from types import SimpleNamespace

import pytest
from PIL import Image

from helpers import backends
from helpers import generate as gen
from helpers import zimage_client


def test_resolve_name_default_fal(monkeypatch):
    monkeypatch.delenv("DEFORUM_BACKEND", raising=False)
    assert backends.resolve_backend_name(None) == "fal"
    assert backends.resolve_backend_name(SimpleNamespace()) == "fal"


def test_resolve_name_from_root_and_env(monkeypatch):
    monkeypatch.delenv("DEFORUM_BACKEND", raising=False)
    assert backends.resolve_backend_name(SimpleNamespace(backend="local")) == "local"
    monkeypatch.setenv("DEFORUM_BACKEND", "local")
    assert backends.resolve_backend_name(SimpleNamespace()) == "local"
    # explicit root wins over env
    assert backends.resolve_backend_name(SimpleNamespace(backend="fal")) == "fal"


def test_resolve_name_unknown_raises(monkeypatch):
    monkeypatch.delenv("DEFORUM_BACKEND", raising=False)
    with pytest.raises(ValueError):
        backends.resolve_backend_name(SimpleNamespace(backend="banana"))


def test_resolve_backend_fal_is_zimage_client(monkeypatch):
    monkeypatch.delenv("DEFORUM_BACKEND", raising=False)
    assert backends.resolve_backend(None) is zimage_client


def _args(tmp_path, **over):
    d = dict(cond_prompt="x", outdir=str(tmp_path), use_init=False, strength=0.0,
             strength_0_no_init=True, n_samples=1, steps=8, seed=1, acceleration="regular",
             init_sample=None, init_image=None, use_alpha_as_mask=False, use_mask=False,
             mask_file=None, invert_mask=False, overlay_mask=False, mask_overlay_blur=0,
             W=8, H=8)
    d.update(over)
    return SimpleNamespace(**d)


def _spies(monkeypatch):
    from helpers import zimage_local
    called = {}

    def mk(tag):
        def fn(*a, **k):
            called[tag] = True
            return [Image.new("RGB", (8, 8))]
        return fn

    monkeypatch.setattr(zimage_local, "txt2img", mk("local"))
    monkeypatch.setattr(zimage_client, "txt2img", mk("fal"))
    return called


def test_generate_routes_to_local_backend(tmp_path, monkeypatch):
    called = _spies(monkeypatch)
    gen.generate(_args(tmp_path, backend="local"), root=SimpleNamespace(backend="local"))
    assert called.get("local") and not called.get("fal")


def test_generate_default_routes_to_fal(tmp_path, monkeypatch):
    called = _spies(monkeypatch)
    gen.generate(_args(tmp_path), root=None)  # no backend -> fal
    assert called.get("fal") and not called.get("local")
