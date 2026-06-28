"""Model selection (z-image default | krea2) and the 2x2 (model x location) dispatch."""
from types import SimpleNamespace

import pytest

from helpers import backends
from helpers import zimage_client


def test_resolve_model_default_zimage(monkeypatch):
    monkeypatch.delenv("DEFORUM_MODEL", raising=False)
    assert backends.resolve_model_name(None) == "z-image"
    assert backends.resolve_model_name(SimpleNamespace()) == "z-image"


def test_resolve_model_from_root_and_env(monkeypatch):
    monkeypatch.delenv("DEFORUM_MODEL", raising=False)
    assert backends.resolve_model_name(SimpleNamespace(model_name="krea2")) == "krea2"
    monkeypatch.setenv("DEFORUM_MODEL", "krea2")
    assert backends.resolve_model_name(SimpleNamespace()) == "krea2"
    # explicit root wins over env
    assert backends.resolve_model_name(SimpleNamespace(model_name="z-image")) == "z-image"


def test_resolve_model_unknown_raises(monkeypatch):
    monkeypatch.delenv("DEFORUM_MODEL", raising=False)
    with pytest.raises(ValueError):
        backends.resolve_model_name(SimpleNamespace(model_name="dalle"))


def test_dispatch_zimage_fal_default(monkeypatch):
    monkeypatch.delenv("DEFORUM_MODEL", raising=False)
    monkeypatch.delenv("DEFORUM_BACKEND", raising=False)
    assert backends.resolve_backend(None) is zimage_client


def test_dispatch_2x2(monkeypatch):
    monkeypatch.delenv("DEFORUM_MODEL", raising=False)
    monkeypatch.delenv("DEFORUM_BACKEND", raising=False)

    def mod(model, backend):
        return backends.resolve_backend(SimpleNamespace(model_name=model, backend=backend)).__name__

    assert mod("z-image", "fal") == "helpers.zimage_client"
    assert mod("z-image", "local") == "helpers.zimage_local"
    assert mod("krea2", "fal") == "helpers.krea2_client"
    assert mod("krea2", "local") == "helpers.krea2_local"
