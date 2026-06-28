"""Generation backend selection.

Two axes select the active backend module, which always exposes the same surface --
txt2img / img2img / inpaint, each returning a list of PIL images:

  location:  "fal"   (default): hosted on fal.ai. Dependency-light and torch-free.
             "local" (opt-in):  diffusers pipelines on a local GPU. Pulls torch +
                                diffusers, so the local modules are imported lazily
                                ONLY when selected, preserving the fal path's
                                torch-free property.
  model:     "z-image" (default): Z-Image Turbo (~6B).
             "krea2":             Krea 2 Turbo (12.9B). Text-to-image only on fal;
                                  local adds an experimental flow-match img2img.

The 2x2 grid of modules:

                 fal                    local
  z-image   zimage_client           zimage_local
  krea2     krea2_client            krea2_local

Selection precedence (each axis): root.<field>  ->  $DEFORUM_<AXIS>  ->  default.
  location: root.backend    -> $DEFORUM_BACKEND -> "fal"
  model:    root.model_name -> $DEFORUM_MODEL   -> "z-image"
"""
import os

from . import zimage_client

VALID_BACKENDS = ("fal", "local")
VALID_MODELS = ("z-image", "krea2")


def resolve_backend_name(root=None):
    name = getattr(root, "backend", None) if root is not None else None
    name = (name or os.environ.get("DEFORUM_BACKEND") or "fal").lower()
    if name not in VALID_BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; expected one of {VALID_BACKENDS}")
    return name


def resolve_model_name(root=None):
    name = getattr(root, "model_name", None) if root is not None else None
    name = (name or os.environ.get("DEFORUM_MODEL") or "z-image").lower()
    if name not in VALID_MODELS:
        raise ValueError(f"Unknown model {name!r}; expected one of {VALID_MODELS}")
    return name


def resolve_backend(root=None):
    """Return the active backend module for the (model, location) pair.

    Local / krea2 modules are imported lazily (they pull torch + diffusers); the
    default z-image+fal path imports nothing extra and stays torch-free.
    """
    location = resolve_backend_name(root)
    model = resolve_model_name(root)
    if model == "z-image":
        if location == "fal":
            return zimage_client
        from . import zimage_local  # lazy: torch + diffusers only for local
        return zimage_local
    # krea2
    if location == "fal":
        from . import krea2_client   # torch-free fal client
        return krea2_client
    from . import krea2_local        # lazy: torch + diffusers only for local
    return krea2_local
