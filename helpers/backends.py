"""Generation backend selection.

Two backends expose the same surface -- txt2img / img2img / inpaint, each
returning a list of PIL images:

  - "fal"   (default): helpers.zimage_client, hosted Z-Image Turbo on fal.ai.
            Dependency-light and torch-free.
  - "local" (opt-in):  helpers.zimage_local, diffusers Z-Image pipelines on a
            local GPU. Pulls torch + diffusers, so it is imported lazily ONLY
            when selected, preserving the fal path's torch-free property.

Selection precedence: root.backend  ->  $DEFORUM_BACKEND  ->  "fal".
"""
import os

from . import zimage_client

VALID_BACKENDS = ("fal", "local")


def resolve_backend_name(root=None):
    name = getattr(root, "backend", None) if root is not None else None
    name = (name or os.environ.get("DEFORUM_BACKEND") or "fal").lower()
    if name not in VALID_BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; expected one of {VALID_BACKENDS}")
    return name


def resolve_backend(root=None):
    """Return the active backend module. Local is imported lazily (pulls torch)."""
    name = resolve_backend_name(root)
    if name == "fal":
        return zimage_client
    from . import zimage_local  # lazy: torch + diffusers only loaded for local
    return zimage_local
