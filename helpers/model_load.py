import os
from types import SimpleNamespace

from . import zimage_client as zc
from .backends import resolve_backend_name


def _detect_device():
    # torch is not in the light/fal install, so detect lazily -- the fal path must
    # import this module without torch present.
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_model(root, load_on_run_all=True, check_sha256=True, map_location="cuda"):
    """Resolve the active generation backend handle.

    Returns ``(model, device)`` for signature compatibility with existing call
    sites. No checkpoint is loaded here either way. For ``backend='fal'`` (default)
    it validates ``FAL_KEY``; for ``backend='local'`` it fails fast if torch/diffusers
    aren't installed (the diffusers pipeline itself loads lazily on first generate).
    The ``load_on_run_all`` / ``check_sha256`` / ``map_location`` args are accepted
    but unused.
    """
    backend = resolve_backend_name(root)

    if backend == "local":
        try:
            import torch  # noqa: F401
            import diffusers  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "backend='local' selected, but torch/diffusers are not installed. "
                "Run `python install_requirements.py --with-local` and download the "
                "Z-Image-Turbo weights (or use backend='fal', the default)."
            ) from e
        model = SimpleNamespace(backend="z-image-local")
        print("..using local Z-Image (diffusers); pipeline loads on first generate")
    else:
        zc.resolve_fal_key()
        model = SimpleNamespace(backend="z-image-turbo")
        print("..using Z-Image Turbo (fal.ai); no local model weights loaded")

    return model, _detect_device()


def get_model_output_paths(root):

    models_path = root.models_path
    output_path = root.output_path

    #@markdown **Google Drive Path Variables (Optional)**

    force_remount = False

    try:
        ipy = get_ipython()
    except:
        ipy = 'could not get_ipython'

    if 'google.colab' in str(ipy):
        if root.mount_google_drive:
            from google.colab import drive # type: ignore
            try:
                drive_path = "/content/drive"
                drive.mount(drive_path,force_remount=force_remount)
                models_path = root.models_path_gdrive
                output_path = root.output_path_gdrive
            except:
                print("..error mounting drive or with drive path variables")
                print("..reverting to default path variables")

    models_path = os.path.abspath(models_path)
    output_path = os.path.abspath(output_path)
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    print(f"models_path: {models_path}")
    print(f"output_path: {output_path}")

    return models_path, output_path
