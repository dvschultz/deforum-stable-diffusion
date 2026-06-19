import os
from types import SimpleNamespace

import torch

from . import zimage_client as zc


def load_model(root, load_on_run_all=True, check_sha256=True, map_location="cuda"):
    """Resolve the Z-Image Turbo (fal.ai) backend handle.

    Returns ``(model, device)`` for signature compatibility with existing call
    sites. No local weights are downloaded or instantiated -- generation runs on
    the hosted Z-Image Turbo model. ``root.model`` becomes a lightweight handle
    carrying the endpoint identifiers; ``FAL_KEY`` is validated here so a missing
    key fails at setup time rather than on the first frame. The ``load_on_run_all``,
    ``check_sha256``, and ``map_location`` arguments are accepted but unused.
    """
    zc.resolve_fal_key()

    # Lightweight handle. generate() calls the zimage_client endpoints directly,
    # so this only marks the active backend (for logging / assertions).
    model = SimpleNamespace(backend="z-image-turbo")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("..using Z-Image Turbo (fal.ai); no local model weights loaded")
    return model, device


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
