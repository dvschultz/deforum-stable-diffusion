"""Guard: the 2D / txt2img / img2img path must import without torch.

Generation runs on fal.ai, and 2D animation + image batches need no GPU. torch is
only for 3D depth warping (lazy-imported). This runs in a subprocess with torch
(and friends) blocked, so a stray top-level `import torch` in the 2D path fails CI.
"""
import os
import subprocess
import sys


def test_2d_path_imports_without_torch():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = (
        "import sys\n"
        "for m in ('torch', 'torchvision', 'py3d_tools', 'timm'):\n"
        "    sys.modules[m] = None\n"  # any import of these now raises ImportError
        f"sys.path.insert(0, {os.path.join(root, 'src')!r})\n"
        f"sys.path.insert(0, {root!r})\n"
        "import helpers.zimage_client, helpers.krea2_client, helpers.generate, helpers.animation\n"
        "import helpers.load_images, helpers.save_images, helpers.colors\n"
        "import helpers.prompt, helpers.prompts, helpers.hybrid_video, helpers.settings\n"
        # backends + model_load are on the fal path (entry script imports load_model);
        # they must import torch-free. zimage_local is NOT imported here (it's local-only).
        "import helpers.backends, helpers.model_load\n"
        "print('TORCH_FREE_OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert r.returncode == 0, f"2D path pulled torch:\n{r.stderr}"
    assert "TORCH_FREE_OK" in r.stdout
