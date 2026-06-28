import argparse
import platform
import subprocess


def pip_install_packages(packages, extra_index_url=None, verbose=False, pre=False):
    for package in packages:
        try:
            print(f"..installing {package}")
            
            # base command
            cmd = ["pip", "install"]

            if pre:
                cmd.append("--pre")

            # add '-q' if not verbose
            if not verbose:
                cmd.append("-q")

            # add package name
            cmd.append(package)

            # add extra_index_url if it exists
            if extra_index_url:
                cmd.extend(["--extra-index-url", extra_index_url])

            if verbose:
                print(cmd)

            # run the command and capture output
            result = subprocess.run(cmd, capture_output=not verbose, text=True)
            
            if verbose:
                # print stdout and stderr if verbose
                print(result.stdout)
                print(result.stderr)

        except Exception as e:
            print(f"failed to install {package}: {e}")
    return


# Light core: everything the hosted-generation (Z-Image Turbo on fal.ai) path,
# 2D/3D motion, color coherence, hybrid video, the notebook, and ffmpeg assembly
# need. No torch -- generation runs remotely.
CORE = [
    "fal-client",
    "pillow",
    "numpy",
    "opencv-python",
    "pandas",
    "einops",
    "requests",
    "scipy",
    "numexpr",
    "numpngw",
    "scikit-image>=0.24",   # colors.maintain_colors (match_histograms); >=0.24 for numpy 2 compat
    "pydantic",
    "colab-convert",
    "ipython",
    "ipywidgets",
    "jupyterlab",
    "notebook",
    "jupyter_http_over_ws",
]

# Optional: only needed for 3D depth warping (MiDaS/AdaBins) and grid previews.
# Pulls the heavy GPU stack; install with --with-3d.
THREE_D = [
    "torch",
    "torchvision",
    "torchaudio",
    "timm",   # MiDaS backbone
]

# Optional (EXPERIMENTAL): the local backend (backend='local'). Runs the model on your
# own CUDA GPU instead of fal.ai. Large; needs the weights too:
#   hf download Tongyi-MAI/Z-Image-Turbo   (Z-Image Turbo, ~6B)
#   hf download krea/Krea-2-Turbo          (Krea 2 Turbo, 12.9B, ~26GB)
# diffusers must be from source -- the ZImage AND Krea2 pipelines aren't in a stable
# release (Krea2Pipeline needs a recent main; re-run --with-local to upgrade).
LOCAL = [
    "torch",
    "torchvision",
    "git+https://github.com/huggingface/diffusers.git",
    "transformers",
    "accelerate",
    "huggingface_hub",
    "safetensors",
    "sentencepiece",
    "ftfy",
    "regex",   # ftfy + the vendored CLIP (used by experimental gradient guidance)
    "scikit-learn",   # conditioning.KMeans color-palette loss (experimental gradient guidance)
    "bitsandbytes",   # optional: ZIMAGE_QUANTIZE=int8/nf4 to halve/quarter resident VRAM
]
PYTORCH_INDEX = "https://download.pytorch.org/whl/nightly/cu121"


def install_requirements(verbose=False, with_3d=False, with_local=False):

    # Detect System
    os_system = platform.system()
    print(f"system detected: {os_system}")

    pip_install_packages(CORE, verbose=verbose)

    if with_3d:
        print("..installing 3D/depth extras (torch stack -- this is large)")
        pip_install_packages(THREE_D, extra_index_url=PYTORCH_INDEX, verbose=verbose, pre=True)

    if with_local:
        print("..installing LOCAL backend (EXPERIMENTAL; torch + diffusers-from-source)")
        pip_install_packages(LOCAL, extra_index_url=PYTORCH_INDEX, verbose=verbose, pre=True)
        print("..now download the weights for the model you want:")
        print("..  Z-Image Turbo (~6B):    hf download Tongyi-MAI/Z-Image-Turbo")
        print("..  Krea 2 Turbo (12.9B):   hf download krea/Krea-2-Turbo   (~26GB)")
        print("..then run with backend='local' (ModelSetup) or DEFORUM_BACKEND=local,")
        print("..and model='krea2' / DEFORUM_MODEL=krea2 for Krea 2 (default model='z-image').")
        print("..  Krea 2 is 12.9B: set ZIMAGE_QUANTIZE=nf4 to fit a single 24GB GPU.")
        print("..  Krea 2 weights are under the Krea 2 Community License.")

    if not (with_3d or with_local):
        print("..skipping the torch stack: generation runs on fal.ai (Z-Image Turbo),")
        print("  and 2D animation / image batches need no GPU.")
        print("  For 3D depth warping: --with-3d   |   for the local backend: --with-local")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-3d', action='store_true',
                        help='also install torch + MiDaS/AdaBins deps for 3D depth warping')
    parser.add_argument('--with-local', action='store_true',
                        help='also install the EXPERIMENTAL local Z-Image backend (torch + diffusers)')
    parser.add_argument('--verbose', action='store_true', help='print pip install stuff')
    args = parser.parse_args()
    install_requirements(verbose=args.verbose, with_3d=args.with_3d, with_local=args.with_local)