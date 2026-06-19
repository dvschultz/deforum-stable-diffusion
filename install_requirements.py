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
    "scikit-image==0.19.3",   # colors.maintain_colors (match_histograms)
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
PYTORCH_INDEX = "https://download.pytorch.org/whl/nightly/cu121"


def install_requirements(verbose=False, with_3d=False):

    # Detect System
    os_system = platform.system()
    print(f"system detected: {os_system}")

    pip_install_packages(CORE, verbose=verbose)

    if with_3d:
        print("..installing 3D/depth extras (torch stack -- this is large)")
        pip_install_packages(THREE_D, extra_index_url=PYTORCH_INDEX, verbose=verbose, pre=True)
    else:
        print("..skipping the torch stack: generation runs on fal.ai (Z-Image Turbo),")
        print("  and 2D animation / image batches need no GPU.")
        print("  For 3D depth warping or grid previews, re-run with --with-3d")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-3d', action='store_true',
                        help='also install torch + MiDaS/AdaBins deps for 3D depth warping')
    parser.add_argument('--verbose', action='store_true', help='print pip install stuff')
    args = parser.parse_args()
    install_requirements(verbose=args.verbose, with_3d=args.with_3d)