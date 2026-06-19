"""Z-Image Turbo (fal.ai) client wrapper.

This module is the only place that talks to the hosted Z-Image Turbo model. It
replaces the in-process Stable Diffusion sampler: callers hand it a prompt (and,
for image-to-image, a PIL init image plus a strength), and get back PIL images.

All animation logic (warping, scheduling, color coherence, video assembly) lives
above this module and is model-agnostic. Nothing here knows about frames.

Endpoints (fal.ai):
  - text-to-image:   fal-ai/z-image/turbo
  - image-to-image:  fal-ai/z-image/turbo/image-to-image   (image_url + strength)
  - inpaint:         fal-ai/z-image/turbo/inpaint           (image_url + mask_url)

Auth is via the FAL_KEY environment variable (fal-client's native convention).
"""

# Standard library imports
import io
import os
import time

# Related third-party imports
import requests
from PIL import Image

import fal_client


# fal.ai endpoint identifiers
ENDPOINT_TXT2IMG = "fal-ai/z-image/turbo"
ENDPOINT_IMG2IMG = "fal-ai/z-image/turbo/image-to-image"
ENDPOINT_INPAINT = "fal-ai/z-image/turbo/inpaint"

# Z-Image Turbo is distilled to a small step budget.
MAX_STEPS = 8
MIN_STEPS = 1

# Per-call timeouts (seconds) so a stuck fal queue raises and the retry loop
# engages, instead of hanging an unattended render indefinitely.
START_TIMEOUT = 120
CLIENT_TIMEOUT = 300

# Substrings that mark a non-retryable authentication/authorization failure.
_AUTH_ERROR_MARKERS = ("401", "403", "unauthorized", "forbidden", "invalid api key", "fal_key")

_dotenv_loaded = False


def load_dotenv():
    """Load KEY=VALUE pairs from a `.env` (cwd, then repo root) into os.environ, once.

    Dependency-free (honors the 'fal-client only' constraint). A real environment
    variable always wins -- keys already set in os.environ are never overwritten,
    so `.env` is only a fallback. `export KEY=val` lines and quoted values are
    tolerated. Missing/unreadable files are ignored.
    """
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    seen = set()
    for path in (os.path.join(os.getcwd(), ".env"), os.path.join(repo_root, ".env")):
        if path in seen or not os.path.isfile(path):
            continue
        seen.add(path)
        try:
            with open(path) as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    if line.startswith("export "):
                        line = line[len("export "):]
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
        except OSError:
            pass


def resolve_fal_key():
    """Return the FAL_KEY from the environment (or a local .env), or raise clearly.

    fal-client reads FAL_KEY itself; this exists so a missing key fails loudly at
    setup time with an actionable message instead of on the first frame. If the key
    is not already in the environment, a repo-root/cwd `.env` is consulted.
    """
    if not os.environ.get("FAL_KEY", "").strip():
        load_dotenv()
    key = os.environ.get("FAL_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "FAL_KEY is not set. Generation runs on Z-Image Turbo via fal.ai. "
            "Create a key at https://fal.ai/dashboard/keys and set it, e.g. "
            "`export FAL_KEY=...` (or paste it into the notebook's fal.ai API Key cell)."
        )
    return key


def to_fal_strength(deforum_strength):
    """Convert Deforum's strength convention to fal's.

    Deforum: higher strength -> fewer denoising steps -> output closer to the
    init image (MORE coherence). fal z-image image-to-image: LOWER strength
    preserves more of the source. The two are inverses, so map and clamp.
    """
    return max(0.0, min(1.0, 1.0 - float(deforum_strength)))


def clamp_steps(steps):
    """Clamp a requested step count to Z-Image Turbo's 1-8 budget."""
    try:
        steps = int(steps)
    except (TypeError, ValueError):
        steps = MAX_STEPS
    return max(MIN_STEPS, min(MAX_STEPS, steps))


def image_size_arg(W, H):
    """Return a custom fal image_size object. W/H are already forced to /64 upstream."""
    return {"width": int(W), "height": int(H)}


def _is_auth_error(exc):
    msg = str(exc).lower()
    return any(marker in msg for marker in _AUTH_ERROR_MARKERS)


def _submit(endpoint, arguments, max_retries=4, base_delay=1.0, sleep=time.sleep):
    """Synchronously run a fal endpoint with bounded retry on transient failures.

    Auth errors fail fast (retrying a bad key is pointless); anything else is
    treated as transient (network blip, 5xx, timeout) and retried with backoff so
    a single hiccup doesn't abort a long render.
    """
    resolve_fal_key()
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fal_client.subscribe(
                endpoint,
                arguments=arguments,
                start_timeout=START_TIMEOUT,
                client_timeout=CLIENT_TIMEOUT,
            )
        except Exception as exc:  # noqa: BLE001 - fal/httpx raise varied types
            if _is_auth_error(exc):
                raise
            last_exc = exc
            if attempt < max_retries - 1:
                sleep(base_delay * (2 ** attempt))
    raise RuntimeError(
        f"Z-Image Turbo request to {endpoint} failed after {max_retries} attempts: {last_exc}"
    )


def _upload(image):
    """Upload a PIL image to fal as PNG and return its URL.

    Resolves the key first: uploads happen before _submit in img2img/inpaint, so a
    key that lives only in .env must be loaded here too -- otherwise fal_client
    raises MissingCredentialsError before _submit ever runs.

    PNG (not the client default of JPEG) matters here: init frames are re-uploaded
    every animation frame, so lossy recompression would compound artifacts and
    degrade coherence, and a JPEG-compressed mask gets soft edges that shift which
    pixels the inpaint endpoint treats as masked.
    """
    resolve_fal_key()
    return fal_client.upload_image(image, format="png")


def _download_image(url):
    """Fetch a result image URL and return a PIL RGB image."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def _result_to_images(result):
    """Extract PIL images from a fal result payload."""
    images = (result or {}).get("images") or []
    out = []
    for entry in images:
        url = entry.get("url") if isinstance(entry, dict) else None
        if url:
            out.append(_download_image(url))
    return out


def _base_arguments(prompt, W, H, seed, steps, num_images, acceleration):
    args = {
        "prompt": prompt,
        "image_size": image_size_arg(W, H),
        "num_inference_steps": clamp_steps(steps),
        "num_images": int(num_images),
        "output_format": "png",
    }
    if acceleration and acceleration != "none":
        args["acceleration"] = acceleration
    if seed is not None:
        args["seed"] = int(seed)
    return args


def txt2img(prompt, W, H, seed=None, steps=MAX_STEPS, num_images=1, acceleration="regular"):
    """Text-to-image. Returns a list of PIL images (length == num_images)."""
    arguments = _base_arguments(prompt, W, H, seed, steps, num_images, acceleration)
    return _result_to_images(_submit(ENDPOINT_TXT2IMG, arguments))


def img2img(prompt, init_image, deforum_strength, W, H, seed=None, steps=MAX_STEPS,
            num_images=1, acceleration="regular"):
    """Image-to-image from a PIL init image. `deforum_strength` uses Deforum's
    convention and is inverted internally for fal. Returns a list of PIL images."""
    arguments = _base_arguments(prompt, W, H, seed, steps, num_images, acceleration)
    arguments["image_url"] = _upload(init_image)
    arguments["strength"] = to_fal_strength(deforum_strength)
    return _result_to_images(_submit(ENDPOINT_IMG2IMG, arguments))


def inpaint(prompt, init_image, mask_image, deforum_strength, W, H, seed=None,
            steps=MAX_STEPS, num_images=1, acceleration="regular"):
    """Masked generation via the inpaint endpoint. White areas of the mask change.
    Returns a list of PIL images."""
    arguments = _base_arguments(prompt, W, H, seed, steps, num_images, acceleration)
    arguments["image_url"] = _upload(init_image)
    arguments["mask_url"] = _upload(mask_image)
    arguments["strength"] = to_fal_strength(deforum_strength)
    return _result_to_images(_submit(ENDPOINT_INPAINT, arguments))
