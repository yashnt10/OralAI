"""
utils.py
========
Case ID generator, image quality checks, base64 helpers.
"""

import base64
import re
from datetime import datetime

import cv2
import numpy as np


# ── Case ID ──────────────────────────────────────────────────────────────────

# Thread-local counter per day (simple in-memory; resets on server restart)
_case_counters: dict[str, int] = {}


def generate_case_id() -> str:
    """
    Generate an anonymous case ID like  SCR-20261026-0001.
    Counter resets on each new UTC day.
    """
    today = datetime.utcnow().strftime("%Y%m%d")
    _case_counters[today] = _case_counters.get(today, 0) + 1
    return f"SCR-{today}-{_case_counters[today]:04d}"


# ── Image Quality Checks ─────────────────────────────────────────────────────

def check_image_quality(img_bgr: np.ndarray, glare_mask: np.ndarray, thresholds: dict) -> list[str]:
    """
    Return a list of warning strings. Empty list = no quality issues.

    Checks
    ------
    1. Blur  : variance of Laplacian on gray image
    2. Dark  : mean pixel value of gray image
    3. Bright: mean pixel value of gray image
    4. Glare : fraction of pixels flagged by glare mask
    """
    warnings: list[str] = []
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Blur
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < thresholds["blur_variance"]:
        warnings.append(
            f"Image may be blurry (sharpness score {lap_var:.1f} < {thresholds['blur_variance']}). "
            "Try retaking with the camera steady."
        )

    # Brightness
    mean_val = float(gray.mean())
    if mean_val < thresholds["dark_mean"]:
        warnings.append(
            f"Image appears too dark (mean brightness {mean_val:.1f}). "
            "Use a dental light or flash."
        )
    elif mean_val > thresholds["bright_mean"]:
        warnings.append(
            f"Image appears overexposed (mean brightness {mean_val:.1f}). "
            "Reduce ambient light or move the camera."
        )

    # Glare coverage
    glare_fraction = float((glare_mask > 0).mean())
    if glare_fraction > thresholds["glare_fraction"]:
        warnings.append(
            f"High glare detected ({glare_fraction * 100:.1f}% of image). "
            "Reposition the light source; glare has been removed by inpainting."
        )

    return warnings


# ── Base-64 helpers ───────────────────────────────────────────────────────────

def ndarray_to_base64(img_rgb: np.ndarray, fmt: str = ".jpg") -> str:
    """Encode an RGB numpy array to a base64 data-URI string."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(fmt, img_bgr)
    if not ok:
        raise ValueError("Could not encode image to base64.")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    mime = "image/jpeg" if fmt in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64,{b64}"


def bytes_to_bgr(data: bytes) -> np.ndarray:
    """Decode raw image bytes to a BGR numpy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image — file may be corrupt or unsupported.")
    return img
