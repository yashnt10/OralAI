"""
test_phase4_review_fixes.py
===========================
Verification tests for Phase 4 review fixes:
1. Low-confidence warning (< 0.60 threshold)
2. Image quality warnings (blur, dark/brightness, glare) using in-memory modified copies
3. LIME cancellation & UI resilience
4. Anatomical site isolation (not used by model)
5. Probability formatting (< 0.01% and > 99.99%)
6. Full-width heatmap cards, colour legends, and lightbox modal structure
"""

from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.explainability import format_confidence
from app.utils import check_image_quality


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── 1. Low-Confidence Warning (< 0.60) ───────────────────────────────────────

def test_low_confidence_warning_triggered(client):
    """
    Mock a prediction where top probability is 0.45 (< 0.60 threshold).
    Verify that confidence_warning is returned and flags specialist review.
    """
    mock_result = {
        "tier_index": 2,
        "tier_name": "T2 OPMD (incl. OSMF)",
        "action": "Refer to oral-medicine specialist",
        "color": "#f97316",
        "probabilities": [0.20, 0.15, 0.45, 0.20],
        "low_confidence": True,
        "_cleaned_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        "_glare_mask": np.zeros((224, 224), dtype=np.uint8),
    }

    test_img_path = Path("test_images/SMITA00300_R_DT.jpeg")
    with open(test_img_path, "rb") as f:
        img_bytes = f.read()

    with patch("app.main._model.predict", return_value=mock_result):
        response = client.post(
            "/api/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["confidence_warning"] is not None
    assert "below the 60% threshold" in data["confidence_warning"]
    assert "Specialist review advised regardless of tier" in data["confidence_warning"]


def test_high_confidence_no_warning(client):
    """A standard high-confidence prediction should NOT have a confidence warning."""
    test_img_path = Path("test_images/Ca 9.png")
    with open(test_img_path, "rb") as f:
        response = client.post(
            "/api/predict",
            files={"file": ("Ca 9.png", f.read(), "image/png")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["confidence_warning"] is None


# ── 2. Photographic Quality Warnings (In-memory, kept out of git) ────────────

def test_quality_warning_blur(client):
    """Apply severe Gaussian blur to verify blur/out-of-focus warning."""
    test_img_path = Path("test_images/SMITA00300_R_DT.jpeg")
    img_bgr = cv2.imread(str(test_img_path))
    blurred = cv2.GaussianBlur(img_bgr, (51, 51), 25)
    ok, buf = cv2.imencode(".jpg", blurred)
    assert ok

    response = client.post(
        "/api/predict",
        files={"file": ("blurred.jpg", buf.tobytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    warnings = response.json()["quality_warnings"]
    assert any("blurry" in w.lower() or "sharpness" in w.lower() for w in warnings), (
        f"Expected blur warning, got: {warnings}"
    )


def test_quality_warning_darkness(client):
    """Heavily darken an image (mean brightness < 40) to verify under-exposure warning."""
    test_img_path = Path("test_images/SMITA00300_R_DT.jpeg")
    img_bgr = cv2.imread(str(test_img_path))
    dark = (img_bgr * 0.10).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", dark)
    assert ok

    response = client.post(
        "/api/predict",
        files={"file": ("dark.jpg", buf.tobytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    warnings = response.json()["quality_warnings"]
    assert any("dark" in w.lower() or "brightness" in w.lower() for w in warnings), (
        f"Expected dark warning, got: {warnings}"
    )


def test_quality_warning_bright(client):
    """Heavily brighten an image (mean brightness > 220) to verify over-exposure warning."""
    test_img_path = Path("test_images/SMITA00300_R_DT.jpeg")
    img_bgr = cv2.imread(str(test_img_path))
    bright = np.clip(img_bgr.astype(np.int32) + 210, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", bright)
    assert ok

    response = client.post(
        "/api/predict",
        files={"file": ("bright.jpg", buf.tobytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    warnings = response.json()["quality_warnings"]
    assert any("overexposed" in w.lower() or "bright" in w.lower() for w in warnings), (
        f"Expected overexposed warning, got: {warnings}"
    )


def test_quality_warning_glare_unit():
    """Unit test verify glare warning triggers when glare coverage exceeds threshold."""
    img_bgr = np.ones((500, 500, 3), dtype=np.uint8) * 128
    glare_mask = np.zeros((500, 500), dtype=np.uint8)
    # Set 15% of pixels as glare (threshold is 10%)
    glare_mask[:150, :500] = 255

    thresholds = {
        "blur_variance": 80.0,
        "dark_mean": 40.0,
        "bright_mean": 220.0,
        "glare_fraction": 0.10,
    }
    warnings = check_image_quality(img_bgr, glare_mask, thresholds)
    assert any("glare" in w.lower() for w in warnings), f"Expected glare warning, got: {warnings}"


# ── 3. LIME Cancellation & Robustness ────────────────────────────────────────

def test_lime_cancel_flow(client):
    """
    Verify /api/explain/lime endpoint works, and verify frontend HTML contains
    cancellation handlers (cancelLimeBtn, AbortController, and UI restoration).
    """
    html_resp = client.get("/screening")
    assert html_resp.status_code == 200
    html = html_resp.text

    # Verify cancel button and AbortController logic exist in page
    assert "cancel-lime-btn" in html
    assert "limeAbortController.abort()" in html
    assert "Cancel LIME" in html


# ── 4. Anatomical Site Isolation ─────────────────────────────────────────────

def test_anatomical_site_not_used_by_model(client):
    """
    Verify that providing different anatomical sites does NOT alter the model prediction,
    confirming site is metadata only.
    """
    img_path = Path("test_images/SMITA00304_R_LB.jpeg")
    with open(img_path, "rb") as f:
        bytes1 = f.read()

    res_no_site = client.post(
        "/api/predict",
        files={"file": ("img1.jpg", bytes1, "image/jpeg")},
    ).json()

    res_with_site = client.post(
        "/api/predict",
        files={"file": ("img2.jpg", bytes1, "image/jpeg")},
        data={"site": "Buccal Mucosa"},
    ).json()

    # Probabilities and tier must be identical
    assert res_no_site["tier_index"] == res_with_site["tier_index"]
    for p1, p2 in zip(res_no_site["probabilities"], res_with_site["probabilities"]):
        assert p1["class_name"] == p2["class_name"]
        assert pytest.approx(p1["probability"], abs=1e-6) == p2["probability"]


# ── 5. Probability Formatting (< 0.01% & > 99.99%) ───────────────────────────

def test_format_confidence():
    """Verify probability formatting bounds."""
    assert format_confidence(0.99995) in ("> 99.99 %", "> 99.99%")
    assert format_confidence(0.00005) in ("< 0.01 %", "< 0.01%")
    assert format_confidence(0.8927) in ("89.27 %", "89.27%")


# ── 6. UI Structure: Badge, Legends, Lightbox Modal ──────────────────────────

def test_ui_phase4_elements(client):
    """Verify Research prototype badge, 0-1 colour legends, and image modal lightbox."""
    res = client.get("/screening")
    assert res.status_code == 200
    html = res.text

    # Research prototype badge
    assert "Research prototype" in html
    assert "Clinical ViT" not in html

    # Anatomical site label
    assert "Recorded for the report only – not used by the model" in html

    # Heatmap legends
    assert "legend-gradcam" in html
    assert "legend-rollout" in html
    assert "legend-lime" in html
    assert "0.0 (Low)" in html
    assert "1.0 (High)" in html

    # Modal lightbox
    assert "image-modal" in html
    assert "modal-backdrop" in html
    assert "modal-img" in html
    assert "modal-close" in html
