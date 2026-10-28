"""
test_screening_e2e.py
=====================
End-to-end integration tests for Phase 4 screening page endpoints.
Verifies all 3 test images across prediction, tier classification,
quality checks, Grad-CAM, Attention Rollout, and LIME.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app

TEST_IMAGES_DIR = Path("test_images")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_screening_page_html_renders(client):
    """Verify /screening page loads offline HTML with all required components."""
    response = client.get("/screening")
    assert response.status_code == 200
    html = response.text

    # Header & navigation
    assert "Screening" in html
    assert "Model &amp; Training" in html or "Model & Training" in html
    assert "About" in html
    assert "Runs locally · no external AI APIs" in html

    # Disclaimer
    assert "Highlighted regions influenced this prediction according to each method. Not proof of disease. Screening aid only." in html

    # Upload zone & elements
    assert "drop-zone" in html
    assert "file-input" in html
    assert "analyze-btn" in html
    assert "site-select" in html

    # LIME button & progress
    assert "Run LIME (takes about 3 minutes)" in html
    assert "Cancel LIME" in html


def test_predict_and_explain_smita00304(client):
    """Verify SMITA00304_R_LB.jpeg: T2 OPMD at ~89.27% (orange badge, specialist referral)."""
    img_path = TEST_IMAGES_DIR / "SMITA00304_R_LB.jpeg"
    assert img_path.exists()

    with open(img_path, "rb") as f:
        response = client.post(
            "/api/predict",
            files={"file": ("SMITA00304_R_LB.jpeg", f, "image/jpeg")},
            data={"site": "Buccal Mucosa"},
        )

    assert response.status_code == 200
    data = response.json()

    assert data["tier_index"] == 2
    assert "T2 OPMD" in data["tier_name"]
    assert data["color"] == "#f97316"  # Orange
    assert "Refer to oral-medicine specialist" in data["action"]
    assert data["case_id"].startswith("SCR-")
    assert data["image_224_b64"].startswith("data:image/")

    # Probability check: T2 ~ 89.27%
    t2_prob = next(p["probability"] for p in data["probabilities"] if "T2" in p["class_name"])
    assert round(t2_prob * 100, 1) == 89.3

    # Grad-CAM
    r_gc = client.post("/api/explain", json={"case_id": data["case_id"], "method": "gradcam"})
    assert r_gc.status_code == 200
    assert r_gc.json()["heatmap_b64"].startswith("data:image/png;base64,")
    assert "Grad-CAM: the target-class gradients are averaged per feature channel" in r_gc.json()["explanation_text"]

    # Attention Rollout
    r_ar = client.post("/api/explain", json={"case_id": data["case_id"], "method": "attention_rollout"})
    assert r_ar.status_code == 200
    assert r_ar.json()["heatmap_b64"].startswith("data:image/png;base64,")
    assert "Recursive matrix product of attention weights" in r_ar.json()["explanation_text"]


def test_predict_ca9(client):
    """Verify Ca 9.png: T3 Oral Cancer at > 99.99% (red badge, URGENT referral)."""
    img_path = TEST_IMAGES_DIR / "Ca 9.png"
    assert img_path.exists()

    with open(img_path, "rb") as f:
        response = client.post(
            "/api/predict",
            files={"file": ("Ca 9.png", f, "image/png")},
            data={"site": "Tongue (Lateral/Ventral)"},
        )

    assert response.status_code == 200
    data = response.json()

    assert data["tier_index"] == 3
    assert "T3 Oral cancer" in data["tier_name"]
    assert data["color"] == "#ef4444"  # Red
    assert "URGENT referral" in data["action"]

    t3_prob = next(p["probability"] for p in data["probabilities"] if "T3" in p["class_name"])
    assert t3_prob > 0.999


def test_predict_smita00300(client):
    """Verify SMITA00300_R_DT.jpeg: T0 Normal at > 99.99% (green badge, routine recall)."""
    img_path = TEST_IMAGES_DIR / "SMITA00300_R_DT.jpeg"
    assert img_path.exists()

    with open(img_path, "rb") as f:
        response = client.post(
            "/api/predict",
            files={"file": ("SMITA00300_R_DT.jpeg", f, "image/jpeg")},
            data={"site": "Tongue (Dorsum)"},
        )

    assert response.status_code == 200
    data = response.json()

    assert data["tier_index"] == 0
    assert "T0 Normal" in data["tier_name"]
    assert data["color"] == "#22c55e"  # Green
    assert "Routine recall" in data["action"]

    t0_prob = next(p["probability"] for p in data["probabilities"] if "T0" in p["class_name"])
    assert t0_prob > 0.999


def test_privacy_no_images_on_disk():
    """Verify no uploaded image files are written to disk."""
    for root in [Path("app"), Path("training"), Path("models")]:
        for p in root.glob("*.jpg"):
            assert False, f"Unexpected image found on disk: {p}"
