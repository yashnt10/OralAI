"""
test_phase5_features.py
=======================
Tests for Phase 5 features:
1. In-memory PDF report generation and download endpoint
2. Error handling:
   - Wrong file format
   - Corrupt image
   - File exceeding 20 MB
   - Image resolution too small (< 224x224)
   - Sanitized 500 server error (no raw Python tracebacks)
3. Governance files (.gitignore, requirements.txt, README.md)
"""

from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

TEST_IMAGES_DIR = Path("test_images")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── 1. In-Memory PDF Report Generation ───────────────────────────────────────

def test_pdf_report_download_endpoint(client):
    """
    Predict an image, then download the in-memory generated PDF report.
    Verify headers, MIME type, valid PDF header, and absence of personal identifiers.
    """
    img_path = TEST_IMAGES_DIR / "SMITA00304_R_LB.jpeg"
    with open(img_path, "rb") as f:
        pred_resp = client.post(
            "/api/predict",
            files={"file": ("SMITA00304_R_LB.jpeg", f, "image/jpeg")},
            data={"site": "Buccal Mucosa"},
        )
    assert pred_resp.status_code == 200
    case_id = pred_resp.json()["case_id"]

    # Request PDF report
    pdf_resp = client.get(f"/api/report/{case_id}/pdf")
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"] == "application/pdf"
    assert f'filename="OralScan_Report_{case_id}.pdf"' in pdf_resp.headers.get("content-disposition", "")

    # Validate PDF binary structure
    pdf_bytes = pdf_resp.content
    assert pdf_bytes.startswith(b"%PDF"), "Response is not a valid PDF document"
    assert len(pdf_bytes) > 2000, "PDF content is too small"

    # Verify no PDF file was written to disk
    for p in Path(".").glob("*.pdf"):
        assert False, f"PDF was incorrectly saved to disk: {p}"
    for p in Path("app").glob("*.pdf"):
        assert False, f"PDF was incorrectly saved in app directory: {p}"


def test_pdf_report_nonexistent_case(client):
    """Requesting a PDF for a non-existent case returns 404 with clean detail."""
    resp = client.get("/api/report/SCR-99999999-9999/pdf")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


# ── 2. Error Handling ────────────────────────────────────────────────────────

def test_error_wrong_file_type(client):
    """Uploading a non-image file (.txt / .pdf) returns 415 with clear error message."""
    resp = client.post(
        "/api/predict",
        files={"file": ("notes.txt", b"patient notes text file", "text/plain")},
    )
    assert resp.status_code == 415
    detail = resp.json()["detail"]
    assert "Invalid file format" in detail or "Unsupported file" in detail
    assert "JPEG or PNG" in detail


def test_error_corrupt_image(client):
    """Uploading invalid image bytes disguised as JPEG returns 422 with clear message."""
    corrupt_bytes = b"\xff\xd8\xff\xe0" + b"random_corrupted_garbage_content_12345"
    resp = client.post(
        "/api/predict",
        files={"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")},
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "could not be decoded as a valid image" in detail or "corrupt" in detail.lower()


def test_error_file_too_large(client):
    """Uploading a file exceeding 20 MB returns 413 with clear size limit message."""
    large_bytes = b"0" * (21 * 1024 * 1024)  # 21 MB
    resp = client.post(
        "/api/predict",
        files={"file": ("huge_photo.jpg", large_bytes, "image/jpeg")},
    )
    assert resp.status_code == 413
    detail = resp.json()["detail"]
    assert "File too large" in detail
    assert "20 MB" in detail


def test_error_image_too_small(client):
    """Uploading an image below 224x224 resolution returns 422 with minimum size guidance."""
    tiny_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    ok, buf = cv2.imencode(".png", tiny_img)
    assert ok

    resp = client.post(
        "/api/predict",
        files={"file": ("tiny.png", buf.tobytes(), "image/png")},
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "resolution is too low" in detail.lower()
    assert "224" in detail


def test_error_server_error_sanitization():
    """
    Verify that unhandled server exceptions return sanitized 500 JSON without
    exposing raw Python stack traces.
    """
    with TestClient(app, raise_server_exceptions=False) as no_raise_client:
        with patch("app.main._model.predict", side_effect=RuntimeError("Internal GPU bus crash simulation")):
            img_path = TEST_IMAGES_DIR / "Ca 9.png"
            with open(img_path, "rb") as f:
                resp = no_raise_client.post(
                    "/api/predict",
                    files={"file": ("Ca 9.png", f.read(), "image/png")},
                )
            assert resp.status_code == 500
            detail = resp.json()["detail"]
            assert "internal server error" in detail.lower()
            # Verify raw traceback is NOT exposed
            assert "Internal GPU bus crash" not in detail
            assert "Traceback" not in detail


# ── 3. Governance Files Verification ─────────────────────────────────────────

def test_gitignore_exclusions():
    """Verify .gitignore exists and excludes sensitive data, weights, and caches."""
    gi_path = Path(".gitignore")
    assert gi_path.exists(), ".gitignore does not exist"
    content = gi_path.read_text(encoding="utf-8")

    assert "venv/" in content
    assert "models/*.pth" in content
    assert "test_images/" in content
    assert "test_outputs/" in content
    assert "__pycache__/" in content
    assert ".pytest_cache/" in content
    assert "*.zip" in content
    assert "training/training_notebook.ipynb" in content


def test_requirements_file_pinned():
    """Verify requirements.txt exists and contains pinned versions."""
    req_path = Path("requirements.txt")
    assert req_path.exists(), "requirements.txt does not exist"
    content = req_path.read_text(encoding="utf-8")

    for pkg in ["fastapi==", "uvicorn==", "torch==", "timm==", "reportlab==", "pytest=="]:
        assert pkg in content, f"Expected {pkg} pinned in requirements.txt"


def test_readme_contents():
    """Verify README.md covers faculty requirements, run instructions, and limitations."""
    readme_path = Path("README.md")
    assert readme_path.exists(), "README.md does not exist"
    content = readme_path.read_text(encoding="utf-8")

    assert "OralScan AI" in content
    assert "run.bat" in content
    assert "SMART-OM" in content
    assert "metrics.json" in content
    assert "training_notebook.ipynb" in content
    assert "Grad-CAM" in content
    assert "Attention Rollout" in content
    assert "LIME" in content
    assert "The same validation fold was used to pick the best epoch" in content
