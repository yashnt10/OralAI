"""
test_training_page.py
=====================
Tests for the /model-training and /training page and API endpoints.
Verifies compliance with faculty rules (no external APIs) and checks:
1. When training/metrics.json is present: returns 200, shows exact metrics,
   confusion matrix, patient-wise split, training curves, per-class metrics.
2. When training/metrics.json is missing: returns 200 with "metrics not available".
3. Asset security: only confusion_matrix.png and training_curves.png are served.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    # Lifespan will load config and model
    with TestClient(app) as c:
        yield c


def test_model_training_page_renders_with_metrics(client):
    """Verify /model-training loads and displays real data from training/metrics.json."""
    response = client.get("/model-training")
    assert response.status_code == 200
    html = response.text

    # Check key metrics from metrics.json
    assert "71.22%" in html  # Macro-F1
    assert "72.01%" in html  # Balanced accuracy
    assert "93.59%" in html  # Overall accuracy
    assert "75.0%" in html   # Referral sensitivity
    assert "99.1%" in html   # Referral specificity

    # Check dataset & split details
    assert "SMART-OM" in html
    assert "2469" in html
    assert "StratifiedGroupKFold" in html
    assert "Zero Patient Leakage" in html

    # Check confusion matrix
    assert "Validation Confusion Matrix" in html

    # Check faculty rule notice
    assert "Runs locally · no external AI APIs" in html
    assert "locally on-device" in html

    # Check baseline all-normal metrics
    assert "24.05%" in html  # Baseline macro-f1
    assert "92.71%" in html  # Baseline accuracy


def test_training_alias_route(client):
    """Verify /training aliases to the same page."""
    response = client.get("/training")
    assert response.status_code == 200
    assert "Model &amp; Training Performance" in response.text or "Model & Training Performance" in response.text


def test_training_metrics_api(client):
    """Verify /api/training-metrics returns the parsed JSON."""
    response = client.get("/api/training-metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["evaluation"]["macro_f1"] == 71.22
    assert data["evaluation"]["balanced_accuracy"] == 72.01
    assert data["evaluation"]["accuracy"] == 93.59
    assert data["split"]["val_images"] == 343


def test_training_assets_served_safely(client):
    """Verify visual assets can be fetched, but arbitrary files are blocked."""
    # Allowed
    r_cm = client.get("/training/asset/confusion_matrix.png")
    assert r_cm.status_code == 200
    assert "image" in r_cm.headers.get("content-type", "")

    r_curves = client.get("/training/asset/training_curves.png")
    assert r_curves.status_code == 200
    assert "image" in r_curves.headers.get("content-type", "")

    # Blocked (e.g. notebook or arbitrary files)
    r_notebook = client.get("/training/asset/training_notebook.ipynb")
    assert r_notebook.status_code == 404

    r_invalid = client.get("/training/asset/random_file.png")
    assert r_invalid.status_code == 404


def test_missing_metrics_shows_metrics_not_available(client, monkeypatch):
    """Verify that if metrics.json is missing, the page displays 'metrics not available'."""
    from app import main as main_module

    # Mock load_training_metrics to return None
    monkeypatch.setattr(main_module, "load_training_metrics", lambda: None)

    response = client.get("/model-training")
    assert response.status_code == 200
    assert "metrics not available" in response.text

    api_response = client.get("/api/training-metrics")
    assert api_response.status_code == 404
    assert api_response.json()["message"] == "metrics not available"
