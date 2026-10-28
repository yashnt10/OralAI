"""
main.py
=======
FastAPI application entry point.

Startup
-------
- Reads config.yaml
- Loads OralCancerModel ONCE via lifespan context manager
- Mounts static files and Jinja2 templates

Endpoints
---------
GET  /                  -> Home page
GET  /screening         -> Screening page
GET  /result            -> Result page
GET  /about             -> About / model card page

GET  /api/health        -> {"status": "ok", "model_loaded": true}
GET  /api/model-info    -> Version, metrics, limitations
POST /api/predict       -> Main prediction endpoint
POST /api/explain       -> Grad-CAM / Attention Rollout heatmap
POST /api/explain/lime  -> LIME heatmap (slow, on-demand)
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from datetime import datetime, timezone
import json

import cv2
import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.model import OralCancerModel
from app.explainability import (
    AttentionRollout,
    compute_gradcam,
    compute_lime,
    format_confidence,
    get_224_image,
    make_explanation_figure_b64,
    overlay_heatmap,
)
from app.report import generate_screening_report_pdf
from app.schemas import (
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictResponse,
    ProbabilityItem,
    ValidationMetrics,
)
from app.utils import (
    bytes_to_bgr,
    check_image_quality,
    generate_case_id,
    ndarray_to_base64,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

# ── In-memory cache: case_id -> {img_bgr, cleaned_rgb, tensor_224, img_224_rgb, result, site} ──
_case_cache: dict[str, dict] = {}

# ── Config & model (populated in lifespan) ───────────────────────────────────
_config: dict = {}
_model: OralCancerModel | None = None
_attention_rollout: AttentionRollout | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load config + model on startup; clean up on shutdown."""
    global _config, _model, _attention_rollout

    config_path = Path("config.yaml")
    if not config_path.exists():
        raise RuntimeError("config.yaml not found. Run from the project root.")

    with open(config_path, "r") as f:
        _config = yaml.safe_load(f)

    logger.info("Loaded config.yaml")
    _model = OralCancerModel(_config)
    _attention_rollout = AttentionRollout(_model.model)
    logger.info("Model ready.")
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

app = FastAPI(
    title="Oral Pre-Cancer Screening App",
    description="Explainable AI screening tool for oral pre-cancer and cancer triage.",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to sanitize unexpected server errors and avoid leaking raw tracebacks."""
    logger.exception("Unhandled server exception: %s", exc)
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return HTMLResponse(
            content="""<!DOCTYPE html><html><head><title>Server Error — OralScan AI</title>
            <link rel="stylesheet" href="/static/css/style.css"></head><body>
            <main class="container" style="padding: 4rem 1rem; text-align: center;">
            <h1 style="color: #ef4444;">Server Error</h1>
            <p style="color: var(--text-muted); margin-top: 1rem;">An unexpected server error occurred while processing your request. Please try again.</p>
            <a href="/screening" class="btn btn-primary" style="margin-top: 1.5rem; display: inline-block;">Return to Screening</a>
            </main></body></html>""",
            status_code=500,
        )
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred while processing the request. Please try again."},
    )


# ══════════════════════════════════════════════════════════════════════════════
#  HTML pages
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/screening", response_class=HTMLResponse)
async def screening(request: Request):
    return templates.TemplateResponse(request=request, name="screening.html")


@app.get("/result", response_class=HTMLResponse)
async def result_page(request: Request):
    return templates.TemplateResponse(request=request, name="result.html")


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse(request=request, name="about.html")


def load_training_metrics() -> dict | None:
    """Read training/metrics.json if present; return None if missing or unreadable."""
    metrics_path = Path("training") / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Error reading training/metrics.json: %s", exc)
        return None


@app.get("/model-training", response_class=HTMLResponse)
@app.get("/training", response_class=HTMLResponse)
async def model_training(request: Request):
    metrics = load_training_metrics()
    has_curves_png = (Path("training") / "training_curves.png").exists()
    has_cm_png = (Path("training") / "confusion_matrix.png").exists()
    return templates.TemplateResponse(
        request=request,
        name="model_training.html",
        context={
            "metrics": metrics,
            "has_curves_png": has_curves_png,
            "has_cm_png": has_cm_png,
        },
    )


@app.get("/training/asset/{filename}")
async def training_asset(filename: str):
    """Serve specific training visual assets (curves and confusion matrix) safely."""
    allowed = {"confusion_matrix.png", "training_curves.png"}
    if filename not in allowed:
        raise HTTPException(404, "Asset not found.")
    path = Path("training") / filename
    if not path.exists():
        raise HTTPException(404, "Asset not found.")
    return FileResponse(path)


@app.get("/api/training-metrics")
async def api_training_metrics():
    """Return parsed training metrics JSON or 404 with 'metrics not available'."""
    metrics = load_training_metrics()
    if metrics is None:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "metrics not available"},
        )
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  API — health & info
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", model_loaded=_model is not None)


@app.get("/api/model-info", response_model=ModelInfoResponse)
async def model_info():
    if _model is None:
        raise HTTPException(503, "Model not loaded yet.")

    cfg = _config
    val = cfg["validation"]
    ci = f"{val['referral_sensitivity_ci_low']}–{val['referral_sensitivity_ci_high']}%"

    return ModelInfoResponse(
        version_label=cfg["model"]["version_label"],
        architecture=cfg["model"]["architecture"],
        num_classes=cfg["model"]["num_classes"],
        class_names=_model.class_names,
        actions=_model.actions,
        dataset=val["dataset"],
        n_images=val["n_images"],
        validation=ValidationMetrics(
            accuracy=val["accuracy"],
            macro_f1=val["macro_f1"],
            balanced_accuracy=val["balanced_accuracy"],
            referral_sensitivity=val["referral_sensitivity"],
            referral_sensitivity_ci=ci,
            referral_specificity=val["referral_specificity"],
            oral_cancer_detected=val["oral_cancer_detected"],
            opmd_detected=val["opmd_detected"],
        ),
        limitations=[
            "Validated on only 4 OPMD and 4 oral-cancer images — validation set is very small.",
            "Attention analysis showed the model sometimes focuses on teeth, fingers, and instruments rather than lesions.",
            "Prototype screening aid only — NOT a clinical diagnosis tool.",
            "Performance may differ on images taken with different lighting or camera hardware.",
        ],
    )


# ══════════════════════════════════════════════════════════════════════════════
#  API — predict
# ══════════════════════════════════════════════════════════════════════════════

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


@app.post("/api/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(..., description="Intraoral photograph"),
    site: str = Form(None, description="Intraoral anatomical site"),
):
    if _model is None:
        raise HTTPException(503, "Model not loaded.")

    # ── Validation ───────────────────────────────────────────────────────────
    # 1. File size check
    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            413,
            f"File too large ({len(raw_bytes) / 1e6:.1f} MB). Maximum allowed size is 20 MB. Please upload a smaller image file.",
        )

    # 2. File type and extension check
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if file.content_type not in ALLOWED_TYPES or (ext and ext not in ALLOWED_EXTENSIONS):
        raise HTTPException(
            415,
            f"Invalid file format '{filename or file.content_type}'. "
            "Please upload a valid intraoral photograph in JPEG or PNG format (.jpg, .jpeg, .png).",
        )

    # 3. Decode & corrupt image check
    try:
        img_bgr = bytes_to_bgr(raw_bytes)
    except ValueError as exc:
        raise HTTPException(
            422,
            "The uploaded file could not be decoded as a valid image. The file may be corrupt or in an unsupported format.",
        ) from exc

    # 4. Image resolution check (too small)
    h, w = img_bgr.shape[:2]
    if h < 224 or w < 224:
        raise HTTPException(
            422,
            f"Image resolution is too low ({w}×{h} px). Intraoral photographs must be at least 224×224 pixels for triage screening.",
        )

    # ── Predict ──────────────────────────────────────────────────────────────
    # Note: 'site' is recorded for clinical report metadata only; it is never passed to or used by the model.
    result = _model.predict(img_bgr)
    cleaned_rgb = result.pop("_cleaned_rgb")
    glare_mask = result.pop("_glare_mask")

    # ── Quality checks ───────────────────────────────────────────────────────
    thresholds = _config["thresholds"]
    quality_warnings = check_image_quality(img_bgr, glare_mask, thresholds)

    # ── Confidence warning ───────────────────────────────────────────────────
    confidence_warning = None
    if result["low_confidence"]:
        top_prob = max(result["probabilities"])
        top_pct_str = format_confidence(top_prob)
        confidence_warning = (
            f"Low confidence: top probability {top_pct_str} is below the "
            f"{thresholds['low_confidence'] * 100:.0f}% threshold. "
            "Specialist review advised regardless of tier."
        )

    # ── Case ID & cache ──────────────────────────────────────────────────────
    case_id = generate_case_id()
    from app.preprocessing import TF as _TF
    tensor_224 = _TF(cleaned_rgb).unsqueeze(0)       # (1, 3, 224, 224)
    img_224_rgb = get_224_image(cleaned_rgb)          # uint8 RGB 224×224
    _case_cache[case_id] = {
        "img_bgr": img_bgr,
        "cleaned_rgb": cleaned_rgb,
        "tensor_224": tensor_224,
        "img_224_rgb": img_224_rgb,
        "result": result,
        "site": site,
        "created_at": datetime.now(timezone.utc),
    }

    # ── Build response ───────────────────────────────────────────────────────
    class_names = _model.class_names
    prob_items = [
        ProbabilityItem(class_name=class_names[i], probability=p)
        for i, p in enumerate(result["probabilities"])
    ]

    return PredictResponse(
        case_id=case_id,
        tier_index=result["tier_index"],
        tier_name=result["tier_name"],
        action=result["action"],
        color=result["color"],
        probabilities=prob_items,
        quality_warnings=quality_warnings,
        confidence_warning=confidence_warning,
        cleaned_image_b64=ndarray_to_base64(cleaned_rgb),
        image_224_b64=ndarray_to_base64(img_224_rgb),
        site=site,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  API — explain (stub for Phase 3)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/explain", response_model=ExplainResponse)
async def explain(req: ExplainRequest):
    """
    Grad-CAM or Attention Rollout heatmap for an already-predicted case.
    method: 'gradcam' | 'attention_rollout'
    """
    if _model is None:
        raise HTTPException(503, "Model not loaded.")

    case = _case_cache.get(req.case_id)
    if case is None:
        raise HTTPException(404, f"Case '{req.case_id}' not found. Run /api/predict first.")

    tensor_224 = case["tensor_224"]
    img_224_rgb = case["img_224_rgb"]
    result = case["result"]
    tier_idx = result["tier_index"]
    tier_name = result["tier_name"]
    top_prob = result["probabilities"][tier_idx]

    method = req.method.lower()
    if method == "gradcam":
        heatmap = compute_gradcam(_model.model, tensor_224, tier_idx)
        label = "Grad-CAM"
        case["gradcam_overlay"] = overlay_heatmap(img_224_rgb, heatmap, alpha=0.50, colormap="jet")
    elif method == "attention_rollout":
        rollout_fn = _attention_rollout or AttentionRollout(_model.model)
        heatmap = rollout_fn(tensor_224)
        label = "Attention Rollout"
        case["rollout_overlay"] = overlay_heatmap(img_224_rgb, heatmap, alpha=0.50, colormap="jet")
    else:
        raise HTTPException(400, f"Unknown method '{req.method}'. Use 'gradcam' or 'attention_rollout'.")

    heatmap_b64, explanation_text = make_explanation_figure_b64(
        img_224_rgb, heatmap, label, tier_name, top_prob
    )
    return ExplainResponse(
        case_id=req.case_id,
        method=method,
        heatmap_b64=heatmap_b64,
        explanation_text=explanation_text,
    )


@app.post("/api/explain/lime", response_model=ExplainResponse)
async def explain_lime(req: ExplainRequest):
    """
    LIME superpixel explanation (slow, ~30-60 s on CPU).
    Only call when the user explicitly clicks 'Explain further'.
    """
    if _model is None:
        raise HTTPException(503, "Model not loaded.")

    case = _case_cache.get(req.case_id)
    if case is None:
        raise HTTPException(404, f"Case '{req.case_id}' not found. Run /api/predict first.")

    img_224_rgb = case["img_224_rgb"]
    result = case["result"]
    tier_idx = result["tier_index"]
    tier_name = result["tier_name"]
    top_prob = result["probabilities"][tier_idx]

    logger.info("Running LIME for case %s (takes ~2-3 min on CPU) …", req.case_id)
    lime_overlay = compute_lime(_model.model, img_224_rgb, tier_idx, num_samples=300, random_state=42)
    case["lime_overlay"] = lime_overlay

    heatmap_b64, explanation_text = make_explanation_figure_b64(
        img_224_rgb, lime_overlay, "LIME", tier_name, top_prob
    )
    return ExplainResponse(
        case_id=req.case_id,
        method="lime",
        heatmap_b64=heatmap_b64,
        explanation_text=explanation_text,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  API — PDF Report (in-memory streaming)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/report/{case_id}/pdf")
async def download_pdf_report(case_id: str):
    """
    Generate and stream an in-memory PDF clinical triage report.
    No PDF is saved to disk on the server.
    """
    if _model is None:
        raise HTTPException(503, "Model not loaded.")

    case = _case_cache.get(case_id)
    if case is None:
        raise HTTPException(404, f"Case '{case_id}' not found. Please run screening first.")

    tensor_224 = case["tensor_224"]
    img_224_rgb = case["img_224_rgb"]
    result = case["result"]
    tier_idx = result["tier_index"]

    # Ensure Grad-CAM overlay is generated
    if "gradcam_overlay" not in case:
        cam_hm = compute_gradcam(_model.model, tensor_224, tier_idx)
        case["gradcam_overlay"] = overlay_heatmap(img_224_rgb, cam_hm, alpha=0.50, colormap="jet")

    # Ensure Attention Rollout overlay is generated
    if "rollout_overlay" not in case:
        rollout_fn = _attention_rollout or AttentionRollout(_model.model)
        ro_hm = rollout_fn(tensor_224)
        case["rollout_overlay"] = overlay_heatmap(img_224_rgb, ro_hm, alpha=0.50, colormap="jet")

    lime_overlay = case.get("lime_overlay")

    report_payload = {
        "case_id": case_id,
        "timestamp": case.get("created_at", datetime.now(timezone.utc)),
        "site": case.get("site"),
        "result": result,
        "img_224_rgb": img_224_rgb,
        "gradcam_overlay": case["gradcam_overlay"],
        "rollout_overlay": case["rollout_overlay"],
        "lime_overlay": lime_overlay,
        "model_version": _config.get("model", {}).get(
            "version_label", "ViT-B/16 trained on SMART-OM (2,469 smartphone images)"
        ),
    }

    pdf_bytes = generate_screening_report_pdf(report_payload)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="OralScan_Report_{case_id}.pdf"',
            "Cache-Control": "no-cache, no-store, must-revalidate",
        },
    )
