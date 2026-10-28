"""
schemas.py
==========
Pydantic models for all API request / response bodies.
"""

from pydantic import BaseModel, Field


# ── /api/predict ─────────────────────────────────────────────────────────────

class ProbabilityItem(BaseModel):
    class_name: str
    probability: float


class PredictResponse(BaseModel):
    case_id: str = Field(..., description="Anonymous case ID, e.g. SCR-20261026-0001")
    tier_index: int = Field(..., ge=0, le=3)
    tier_name: str
    action: str
    color: str = Field(..., description="Hex colour for UI badge")
    probabilities: list[ProbabilityItem]
    quality_warnings: list[str] = Field(default_factory=list)
    confidence_warning: str | None = Field(
        None,
        description="Set when top probability < threshold",
    )
    cleaned_image_b64: str = Field(
        ...,
        description="Base64 data-URI of the cleaned image (glare removed + colour-corrected)",
    )
    image_224_b64: str | None = Field(
        None,
        description="Base64 data-URI of the preprocessed 224x224 image that the model sees",
    )
    site: str | None = Field(None, description="Intraoral site selected by the user")


# ── /api/explain ─────────────────────────────────────────────────────────────

class ExplainRequest(BaseModel):
    case_id: str
    method: str = Field(
        "gradcam",
        description="'gradcam' | 'attention_rollout' | 'lime'",
    )


class ExplainResponse(BaseModel):
    case_id: str
    method: str
    heatmap_b64: str = Field(
        ...,
        description="Base64 data-URI of the heatmap overlay",
    )
    explanation_text: str = Field(
        ...,
        description="One plain-language sentence explaining what the map shows",
    )


# ── /api/model-info ──────────────────────────────────────────────────────────

class ValidationMetrics(BaseModel):
    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    referral_sensitivity: float
    referral_sensitivity_ci: str
    referral_specificity: float
    oral_cancer_detected: str
    opmd_detected: str


class ModelInfoResponse(BaseModel):
    version_label: str
    architecture: str
    num_classes: int
    class_names: list[str]
    actions: list[str]
    dataset: str
    n_images: int
    validation: ValidationMetrics
    limitations: list[str]


# ── /api/health ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
