# OralScan AI — Intraoral Pre-Cancer Screening & Triage

An Explainable Artificial Intelligence (XAI) clinical screening prototype for early oral pre-cancer and cancer triage using smartphone intraoral photographs. 

OralScan AI runs a locally trained Vision Transformer (`ViT-B/16`) to stratify intraoral photographs into 4 actionable triage tiers, paired with three transparent explainability methods (Grad-CAM, Attention Rollout, and LIME) and in-memory downloadable PDF reports.

---

## Faculty & Course Project Compliance

This project strictly satisfies all requirements for the AI in Healthcare course:

1. **Own Model Trained on a Healthcare Dataset**:
   - Model architecture: `timm vit_base_patch16_224` (85.8M parameters, patch size 16, input $224 \times 224$).
   - Trained on Kaggle using a Tesla T4 GPU (run label `URK23CS7132`).
   - Training notebook: [`training/training_notebook.ipynb`](file:///c:/AI_Healthcare_App/training/training_notebook.ipynb) (contains full training pipeline, augmentation, loss functions, and learning rate scheduling).
   - Metrics and validation records: [`training/metrics.json`](file:///c:/AI_Healthcare_App/training/metrics.json).
   - Verified local checkpoint: `models/best_4tier_vit.pth`.

2. **Validation & Evaluation Results (343 held-out validation images)**:
   - Split methodology: 5-Fold `StratifiedGroupKFold` grouped by subject ID (so no patient appears in both training and validation partitions). Fold 1 was held out for evaluation.
   - **Overall Accuracy**: **93.59 %** (vs. 92.71 % all-Normal baseline).
   - **Macro-F1 Score**: **71.22 %** (vs. 24.05 % all-Normal baseline).
   - **Balanced Accuracy**: **72.01 %**.
   - **High-Risk Referral Sensitivity (T2 OPMD or T3 Oral cancer)**: **75.0 %** (95% CI: [40.9 %, 92.9 %]).
   - **Referral Specificity**: **99.1 %**.
   - **Oral Cancer Recall**: **100.0 %** (4 of 4 oral cancer cases correctly identified).

3. **Zero Third-Party AI APIs**:
   - **100% on-device CPU execution**.
   - **No OpenAI, Gemini, Hugging Face Inference, or external cloud AI APIs**.
   - All image preprocessing, specular highlight inpainting, color constancy, transformer inference, and explainability maps are computed locally.
   - Patient privacy is preserved: no images or clinical metadata are transmitted outside the device or saved to disk.

---

## Quick Start (How to Run)

The application is configured for single-click execution on Windows:

1. **Double-click [`run.bat`](file:///c:/AI_Healthcare_App/run.bat)** in the project root.
2. The launcher script will automatically:
   - Check port 8000 and stop any conflicting existing processes.
   - Activate the project virtual environment (`venv\Scripts\python.exe`).
   - Start the Uvicorn web server on `http://127.0.0.1:8000`.
   - Launch your default browser directly to [`http://127.0.0.1:8000/screening`](http://127.0.0.1:8000/screening).

To run manually via the terminal:
```bash
.\venv\Scripts\python.exe run.py
```

---

## Clinical Triage Framework

Predictions are categorized into 4 actionable triage tiers:

| Tier | Classification | Clinical Description | Actionable Recommendation | Visual Badge |
|---|---|---|---|---|
| **T0** | Normal | Clinically normal oral mucosa | Routine recall | Green (`#22c55e`) |
| **T1** | Variation from normal | Benign variations (e.g. leukoedema, Fordyce granules) | Short-interval review | Blue (`#3b82f6`) |
| **T2** | OPMD (incl. OSMF) | Oral Potentially Malignant Disorders (leukoplakia, erythroplakia, oral submucous fibrosis) | Refer to oral-medicine specialist | Orange (`#f97316`) |
| **T3** | Oral cancer | Clinically suspected Oral Squamous Cell Carcinoma | **URGENT referral** | Red (`#ef4444`) |

*Threshold Warning*: When the top predicted probability is below **0.60**, a prominent **Low Confidence Warning** banner is raised, advising specialist clinical review regardless of the predicted tier.

---

## Image Preprocessing Pipeline

Intraoral smartphone photos are standardized using a 5-step clinical preprocessing pipeline ([`app/preprocessing.py`](file:///c:/AI_Healthcare_App/app/preprocessing.py)) before reaching the model:

1. **Long-Side Rescaling**: Rescales the maximum dimension of the image to 1024 pixels while preserving native aspect ratio.
2. **Specular Glare Detection**: Detects moist intraoral flash reflections in HSV color space:
   $$V \ge 0.92 \quad \text{and} \quad S \le 0.18 \quad \text{and} \quad (V - \text{median}_{21}(V)) \ge 0.12$$
   Dilated with a $3 \times 3$ kernel (2 iterations) to encompass transition borders.
3. **Telea Inpainting**: Restores specular glare areas via Alexandru Telea's fast marching inpainting algorithm ($r = 5\text{ px}$).
4. **Shades-of-Gray Color Constancy**: Compensates for diverse smartphone camera white-balance and illumination tints using Minkowski $p=6$ norm:
   $$e = \left( \frac{1}{|V|} \sum_{x \in V} I(x)^6 \right)^{1/6}, \quad I_{\text{corrected}} = \frac{I}{\sqrt{3} e}$$
5. **Center Crop & Tensor Transformation**: Center crops to $1024 \times 1024$ (with symmetric zero-padding if non-square), resizes to $224 \times 224$ pixels, and normalizes pixel values with mean 0.5 and standard deviation 0.5.

---

## Explainable AI (XAI) Suite

Three complementary explainability methods provide full transparency into model behavior ([`app/explainability.py`](file:///c:/AI_Healthcare_App/app/explainability.py)):

1. **Grad-CAM (Gradient-Weighted Class Activation Mapping)**:
   - Computed at the final transformer block's LayerNorm (`model.blocks[-1].norm1`).
   - Gradients of the target class score with respect to patch feature activations are channel-averaged to compute importance weights.
   - Token activations ($14 \times 14$) are passed through ReLU, bilinearly upsampled to $224 \times 224$, and scaled to $[0, 1]$.
2. **Attention Rollout**:
   - Computes the recursive matrix product of attention weights across all 12 multi-head self-attention layers with residual skip connections:
     $$A_{\text{rollout}} = \prod_{l=1}^{12} \left( 0.5 \cdot A^{(l)} + 0.5 \cdot I \right)$$
   - Represents the attention distribution from the `[CLS]` token to all 196 patch tokens.
   - Raw border effects are preserved without artificial smoothing.
3. **LIME (Local Interpretable Model-agnostic Explanations)**:
   - On-demand superpixel perturbation analysis (300 samples, Quickshift segmentation, fixed seed 42).
   - Black background padding is excluded from segmentation and scoring.
   - Highlights the top-5 positive superpixel boundary outlines (gold) that exerted the greatest positive influence on the predicted class.

---

## In-Memory PDF Clinical Report

- Generates a clinical summary report via [ReportLab](https://www.reportlab.com/) ([`app/report.py`](file:///c:/AI_Healthcare_App/app/report.py)).
- **Zero Disk Persistence**: Generated purely in memory (`io.BytesIO`) and streamed directly as an HTTP download (`GET /api/report/{case_id}/pdf`).
- **Privacy Compliant**: Contains Case ID, timestamp, and anatomical site (labeled *"Recorded for report only – not used by model"*), with no patient names or identifiers.
- **Contents**: Predicted tier and action, class probability distribution ($< 0.01 \%$ and $> 99.99 \%$ formatting), preprocessed input, Grad-CAM overlay, Attention Rollout overlay, LIME overlay (if computed), one-line computational definitions, model version, and mandatory screening-aid disclaimer.

---

## Project Structure

```text
c:\AI_Healthcare_App\
├── app\
│   ├── explainability.py      # Grad-CAM, Attention Rollout, LIME, overlay rendering
│   ├── main.py                # FastAPI endpoints, exception handlers, PDF routes
│   ├── model.py               # OralCancerModel wrapper (predict, class names, colors)
│   ├── preprocessing.py       # Specular inpainting, Shades-of-Gray, resize
│   ├── report.py              # In-memory ReportLab PDF generator
│   ├── schemas.py             # Pydantic request & response schemas
│   ├── utils.py               # Case ID generator, image quality checks, base64 helpers
│   ├── static\
│   │   ├── css\style.css      # Dark-mode medical UI styling, modals, color legends
│   │   └── js\app.js          # Client-side utility functions
│   └── templates\
│       ├── about.html         # Project mission, clinical context, local processing
│       ├── index.html         # Portal home page
│       ├── model_training.html# Live model card & metrics from metrics.json
│       └── screening.html     # Interactive screening, XAI cards, modal lightbox
├── models\
│   └── best_4tier_vit.pth     # Locally trained ViT-B/16 checkpoint
├── training\
│   ├── metrics.json           # All official validation numbers & limitations
│   └── training_notebook.ipynb# Kaggle training notebook (kept local)
├── tests\
│   ├── test_screening_e2e.py       # Integration tests across test images
│   ├── test_training_page.py       # Metrics page and verification tests
│   └── test_phase4_review_fixes.py # Quality warnings, low-confidence, LIME cancel tests
├── crosscheck.py              # Exact probability parity check against original pipeline
├── config.yaml                # Model metadata, class labels, and quality thresholds
├── requirements.txt           # Exact pinned dependencies from virtual environment
├── run.bat                    # Windows batch launcher (stops old port 8000, starts app)
├── run.py                     # Python launcher script with browser auto-open
├── PROJECT_STATUS.md          # Project governance, rules, and change log
└── README.md                  # Comprehensive project documentation
```

---

## Limitations & Clinical Boundary Conditions

*(Copied verbatim from [`training/metrics.json`](file:///c:/AI_Healthcare_App/training/metrics.json)):*

1. "The same validation fold was used to pick the best epoch and to report results, so these figures are optimistic; no separate test set."
2. "Only 4 OPMD and 4 oral-cancer images are in the validation fold, so their recalls have very wide confidence intervals."
3. "All 12 OSMF images are in the training partition; OSMF itself is not represented in the validation fold."
4. "Oral-cancer images have no subject ID, so patient overlap for that class cannot be fully ruled out."
5. "Single dataset from one region (Tamil Nadu); not externally validated. Screening aid only, not a diagnosis."

---

## Dataset Citation & Reference

The model was trained on the open-access **SMART-OM** dataset:

- **Dataset**: SMART-OM: Smartphone intraoral images for oral potentially malignant disorders and oral cancer screening.
- **Authors**: A. R. Harisankar, et al.
- **Repository**: Figshare
- **DOI / Link**: [https://doi.org/10.6084/m9.figshare.31341790](https://doi.org/10.6084/m9.figshare.31341790)
- **Summary**: 2,469 smartphone intraoral images across 331 subjects (2,145 Normal, 179 Variation from normal, 125 OPMD, 20 Oral cancer).
