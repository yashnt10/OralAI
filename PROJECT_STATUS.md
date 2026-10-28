# OralScan AI – Project Status (read this first in every new chat)

AI in Healthcare course project. FastAPI web app that runs a locally trained ViT-B/16 model
(`models/best_4tier_vit.pth`) for 4-tier oral pre-cancer triage on smartphone intraoral photos,
with explainable AI (Grad-CAM, Attention Rollout, LIME).

## Mandatory rules
1. **No third-party AI APIs anywhere** (no OpenAI, Gemini, Hugging Face Inference, etc.).
   All predictions and explanations come from the local model. (Faculty rule: breaking it = zero marks.)
2. Do **NOT** modify `app/preprocessing.py`, `app/model.py`, or anything in `training/`. They are verified.
   Cross-check after any change: `SMITA00304_R_LB.jpeg` → T2 OPMD 89.27 %, `Ca 9.png` → T3 > 99.99 %,
   `SMITA00300_R_DT.jpeg` → T0 > 99.99 %.
3. Run every command yourself in the terminal. The user does not use PowerShell. Use the existing `venv`.
4. Never put test images, patient data or the model weights in public files or git.
5. One phase at a time. Wait for the user's approval before starting the next.
6. Wait for scripts, tests and LIME runs to finish before ending your turn, and report the result.
7. Do not write interpretations of what the model "looked at". Describe only what each XAI method computes.
8. Never claim clinical validation, certification or institutional approval. It is a research prototype / screening aid.
9. All numbers on the Model & Training page come from `training/metrics.json`. Never type numbers by hand.

## Phases
| Phase | Content | Status |
|---|---|---|
| 1 | Project setup, venv, config.yaml | Done |
| 2 | Backend: preprocessing, model loading, /api/health, /api/model-info, /api/predict, quality checks | Done |
| 3 | XAI: Grad-CAM (blocks[-1].norm1), Attention Rollout, LIME (on demand, top-5 outlines, padding excluded, seed 42) | Done, approved |
| – | Model & Training page (/model-training) from training/metrics.json | Done |
| 4 | Frontend: /screening, /model-training, /about | Done |
| 5 | PDF report, error handling, tests, README, .gitignore, run.bat | Done |
| 6 | Deployment | Only if the user asks |

## Key facts
- Model: timm `vit_base_patch16_224`, 4 classes, input 224×224, normalise mean/std 0.5.
- Preprocessing: long side 1024 → HSV specular mask → Telea inpaint r=5 → Shades-of-Gray p=6 → CenterCrop 1024 → Resize 224.
- Tiers: T0 Normal (routine recall), T1 Variation from normal (short-interval review),
  T2 OPMD incl. OSMF (refer to oral-medicine specialist), T3 Oral cancer (URGENT referral).
- Low-confidence threshold: 0.60.
- Validation results (343 images): macro-F1 71.22 %, balanced accuracy 72.01 %, accuracy 93.59 %
  (all-Normal baseline: accuracy 92.71 %, macro-F1 24.05 %).
- Server: http://127.0.0.1:8000 (/screening, /model-training, /about).

## Change log
(Agent: add one line per finished step: date – what changed – files.)
- 2026-10-26 – Phase 4 review fixes: renamed badge to 'Research prototype', updated About page heading to 'Privacy & Local Processing', expanded heatmaps to full card width with 0-1 colour legends and click-to-enlarge lightbox modal, formatted probabilities < 0.01 %, labeled anatomical site as report-only, configured run.bat & run.py with venv, port 8000 killer, and browser launch, added automated test suite for quality checks and low-confidence banner – app/templates/screening.html, app/templates/about.html, app/templates/model_training.html, app/templates/index.html, app/static/css/style.css, run.bat, run.py, crosscheck.py, tests/test_phase4_review_fixes.py, PROJECT_STATUS.md
- 2026-10-26 – Phase 5: added in-memory PDF report generation via ReportLab, comprehensive error handling (wrong file type, corrupt image, file >20MB, image <224x224, sanitized 500 handler), README.md, .gitignore, requirements.txt pinned versions, and test suite – app/report.py, app/main.py, app/templates/screening.html, README.md, .gitignore, requirements.txt, tests/test_phase5_features.py, PROJECT_STATUS.md
- 2026-10-28 – Submission & GitHub: created standalone verified submission_code.py pipeline, initialized Git repository adhering to strict privacy/weights rules, and pushed clean source code to GitHub (https://github.com/yashnt10/OralAI) – submission_code.py, PROJECT_STATUS.md

