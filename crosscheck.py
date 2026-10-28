"""
crosscheck.py
=============
Runs predictions on all images in test_images/ using BOTH:
  A) The new app pipeline  (app.model + app.preprocessing)
  B) The original osmf_predict_app.py pipeline

Prints a side-by-side comparison table and flags any probability
differences > 0.1 % (floating-point noise only — should match exactly).

Usage (from project root with venv active):
    python crosscheck.py
"""

import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path

import cv2
import numpy as np
import torch
import timm
from torchvision import transforms

# ── Make sure app package is importable ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from app.preprocessing import clinical_preprocessing, TF  # noqa: E402

# ── Original osmf_predict_app.py logic (verbatim) ────────────────────────────

TIERS = ['T0 Normal', 'T1 Variation from normal', 'T2 OPMD (incl. OSMF)', 'T3 Oral cancer']
ACTIONS = ['Routine recall', 'Short-interval review', 'Refer to oral-medicine specialist', 'URGENT referral']

ORIG_TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(1024),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


def orig_clinical_preprocessing(img_bgr):
    """Verbatim copy from osmf_predict_app.py."""
    h, w = img_bgr.shape[:2]; scale = 1024 / max(h, w)
    img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v, s = hsv[:, :, 2] / 255.0, hsv[:, :, 1] / 255.0
    med = cv2.medianBlur(hsv[:, :, 2], 21) / 255.0
    mask = ((v >= 0.92) & (s <= 0.18) & ((v - med) >= 0.12)).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    inp = cv2.inpaint(img_bgr, mask, 5, cv2.INPAINT_TELEA)
    rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB); n = rgb / 255.0
    valid = ((n >= 0.02) & (n <= 0.98)).all(axis=2) & (mask == 0)
    if valid.sum() > 100:
        e = np.power(np.mean(np.power(n[valid], 6), axis=0), 1 / 6); e = e / np.linalg.norm(e)
        rgb = np.clip(n / (np.sqrt(3) * e) * 255.0, 0, 255).astype(np.uint8)
    return rgb


@torch.no_grad()
def orig_predict(model, img_rgb):
    rgb = orig_clinical_preprocessing(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    p = torch.softmax(model(ORIG_TF(rgb)[None]), 1)[0].cpu().numpy()
    k = int(p.argmax())
    return k, p


# ── New app pipeline ──────────────────────────────────────────────────────────

@torch.no_grad()
def new_predict(model, img_bgr):
    rgb = clinical_preprocessing(img_bgr)
    tensor = TF(rgb).unsqueeze(0)
    p = torch.softmax(model(tensor), 1)[0].cpu().numpy()
    k = int(p.argmax())
    return k, p


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    weights = Path("models/best_4tier_vit.pth")
    print(f"\nLoading model from {weights} …\n")

    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=4)
    model.load_state_dict(torch.load(weights, map_location='cpu'))
    model.eval()

    test_dir = Path("test_images")
    images = sorted(test_dir.glob("*"))
    images = [p for p in images if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}]

    if not images:
        print("No images found in test_images/. Aborting.")
        return

    col_w = 28

    for img_path in images:
        print("=" * 90)
        print(f"  IMAGE: {img_path.name}")
        print("=" * 90)

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  ERROR: could not read {img_path.name}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Run both pipelines
        orig_k, orig_p = orig_predict(model, img_rgb)
        new_k, new_p   = new_predict(model, img_bgr)

        # Header
        print(f"  {'Class':<{col_w}}  {'osmf_predict_app.py':>12}  {'New App':>12}  {'Δ (pp)':>10}")
        print(f"  {'-'*col_w}  {'-'*12}  {'-'*12}  {'-'*10}")

        all_match = True
        for i, tier in enumerate(TIERS):
            delta = abs(orig_p[i] - new_p[i]) * 100
            flag = " ⚠️  MISMATCH" if delta > 0.1 else ""
            if delta > 0.1:
                all_match = False
            print(f"  {tier:<{col_w}}  {orig_p[i]*100:>11.2f}%  {new_p[i]*100:>11.2f}%  {delta:>9.4f}{flag}")

        print()
        print(f"  osmf_predict_app  → {TIERS[orig_k]}  ({orig_p[orig_k]*100:.2f}%)  → {ACTIONS[orig_k]}")
        print(f"  New App           → {TIERS[new_k]}  ({new_p[new_k]*100:.2f}%)  → {ACTIONS[new_k]}")
        match_str = "✅  EXACT MATCH" if all_match else "❌  PROBABILITIES DIFFER"
        print(f"\n  Result: {match_str}")
        print()


if __name__ == "__main__":
    main()
