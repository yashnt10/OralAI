"""
generate_test_heatmaps.py
=========================
Generates Grad-CAM and Attention Rollout heatmaps for every image in
test_images/, saves side-by-side PNG figures to test_outputs/.

Heatmaps are on the EXACT 224x224 image the model sees (after clinical
preprocessing + CenterCrop(1024) + Resize(224,224)).  No adjustments are
made -- if the model looks at teeth, fingers, or instruments, that is
shown honestly.

Usage (from project root with venv active):
    python generate_test_heatmaps.py
"""

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import timm

sys.path.insert(0, str(Path(__file__).parent))

from app.preprocessing import clinical_preprocessing, TF
from app.explainability import (
    AttentionRollout,
    compute_gradcam,
    get_224_image,
    overlay_heatmap,
)

TIERS = [
    "T0 Normal",
    "T1 Variation from normal",
    "T2 OPMD (incl. OSMF)",
    "T3 Oral cancer",
]
ACTIONS = [
    "Routine recall",
    "Short-interval review",
    "Refer to oral-medicine specialist",
    "URGENT referral",
]
COLORS = ["#22c55e", "#eab308", "#f97316", "#ef4444"]


def main():
    out_dir = Path("test_outputs")
    out_dir.mkdir(exist_ok=True)

    print("\nLoading model from models/best_4tier_vit.pth ...")
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=4)
    model.load_state_dict(torch.load("models/best_4tier_vit.pth", map_location="cpu"))
    model.eval()
    print("Model loaded.\n")

    rollout = AttentionRollout(model)

    test_dir = Path("test_images")
    image_paths = sorted(
        [p for p in test_dir.glob("*")
         if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    )

    if not image_paths:
        print("No images found in test_images/")
        return

    all_outputs = []

    for img_path in image_paths:
        print(f"{'='*60}")
        print(f"  {img_path.name}")
        print(f"{'='*60}")

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  ERROR: could not read {img_path.name}")
            continue

        # ── Preprocess ─────────────────────────────────────────────────────
        cleaned_rgb = clinical_preprocessing(img_bgr)
        tensor_224  = TF(cleaned_rgb).unsqueeze(0)   # (1, 3, 224, 224)
        img_224     = get_224_image(cleaned_rgb)      # uint8 RGB 224x224
        img_224_f   = img_224.astype(np.float32) / 255.0

        # ── Predict ─────────────────────────────────────────────────────────
        with torch.no_grad():
            probs = torch.softmax(model(tensor_224), dim=1)[0].numpy()
        tier_idx = int(probs.argmax())

        print(f"  Tier  : {TIERS[tier_idx]}  ({probs[tier_idx]*100:.2f}%)")
        print(f"  Action: {ACTIONS[tier_idx]}")
        for i, (t, p) in enumerate(zip(TIERS, probs)):
            bar = "#" * int(p * 40)
            marker = " <-- predicted" if i == tier_idx else ""
            print(f"    {t:<30s} {p*100:6.2f}%  {bar}{marker}")
        print()

        # ── Grad-CAM ────────────────────────────────────────────────────────
        print("  Running Grad-CAM ...")
        gradcam_map = compute_gradcam(model, tensor_224, tier_idx)
        gradcam_overlay = overlay_heatmap(img_224, gradcam_map, alpha=0.50)

        # ── Attention Rollout ────────────────────────────────────────────────
        print("  Running Attention Rollout ...")
        attn_map = rollout(tensor_224)
        # Resize 14x14 -> 224x224 for display (nearest for exact pixel boundary)
        attn_map_224 = cv2.resize(attn_map, (224, 224), interpolation=cv2.INTER_LINEAR)
        attn_overlay = overlay_heatmap(img_224, attn_map_224, alpha=0.50)

        # ── Save figure ─────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

        # -- Panel 1: clean 224x224 input
        axes[0].imshow(img_224)
        axes[0].set_title(
            f"Preprocessed input (224x224)\n{img_path.name}",
            fontsize=10, fontweight="bold"
        )
        axes[0].axis("off")

        # -- Panel 2: Grad-CAM
        im1 = axes[1].imshow(gradcam_overlay)
        axes[1].set_title(
            f"Grad-CAM\ntarget layer: blocks[-1].norm1",
            fontsize=10
        )
        axes[1].axis("off")
        mappable1 = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
        mappable1.set_array(gradcam_map)
        fig.colorbar(mappable1, ax=axes[1], fraction=0.046, pad=0.04,
                     label="Importance (0-1)")

        # -- Panel 3: Attention Rollout
        axes[2].imshow(attn_overlay)
        axes[2].set_title(
            f"Attention Rollout\nCLS->patch, all 12 blocks",
            fontsize=10
        )
        axes[2].axis("off")
        mappable2 = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
        mappable2.set_array(attn_map)
        fig.colorbar(mappable2, ax=axes[2], fraction=0.046, pad=0.04,
                     label="Attention (0-1)")

        # -- Super-title with prediction
        fig.suptitle(
            f"Prediction: {TIERS[tier_idx]}  ({probs[tier_idx]*100:.2f}%)  |  "
            f"Action: {ACTIONS[tier_idx]}",
            fontsize=12, fontweight="bold", color=COLORS[tier_idx],
        )

        # -- Disclaimer footer
        fig.text(
            0.5, 0.01,
            "Screening aid only - not a medical diagnosis. "
            "Heatmaps show where the model looked, not proof of disease.",
            ha="center", fontsize=8, color="gray", style="italic",
        )

        plt.tight_layout(rect=[0, 0.04, 1, 0.94])

        out_path = out_dir / f"{img_path.stem}_xai.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        all_outputs.append(out_path)
        print(f"  Saved: {out_path}\n")

    print(f"Done. {len(all_outputs)} figure(s) saved to test_outputs/")
    for p in all_outputs:
        print(f"  {p}")


if __name__ == "__main__":
    main()
