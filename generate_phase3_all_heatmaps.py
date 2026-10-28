"""
generate_phase3_all_heatmaps.py
===============================
Generates 4-panel figures for all 3 test images:
1. Preprocessed 224x224 input
2. Grad-CAM (gradient-weighted activations)
3. Attention Rollout (all 12 layers, unsmoothed raw patch boundaries)
4. LIME (top 5 positive superpixels as outlines, black padding excluded, seed 42)

Footer: "Highlighted regions influenced this prediction according to each method. Not proof of disease. Screening aid only."
Confidence: "> 99.99%" when > 99.99%.
"""

import sys
import time
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
    compute_lime,
    format_confidence,
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

    print("\n" + "=" * 70)
    print("Loading ViT-B/16 model from models/best_4tier_vit.pth on CPU...")
    print("=" * 70)
    t0_load = time.perf_counter()
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=4)
    model.load_state_dict(torch.load("models/best_4tier_vit.pth", map_location="cpu"))
    model.eval()
    t_load = time.perf_counter() - t0_load
    print(f"Model loaded in {t_load:.2f} s on CPU.\n")

    rollout = AttentionRollout(model)

    test_dir = Path("test_images")
    image_paths = sorted(
        [p for p in test_dir.glob("*")
         if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    )

    if not image_paths:
        print("No test images found in test_images/!")
        return

    timings = {}
    saved_paths = []

    for img_path in image_paths:
        print("\n" + "#" * 70)
        print(f"  IMAGE: {img_path.name}")
        print("#" * 70)

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  ERROR: Could not read {img_path.name}")
            continue

        # ── Preprocessing ──────────────────────────────────────────────────
        cleaned_rgb = clinical_preprocessing(img_bgr)
        tensor_224 = TF(cleaned_rgb).unsqueeze(0)   # (1, 3, 224, 224)
        img_224 = get_224_image(cleaned_rgb)        # uint8 RGB 224x224

        # ── Inference ──────────────────────────────────────────────────────
        with torch.no_grad():
            logits = model(tensor_224)
            probs = torch.softmax(logits, dim=1)[0].numpy()
        tier_idx = int(probs.argmax())
        top_prob = float(probs[tier_idx])
        conf_str = format_confidence(top_prob)

        print(f"  Prediction : {TIERS[tier_idx]} ({conf_str})")
        print(f"  Action     : {ACTIONS[tier_idx]}")
        for i, (t_name, p_val) in enumerate(zip(TIERS, probs)):
            marker = "  <-- PREDICTED" if i == tier_idx else ""
            c_str = format_confidence(float(p_val))
            print(f"    {t_name:<28s} {c_str:>9s}{marker}")
        print()

        # ── 1. Grad-CAM ────────────────────────────────────────────────────
        print("  Computing Grad-CAM (gradient-weighted activations)...")
        t0_gc = time.perf_counter()
        gradcam_map = compute_gradcam(model, tensor_224, tier_idx)
        t_gc = time.perf_counter() - t0_gc
        print(f"    Grad-CAM CPU time: {t_gc:.3f} s")
        gradcam_overlay = overlay_heatmap(img_224, gradcam_map, alpha=0.50)

        # ── 2. Attention Rollout ───────────────────────────────────────────
        print("  Computing Attention Rollout (unsmoothed patch boundaries)...")
        t0_ar = time.perf_counter()
        attn_map_14 = rollout(tensor_224)
        # Use INTER_NEAREST to strictly preserve raw patch values & border effects without smoothing
        attn_map = cv2.resize(attn_map_14, (224, 224), interpolation=cv2.INTER_NEAREST)
        t_ar = time.perf_counter() - t0_ar
        print(f"    Attention Rollout CPU time: {t_ar:.3f} s")
        attn_overlay = overlay_heatmap(img_224, attn_map, alpha=0.50)

        # ── 3. LIME ────────────────────────────────────────────────────────
        print("  Computing LIME (top 5 superpixel outlines, padding excluded, seed 42)...")
        t0_lime = time.perf_counter()
        lime_overlay = compute_lime(model, img_224, tier_idx, num_samples=300, random_state=42)
        t_lime = time.perf_counter() - t0_lime
        print(f"    LIME CPU time: {t_lime:.3f} s")

        timings[img_path.name] = {
            "prediction": f"{TIERS[tier_idx]} ({conf_str})",
            "gradcam_s": t_gc,
            "rollout_s": t_ar,
            "lime_s": t_lime,
        }

        # ── Save 4-panel figure ────────────────────────────────────────────
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))

        # Panel 1: Preprocessed Input
        axes[0].imshow(img_224)
        axes[0].set_title(f"Input (224x224)\n{img_path.name}", fontsize=10, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: Grad-CAM
        axes[1].imshow(gradcam_overlay)
        axes[1].set_title(f"Grad-CAM (gradient-weighted activations)\n{t_gc:.2f} s | blocks[-1].norm1", fontsize=9)
        axes[1].axis("off")
        m1 = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
        m1.set_array(gradcam_map)
        fig.colorbar(m1, ax=axes[1], fraction=0.046, pad=0.04, label="Importance (0-1)")

        # Panel 3: Attention Rollout
        axes[2].imshow(attn_overlay)
        axes[2].set_title(f"Attention Rollout (raw border effects)\n{t_ar:.2f} s | 12 layers", fontsize=9)
        axes[2].axis("off")
        m2 = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
        m2.set_array(attn_map)
        fig.colorbar(m2, ax=axes[2], fraction=0.046, pad=0.04, label="Attention (0-1)")

        # Panel 4: LIME
        axes[3].imshow(lime_overlay)
        axes[3].set_title(f"LIME (Top 5 Positive Superpixels)\n{t_lime:.1f} s | seed=42, no pad", fontsize=9)
        axes[3].axis("off")

        # Super-title with formatted confidence
        fig.suptitle(
            f"Prediction: {TIERS[tier_idx]} ({conf_str})  |  Action: {ACTIONS[tier_idx]}",
            fontsize=12,
            fontweight="bold",
            color=COLORS[tier_idx],
        )

        # Exact required footer
        fig.text(
            0.5, 0.01,
            "Highlighted regions influenced this prediction according to each method. Not proof of disease. Screening aid only.",
            ha="center", fontsize=8, color="gray", style="italic",
        )

        plt.tight_layout(rect=[0, 0.04, 1, 0.94])
        out_path = out_dir / f"{img_path.stem}_phase3_xai.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(str(out_path.resolve()))
        print(f"  Saved: {out_path}")

    # Summary table of timings
    print("\n" + "=" * 70)
    print("PHASE 3 CPU TIMINGS SUMMARY")
    print("=" * 70)
    print(f"{'Image':<24} {'Prediction':<32} {'Grad-CAM':>10} {'Rollout':>10} {'LIME':>10}")
    print("-" * 88)
    for name, t in timings.items():
        print(f"{name:<24} {t['prediction']:<32} {t['gradcam_s']:>9.3f}s {t['rollout_s']:>9.3f}s {t['lime_s']:>9.1f}s")
    print("=" * 70 + "\n")
    print("SAVED 4-PANEL FIGURES:")
    for sp in saved_paths:
        print("  " + sp)


if __name__ == "__main__":
    main()
