"""
preprocessing.py
================
Exact copy of the training-time preprocessing from osmf_predict_app.py.
Do NOT change any step, constant, or order — the model was trained with this.

Pipeline (in order):
  1. Resize long side to 1024 (preserve aspect ratio)
  2. Glare mask: HSV with V >= 0.92, S <= 0.18, V - medianBlur21(V) >= 0.12
  3. Dilate mask 3x3 kernel, 2 iterations
  4. cv2.inpaint radius=5, TELEA
  5. Shades-of-Gray (p=6) on valid pixels (0.02–0.98, not masked, >100 pixels)
  6. CenterCrop(1024) -> Resize(224,224) -> ToTensor -> Normalize(mean=0.5, std=0.5)
"""

import cv2
import numpy as np
import torch
from torchvision import transforms


# ── Transform applied AFTER clinical_preprocessing ──────────────────────────
TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(1024),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


def clinical_preprocessing(img_bgr: np.ndarray) -> np.ndarray:
    """
    Identical to the training script: specular mask -> Telea inpainting ->
    Shades-of-Gray (p=6).

    Parameters
    ----------
    img_bgr : np.ndarray
        Input image in BGR format (as returned by cv2.imread / cv2.imdecode).

    Returns
    -------
    rgb : np.ndarray
        Preprocessed image in RGB format, uint8, ready for TF().
    """
    # ── 1. Resize long side to 1024 ─────────────────────────────────────────
    h, w = img_bgr.shape[:2]
    scale = 1024 / max(h, w)
    img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

    # ── 2. Glare mask ────────────────────────────────────────────────────────
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2] / 255.0
    s = hsv[:, :, 1] / 255.0
    med = cv2.medianBlur(hsv[:, :, 2], 21) / 255.0
    mask = (
        (v >= 0.92) & (s <= 0.18) & ((v - med) >= 0.12)
    ).astype(np.uint8) * 255

    # ── 3. Dilate mask ───────────────────────────────────────────────────────
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)

    # ── 4. Inpaint ───────────────────────────────────────────────────────────
    inp = cv2.inpaint(img_bgr, mask, 5, cv2.INPAINT_TELEA)

    # ── 5. Shades-of-Gray (p=6) ──────────────────────────────────────────────
    rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    n = rgb / 255.0
    valid = ((n >= 0.02) & (n <= 0.98)).all(axis=2) & (mask == 0)
    if valid.sum() > 100:
        e = np.power(np.mean(np.power(n[valid], 6), axis=0), 1 / 6)
        e = e / np.linalg.norm(e)
        rgb = np.clip(n / (np.sqrt(3) * e) * 255.0, 0, 255).astype(np.uint8)

    return rgb  # uint8 RGB


def preprocess_for_model(img_bgr: np.ndarray) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline: clinical_preprocessing + TF.

    Parameters
    ----------
    img_bgr : np.ndarray
        Raw BGR image.

    Returns
    -------
    tensor : torch.Tensor  shape (1, 3, 224, 224)
    cleaned_rgb : np.ndarray  uint8 RGB after glare removal + colour correction
    glare_mask : np.ndarray  uint8 binary mask (255 = glare pixel)
    """
    # Compute glare mask before preprocessing (on resized image) so callers
    # can use it for quality checks.
    h, w = img_bgr.shape[:2]
    scale = 1024 / max(h, w)
    resized_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    hsv = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2] / 255.0
    s = hsv[:, :, 1] / 255.0
    med = cv2.medianBlur(hsv[:, :, 2], 21) / 255.0
    glare_mask = (
        (v >= 0.92) & (s <= 0.18) & ((v - med) >= 0.12)
    ).astype(np.uint8) * 255
    glare_mask = cv2.dilate(glare_mask, np.ones((3, 3), np.uint8), iterations=2)

    cleaned_rgb = clinical_preprocessing(img_bgr)
    tensor = TF(cleaned_rgb).unsqueeze(0)  # (1, 3, 224, 224)
    return tensor, cleaned_rgb, glare_mask
