"""
explainability.py
=================
Grad-CAM, Attention Rollout, and LIME for the ViT-B/16 model.

All heatmaps are overlaid on the EXACT 224×224 image the model sees
(after clinical_preprocessing + CenterCrop(1024) + Resize(224,224)).

No cosmetic adjustments are made and no subjective focus claims are imposed.
Border effects in attention rollout are preserved without smoothing.
Black padding is excluded from LIME segmentation and scoring.
"""

import io
import base64
import logging

import cv2
import matplotlib
matplotlib.use("Agg")                        # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm_module
import numpy as np
import torch
from torchvision import transforms

logger = logging.getLogger(__name__)

# ── Geometric-only transform (no normalisation) ───────────────────────────────
# Gives the exact 224×224 pixel image the model sees, ready for overlaying.
_GEOM_TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(1024),
    transforms.Resize((224, 224)),
])

_NORMALIZE = transforms.Normalize([0.5] * 3, [0.5] * 3)


def format_confidence(prob: float) -> str:
    """Format confidence: > 99.99% if strictly above 99.99%, < 0.01% if strictly below 0.01%, else two decimals."""
    pct = prob * 100.0
    if pct > 99.99:
        return "> 99.99%"
    if pct < 0.01:
        return "< 0.01%"
    return f"{pct:.2f}%"


def get_224_image(cleaned_rgb: np.ndarray) -> np.ndarray:
    """
    Return the exact 224×224 uint8 RGB image fed to the model
    (CenterCrop 1024 → Resize 224, no normalisation).
    """
    return np.array(_GEOM_TF(cleaned_rgb))


# ─────────────────────────────────────────────────────────────────────────────
#  Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────

def reshape_transform(tensor, height: int = 14, width: int = 14):
    """
    Convert ViT feature maps (B, N_tokens, D) → (B, D, H, W).
    Index 0 is the CLS token — drop it, then reshape 196 patch tokens → 14×14.
    """
    result = tensor[:, 1:, :]                                    # (B, 196, D)
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)              # (B, D, H, W)
    return result


def compute_gradcam(model, tensor_224: torch.Tensor, target_class_idx: int) -> np.ndarray:
    """
    Grad-CAM on the last ViT block's LayerNorm (model.blocks[-1].norm1).

    Parameters
    ----------
    model            : ViT model in eval mode
    tensor_224       : (1, 3, 224, 224) normalised tensor
    target_class_idx : int — class to explain

    Returns
    -------
    heatmap : np.ndarray float32 (224, 224) in [0, 1]
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    target_layers = [model.blocks[-1].norm1]

    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform,
    )

    targets = [ClassifierOutputTarget(target_class_idx)]
    grayscale_cam = cam(input_tensor=tensor_224, targets=targets)  # (1, 224, 224)
    return grayscale_cam[0].astype(np.float32)                      # (224, 224)


# ─────────────────────────────────────────────────────────────────────────────
#  Attention Rollout
# ─────────────────────────────────────────────────────────────────────────────

class AttentionRollout:
    """
    Attention Rollout across all 12 ViT transformer blocks.

    Method
    ------
    1. Temporarily monkey-patch each block's attention forward() to capture
       the raw attention weight matrix (head-averaged) without running the
       model twice.
    2. Add identity to each layer's attention matrix (skip connections: 0.5*A + 0.5*I).
    3. Normalise rows to sum to 1.
    4. Multiply the 12 matrices in sequence.
    5. Take the CLS-token row; reshape the 196 patch tokens → 14×14.

    Works regardless of whether fused_attn is True or False.
    Does NOT smooth or hide border effects.
    """

    def __init__(self, model):
        self.model = model
        self._captured: list[np.ndarray] = []
        self._patches: list[tuple] = []

    # -- internal ----------------------------------------------------------------

    def _patch(self):
        """Replace each attention module's forward() to capture attn weights."""
        self._captured = []
        self._patches = []

        for block in self.model.blocks:
            attn = block.attn
            orig = attn.forward
            self._patches.append((attn, orig))

            # Create a closure over `attn` so each block gets its own copy
            def _make_forward(a):
                def _forward(x):
                    B, N, C = x.shape
                    qkv = a.qkv(x).reshape(
                        B, N, 3, a.num_heads, a.head_dim
                    ).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)

                    # Compute attention weights explicitly (bypasses fused_attn)
                    attn_w = (q * a.scale) @ k.transpose(-2, -1)   # (B, H, N, N)
                    attn_w = attn_w.softmax(dim=-1)

                    # Save head-averaged weights for rollout
                    self._captured.append(
                        attn_w.detach().mean(dim=1).cpu().numpy()  # (B, N, N)
                    )

                    # Standard attention output
                    attn_drop = a.attn_drop(attn_w)
                    out = (attn_drop @ v).transpose(1, 2).reshape(B, N, C)
                    out = a.proj(out)
                    out = a.proj_drop(out)
                    return out
                return _forward

            attn.forward = _make_forward(attn)

    def _unpatch(self):
        """Restore original forward() methods."""
        for attn, orig in self._patches:
            attn.forward = orig
        self._patches = []

    # -- public -----------------------------------------------------------------

    @torch.no_grad()
    def __call__(self, tensor_224: torch.Tensor) -> np.ndarray:
        """
        Parameters
        ----------
        tensor_224 : (1, 3, 224, 224) normalised tensor

        Returns
        -------
        heatmap : np.ndarray float32 (14, 14) in [0, 1]
        """
        self._patch()
        try:
            _ = self.model(tensor_224)
        finally:
            self._unpatch()

        # self._captured: 12 arrays each (1, 197, 197)
        rollout = np.eye(197, dtype=np.float32)
        for attn in self._captured:
            a = attn[0]                              # (197, 197)
            a = 0.5 * a + 0.5 * np.eye(197)         # add residual skip
            a /= a.sum(axis=-1, keepdims=True)       # normalise rows
            rollout = a @ rollout

        # CLS token's attention to all patch tokens
        mask = rollout[0, 1:]                        # (196,)
        mask = mask.reshape(14, 14)
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        return mask.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  LIME
# ─────────────────────────────────────────────────────────────────────────────

def compute_lime(
    model,
    img_224_rgb: np.ndarray,
    target_class_idx: int,
    num_samples: int = 300,
    random_state: int = 42,
) -> np.ndarray:
    """
    LIME superpixel explanation on the 224×224 preprocessed image.
    Excludes black padding from segmentation and scoring.
    Returns the image with the top 5 positive superpixels highlighted as outlines.
    Uses fixed random seed for reproducible sampling.

    Parameters
    ----------
    model            : ViT model in eval mode
    img_224_rgb      : np.ndarray uint8 (224, 224, 3)
    target_class_idx : class to explain
    num_samples      : number of LIME perturbation samples (300)
    random_state     : fixed seed (default 42)

    Returns
    -------
    overlay : np.ndarray uint8 (224, 224, 3) image with top 5 superpixel outlines
    """
    from lime import lime_image
    from skimage.segmentation import quickshift, find_boundaries

    def predict_fn(images: np.ndarray) -> np.ndarray:
        """Takes (N, 224, 224, 3) uint8, returns (N, 4) float probabilities."""
        batch = []
        for img in images:
            t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            t = _NORMALIZE(t)
            batch.append(t)
        tensor = torch.stack(batch)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def custom_segmentation(image: np.ndarray) -> np.ndarray:
        """Quickshift segmentation that maps black padding to segment 0."""
        segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)
        segments = segments + 1
        pad_mask = (image <= 10).all(axis=-1)
        segments[pad_mask] = 0
        return segments

    explainer = lime_image.LimeImageExplainer(random_state=random_state)
    explanation = explainer.explain_instance(
        img_224_rgb,
        predict_fn,
        top_labels=4,
        hide_color=0,
        num_samples=num_samples,
        batch_size=32,
        segmentation_fn=custom_segmentation,
        random_seed=random_state,
    )

    segments = explanation.segments
    pad_mask = (img_224_rgb <= 10).all(axis=-1)

    # Exclude segment 0 (padding) and any segment overlapping >30% with black padding
    pos_segments = []
    for seg_id, weight in explanation.local_exp.get(target_class_idx, []):
        if weight <= 0 or seg_id == 0:
            continue
        seg_pixels = (segments == seg_id)
        if seg_pixels.sum() == 0:
            continue
        if (pad_mask & seg_pixels).mean() > 0.30:
            continue
        pos_segments.append((seg_id, weight))

    # Keep strictly top 5 positive superpixels
    pos_segments.sort(key=lambda x: x[1], reverse=True)
    top5_ids = [seg_id for seg_id, _ in pos_segments[:5]]

    # Generate crisp boundary outlines
    top5_mask = np.isin(segments, top5_ids).astype(np.uint8)
    boundaries = find_boundaries(top5_mask, mode="thick")

    overlay = img_224_rgb.copy()
    # Distinct gold/yellow outlines (255, 215, 0)
    overlay[boundaries] = [255, 215, 0]
    return overlay


# ─────────────────────────────────────────────────────────────────────────────
#  Overlay + figure helpers
# ─────────────────────────────────────────────────────────────────────────────

def overlay_heatmap(
    img_224_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.50,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Blend a grayscale heatmap over the 224×224 image.

    Parameters
    ----------
    img_224_rgb : uint8 (224, 224, 3) RGB
    heatmap     : float32 (H, W) in [0, 1] — any size, will be resized with nearest interpolation
    alpha       : opacity of heatmap overlay
    colormap    : matplotlib colormap name

    Returns
    -------
    overlay : uint8 (224, 224, 3) RGB
    """
    h, w = img_224_rgb.shape[:2]
    if heatmap.shape != (h, w):
        # Use nearest neighbor to avoid artificial smoothing/hiding of border effects
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_NEAREST)

    cmap = plt.colormaps[colormap] if hasattr(plt, "colormaps") else cm_module.get_cmap(colormap)
    colored = cmap(heatmap)[:, :, :3]                  # (H, W, 3) float [0,1]

    img_f = img_224_rgb.astype(np.float32) / 255.0
    blended = np.clip((1 - alpha) * img_f + alpha * colored, 0, 1)
    return (blended * 255).astype(np.uint8)


def make_explanation_figure_b64(
    img_224_rgb: np.ndarray,
    explanation_result: np.ndarray,
    method_label: str,
    tier_name: str,
    prob: float,
    alpha: float = 0.50,
    colormap: str = "jet",
) -> tuple[str, str]:
    """
    Render the explanation overlay alone as a base64 PNG data-URI,
    and return (base64_data_uri, explanation_text).
    """
    if method_label == "LIME":
        # explanation_result is already the outline overlay
        overlay = explanation_result
    else:
        overlay = overlay_heatmap(img_224_rgb, explanation_result, alpha=alpha, colormap=colormap)

    # Convert RGB overlay directly to PNG base64
    from app.utils import ndarray_to_base64
    data_uri = ndarray_to_base64(overlay, fmt=".png")

    # Method descriptions (strictly computational definitions — no subjective focus claims)
    explanations = {
        "Grad-CAM": (
            "Grad-CAM: the target-class gradients are averaged per feature channel to give channel weights; "
            "the weighted sum of the final block's token activations (blocks[-1].norm1, 196 tokens → 14×14) "
            "is passed through ReLU, upsampled to 224×224 and scaled to 0–1."
        ),
        "Attention Rollout": (
            "Recursive matrix product of attention weights across all 12 transformer layers "
            "with residual skip connections (0.5·A + 0.5·I), representing the CLS-token attention distribution."
        ),
        "LIME": (
            "Top 5 positive superpixels (gold outlines) influencing the prediction, fitted with a "
            "local linear surrogate model on 300 perturbed samples with black padding excluded (fixed random seed)."
        ),
    }
    text = explanations.get(method_label, "Heatmap overlay.")
    return data_uri, text
