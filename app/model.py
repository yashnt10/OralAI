"""
model.py
========
Loads the ViT-B/16 model ONCE at startup, exposes predict().
"""

import logging
from pathlib import Path

import numpy as np
import timm
import torch

from app.preprocessing import preprocess_for_model

logger = logging.getLogger(__name__)


class OralCancerModel:
    """Singleton-style wrapper around the ViT-B/16 checkpoint."""

    def __init__(self, config: dict) -> None:
        model_cfg = config["model"]
        weights_path = Path(model_cfg["path"])

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at '{weights_path}'. "
                "Download best_4tier_vit.pth and place it in models/."
            )

        logger.info("Loading model from %s …", weights_path)
        self.model = timm.create_model(
            model_cfg["architecture"],
            pretrained=model_cfg.get("pretrained", False),
            num_classes=model_cfg["num_classes"],
        )
        state = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()
        # CPU only as specified
        self.device = "cpu"
        self.model = self.model.to(self.device)

        # Build class / action lookup from config
        classes_cfg = config["classes"]
        self.class_names = [c["name"] for c in classes_cfg]
        self.actions = [c["action"] for c in classes_cfg]
        self.colors = [c["color"] for c in classes_cfg]

        self.low_confidence_threshold = config["thresholds"]["low_confidence"]
        logger.info("Model loaded successfully. Classes: %s", self.class_names)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, img_bgr: np.ndarray) -> dict:
        """
        Run full prediction pipeline on a raw BGR image.

        Returns
        -------
        dict with keys:
            tier_index   : int
            tier_name    : str
            action       : str
            color        : str  (hex)
            probabilities: list[float]  (4 values, sum to 1)
            low_confidence: bool
        """
        tensor, cleaned_rgb, glare_mask = preprocess_for_model(img_bgr)
        tensor = tensor.to(self.device)

        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        tier_index = int(probs.argmax())
        top_prob = float(probs[tier_index])

        return {
            "tier_index": tier_index,
            "tier_name": self.class_names[tier_index],
            "action": self.actions[tier_index],
            "color": self.colors[tier_index],
            "probabilities": probs.tolist(),
            "low_confidence": top_prob < self.low_confidence_threshold,
            "_cleaned_rgb": cleaned_rgb,   # internal — used by /api/predict
            "_glare_mask": glare_mask,     # internal — used by quality check
        }
