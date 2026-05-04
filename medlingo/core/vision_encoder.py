"""
medlingo/core/vision_encoder.py

Frozen BioMedCLIP vision encoder.
Produces 768-dimensional feature vectors fed into both projection pathways.

Reference: §III-A — "We utilize a frozen BioMedCLIP vision encoder to
generate high-dimensional feature maps from raw medical imagery."
"""

from __future__ import annotations

import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Union, Optional

logger = logging.getLogger(__name__)

_BIOMEDCLIP_HF_ID = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
_FEATURE_DIM = 768


class VisionEncoder(nn.Module):
    """
    Wrapper around the frozen BioMedCLIP ViT-B/16 encoder.

    The encoder is loaded in FP16 and kept strictly frozen throughout all
    training stages. Only the linear projection bridges (PathwayA / PathwayB)
    are updated during Stage 1.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to locally saved BioMedCLIP weights. Falls back to HF Hub.
    device : str
        Target device (e.g. "cuda", "cpu").
    """

    output_dim: int = _FEATURE_DIM

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.device = device
        self._model_id = str(model_path) if model_path else _BIOMEDCLIP_HF_ID
        self._encoder, self._processor = self._load()
        self._freeze()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract CLS-token visual features.

        Parameters
        ----------
        pixel_values : torch.Tensor
            Shape (B, 3, H, W) in the range [0, 1].

        Returns
        -------
        torch.Tensor
            Shape (B, 768) — batch of visual feature vectors.
        """
        pixel_values = pixel_values.to(self.device, dtype=torch.float16)
        outputs = self._encoder(pixel_values=pixel_values)
        # BioMedCLIP uses a CLIP-style ViT; grab pooler_output (CLS token)
        features = outputs.pooler_output          # (B, 768)
        return features.to(torch.float16)

    def preprocess(self, images) -> torch.Tensor:
        """Run the BioMedCLIP processor and return pixel_values."""
        inputs = self._processor(images=images, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self):
        try:
            from transformers import CLIPVisionModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers >= 4.40 required. "
                "Run: pip install transformers>=4.40"
            ) from exc

        logger.info("Loading BioMedCLIP vision encoder from: %s", self._model_id)

        encoder = CLIPVisionModel.from_pretrained(
            self._model_id,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        processor = CLIPProcessor.from_pretrained(self._model_id)
        return encoder, processor

    def _freeze(self) -> None:
        """Freeze all encoder parameters — no gradients ever flow here."""
        for param in self._encoder.parameters():
            param.requires_grad_(False)
        self._encoder.eval()
        logger.debug(
            "BioMedCLIP frozen: %d parameters, all require_grad=False",
            sum(p.numel() for p in self._encoder.parameters()),
        )

    @property
    def processor(self):
        return self._processor
