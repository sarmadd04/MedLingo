"""
medlingo/data/preprocessing.py

Data preprocessing utilities shared across training stages.

Handles image normalization, tokenization, and collation for the
multimodal inputs that flow through the BioMedCLIP encoder and
both projection pathways.
"""

from __future__ import annotations

import logging
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# BioMedCLIP normalization constants (ImageNet-derived for ViT)
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
_IMAGE_SIZE = 224


@dataclass
class MultimodalSample:
    """A single preprocessed image-text pair ready for model input."""
    pixel_values: torch.Tensor      # (3, 224, 224)
    input_ids: torch.Tensor         # (seq_len,)
    labels: torch.Tensor            # (seq_len,) with -100 padding
    attention_mask: torch.Tensor    # (seq_len,)
    domain: Optional[str] = None
    image_path: Optional[str] = None


class MedicalImageProcessor:
    """
    Preprocessing pipeline for medical images fed into BioMedCLIP.

    Handles DICOM, PNG, JPEG, and TIFF inputs with medical-domain
    appropriate normalisation.
    """

    def __init__(self, image_size: int = _IMAGE_SIZE) -> None:
        self._size = image_size
        self._transform = self._build_transform()

    def process(self, image) -> torch.Tensor:
        """
        Process a single PIL Image or file path into normalized pixel values.

        Parameters
        ----------
        image : PIL.Image or str or Path
            Input medical image.

        Returns
        -------
        torch.Tensor
            Shape (3, 224, 224), normalized per BioMedCLIP stats.
        """
        from PIL import Image as PILImage

        if isinstance(image, (str, Path)):
            image = PILImage.open(str(image)).convert("RGB")
        elif not hasattr(image, "convert"):
            raise TypeError(f"Expected PIL Image or path, got {type(image)}")

        image = image.convert("RGB")
        return self._transform(image)

    def process_batch(self, images: List) -> torch.Tensor:
        """
        Process a list of images into a batched tensor.

        Returns
        -------
        torch.Tensor
            Shape (B, 3, 224, 224).
        """
        tensors = [self.process(img) for img in images]
        return torch.stack(tensors, dim=0)

    def _build_transform(self):
        try:
            from torchvision import transforms
            return transforms.Compose([
                transforms.Resize((self._size, self._size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
            ])
        except ImportError:
            logger.warning("torchvision not available — using fallback transform.")
            return self._fallback_transform

    @staticmethod
    def _fallback_transform(image) -> torch.Tensor:
        """Minimal fallback when torchvision is absent."""
        arr = np.array(image.resize((_IMAGE_SIZE, _IMAGE_SIZE))).astype(np.float32) / 255.0
        arr = (arr - np.array(_CLIP_MEAN)) / np.array(_CLIP_STD)
        return torch.from_numpy(arr.transpose(2, 0, 1))


class MultimodalCollator:
    """
    DataCollator for multimodal batches.

    Pads sequences to the same length and stacks pixel tensors.
    Used by all four training stages.
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        pad_token_id: Optional[int] = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._pad_id = pad_token_id or tokenizer.pad_token_id or 0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids   = [f["input_ids"]     for f in features]
        labels      = [f["labels"]        for f in features]
        attn_masks  = [f["attention_mask"] for f in features]

        input_ids   = self._pad(input_ids,  self._pad_id)
        labels      = self._pad(labels,     -100)
        attn_masks  = self._pad(attn_masks, 0)

        batch = {
            "input_ids":      input_ids,
            "labels":         labels,
            "attention_mask": attn_masks,
        }

        if "pixel_values" in features[0]:
            pixel_values = torch.stack(
                [torch.tensor(f["pixel_values"]) for f in features], dim=0
            )
            batch["pixel_values"] = pixel_values

        return batch

    def _pad(self, sequences: List, pad_value: int) -> torch.Tensor:
        max_len = min(max(len(s) for s in sequences), self._max_length)
        padded = []
        for seq in sequences:
            seq = seq[:max_len]
            padded.append(seq + [pad_value] * (max_len - len(seq)))
        return torch.tensor(padded, dtype=torch.long)


def build_vqa_prompt(
    question: str,
    answer: str,
    domain: str,
    tokenizer,
    max_length: int = 2048,
    image_description: Optional[str] = None,
) -> Dict[str, List[int]]:
    """
    Construct a tokenized VQA prompt for Stage 2 specialist training.

    The prompt format follows the instruction-tuning template used
    for Llama-3.2-Instruct models.

    Parameters
    ----------
    question : str
        Clinical question.
    answer : str
        Ground truth clinical answer.
    domain : str
        Specialty domain for system prompt injection.
    tokenizer : transformers.PreTrainedTokenizer
    max_length : int
    image_description : str, optional
        Text description of the image.

    Returns
    -------
    dict with keys: input_ids, labels, attention_mask
    """
    system = (
        f"You are a board-certified {domain.replace('_', ' ')} specialist. "
        f"Analyze the provided medical image and answer the clinical question precisely."
    )

    image_ctx = f"[Image context: {image_description}]\n\n" if image_description else ""
    user_turn = f"{image_ctx}Question: {question}"

    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"{user_turn}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
        f"{answer}<|eot_id|>"
    )

    enc = tokenizer(
        prompt,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None,
    )

    # Mask the prompt portion in labels — only train on the answer
    answer_start = prompt.find("<|start_header_id|>assistant<|end_header_id|>")
    prompt_only  = prompt[:answer_start]
    prompt_ids   = tokenizer(prompt_only, add_special_tokens=False)["input_ids"]
    prompt_len   = len(prompt_ids)

    labels = enc["input_ids"].copy()
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids":      enc["input_ids"],
        "labels":         labels,
        "attention_mask": enc["attention_mask"],
    }
