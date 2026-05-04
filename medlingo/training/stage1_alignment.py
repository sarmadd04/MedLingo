"""
medlingo/training/stage1_alignment.py

Stage 1: Dual-Path Multimodal Alignment — §IV-A.

Trains the two Linear Projection Bridges (Pathway A and B) while all LLM
and vision encoder parameters remain frozen. The bridges learn to map
BioMedCLIP 768-dim embeddings into the specialist (2048) and adjudicator
(4096) latent spaces via cross-entropy loss on clinical token prediction.

Dataset : LLaVA-Med (~600k image-caption pairs)
LR      : 1e-3, cosine schedule
Epochs  : 1
Batch   : 256 (effective via gradient accumulation)
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from medlingo.core.vision_encoder import VisionEncoder
from medlingo.core.projection_bridges import DualPathBridges
from medlingo.utils.model_registry import resolve_paths
from medlingo.utils.memory_manager import get_memory_manager
from medlingo.utils.logging_utils import StageTimer

logger = logging.getLogger(__name__)


class AlignmentLoss(nn.Module):
    """
    Cross-entropy alignment loss per Eq. 4:
        L_align = -Σ log P(x_t | x_<t, V_tokens)

    We implement this by prepending the projected visual tokens to the
    LLM input embedding sequence and computing the standard causal LM loss.
    """

    def __init__(self) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        logits: torch.Tensor,        # (B, T, V)
        labels: torch.Tensor,        # (B, T)
    ) -> torch.Tensor:
        B, T, V = logits.shape
        return self.cross_entropy(logits.view(B * T, V), labels.view(B * T))


class Stage1Trainer:
    """
    Trains the dual projection bridges for multimodal alignment.

    Only PathwayA and PathwayB parameters receive gradient updates.
    The vision encoder and both LLM backbones are fully frozen.

    Parameters
    ----------
    specialist_backbone_id : str
        HF Hub ID or local path for the 1B specialist backbone.
    adjudicator_backbone_id : str
        HF Hub ID or local path for the 8B adjudicator backbone.
    output_dir : str or Path
        Directory to save trained bridge weights.
    device : str
    """

    def __init__(
        self,
        specialist_backbone_id: Optional[str] = None,
        adjudicator_backbone_id: Optional[str] = None,
        output_dir: Optional[str] = None,
        device: str = "cuda",
    ) -> None:
        paths = resolve_paths()
        self._spec_id  = specialist_backbone_id  or "meta-llama/Llama-3.2-1B-Instruct"
        self._orch_id  = adjudicator_backbone_id or "m42-health/Llama3-Med42-8B"
        self._out_dir  = Path(output_dir or paths.weights_base)
        self._device   = device
        self._mm       = get_memory_manager()

    # ------------------------------------------------------------------

    def train(
        self,
        dataset,
        learning_rate: float = 1e-3,
        num_epochs: int = 1,
        per_device_batch_size: int = 16,
        gradient_accumulation_steps: int = 16,
        warmup_ratio: float = 0.03,
        max_seq_length: int = 2048,
    ) -> None:
        """
        Run Stage 1 alignment training.

        Parameters
        ----------
        dataset : datasets.Dataset or torch.utils.data.Dataset
            LLaVA-Med image-caption pairs.
        """
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM,
            get_cosine_schedule_with_warmup,
        )
        from medlingo.utils.quantization import nf4_config

        with StageTimer("Stage 1: Dual-Path Alignment"):

            # Load frozen backbones (NF4 to save memory)
            logger.info("Loading frozen 1B specialist backbone (NF4)...")
            spec_tokenizer = AutoTokenizer.from_pretrained(self._spec_id)
            spec_model = AutoModelForCausalLM.from_pretrained(
                self._spec_id,
                quantization_config=nf4_config(),
                device_map=self._device,
            )
            for p in spec_model.parameters():
                p.requires_grad_(False)

            # Initialize trainable projection bridges
            bridges = DualPathBridges(dropout=0.1).to(self._device)
            vision_enc = VisionEncoder(device=self._device)

            trainable_params = list(bridges.parameters())
            n_trainable = sum(p.numel() for p in trainable_params)
            logger.info("Trainable parameters (bridges only): %s", f"{n_trainable:,}")

            optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
            loss_fn = AlignmentLoss()

            from torch.utils.data import DataLoader
            loader = DataLoader(
                dataset,
                batch_size=per_device_batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
            )

            total_steps = len(loader) * num_epochs // gradient_accumulation_steps
            warmup_steps = int(total_steps * warmup_ratio)

            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )

            step = 0
            optimizer.zero_grad()

            for epoch in range(num_epochs):
                bridges.train()
                epoch_loss = 0.0

                for batch_idx, batch in enumerate(loader):
                    images      = batch["pixel_values"].to(self._device)
                    input_ids   = batch["input_ids"].to(self._device)
                    labels      = batch["labels"].to(self._device)

                    # Visual encoding (frozen)
                    with torch.no_grad():
                        visual_features = vision_enc(images)       # (B, 768)

                    # Project to specialist space (trainable)
                    z_s, _ = bridges(visual_features)              # (B, 2048)

                    # Prepend visual tokens as embeddings to the LLM
                    with torch.no_grad():
                        text_embeds = spec_model.get_input_embeddings()(input_ids)  # (B,T,H)

                    # Project z_s to match hidden dim and prepend
                    visual_prefix = z_s.unsqueeze(1)               # (B, 1, 2048)
                    combined = torch.cat([visual_prefix, text_embeds], dim=1)

                    # Shift labels for visual prefix
                    prefix_labels = torch.full(
                        (labels.shape[0], 1), -100,
                        dtype=labels.dtype, device=self._device,
                    )
                    shifted_labels = torch.cat([prefix_labels, labels], dim=1)

                    # Forward pass through frozen backbone
                    with torch.no_grad():
                        outputs = spec_model(
                            inputs_embeds=combined,
                            labels=None,
                        )
                    logits = outputs.logits                         # (B, T+1, V)

                    # Alignment loss (Eq. 4)
                    loss = loss_fn(logits, shifted_labels)
                    loss = loss / gradient_accumulation_steps
                    loss.backward()

                    epoch_loss += loss.item() * gradient_accumulation_steps

                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        step += 1

                        if step % 50 == 0:
                            logger.info(
                                "Stage1 | Epoch %d | Step %d | Loss: %.4f | LR: %.2e",
                                epoch + 1, step, epoch_loss / (batch_idx + 1),
                                scheduler.get_last_lr()[0],
                            )

            # Save bridge weights
            save_dir = self._out_dir / "bridges"
            bridges.save(save_dir)
            logger.info("Stage 1 complete. Bridges saved to %s", save_dir)
