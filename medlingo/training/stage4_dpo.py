"""
medlingo/training/stage4_dpo.py

Stage 4: Grounded Adjudication via Multimodal DPO — §IV-D.

Aligns the 8B Chief Adjudicator to prefer visually-grounded clinical
reports over hallucinated alternatives using Direct Preference Optimization.

DPO objective (Eq. 6):
  L_DPO = -E[(x,y_w,y_l)] [ log σ ( β·log(π_θ(y_w|x)/π_ref(y_w|x))
                                    - β·log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]

where:
  y_w = grounded clinical report (chosen)
  y_l = hallucinated report (rejected)
  β   = 0.1 (prevents excessive drift from reference policy)

Dataset : MIMIC-IV-VQA, Quilt-1M, MedMNIST-Brain, MedQA-USMLE, PAD-UFES-20
LoRA    : r=128, α=256
LR      : 5e-6, linear schedule
Epochs  : 1
Batch   : 32 (effective)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict

import torch

from medlingo.utils.model_registry import resolve_paths
from medlingo.utils.logging_utils import StageTimer

logger = logging.getLogger(__name__)


class Stage4DPOTrainer:
    """
    DPO fine-tuning for the 8B Chief Adjudicator.

    Constructs preference triplets (x, y_w, y_l) where:
      x   = image tokens (Pathway B) + specialist SPR transcript
      y_w = grounded report corroborated by visual evidence
      y_l = hallucinated or visually inconsistent report

    Uses TRL's DPOTrainer for the preference optimization loop.

    Parameters
    ----------
    base_model_path : str, optional
        Path or HF Hub ID for the 8B adjudicator backbone.
    output_dir : str or Path, optional
        Where to save the DPO-aligned adapter.
    device : str
    beta : float
        DPO regularization coefficient (paper: 0.1).
    """

    def __init__(
        self,
        base_model_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        device: str = "cuda",
        beta: float = 0.1,
    ) -> None:
        paths = resolve_paths()
        self._base_path = base_model_path or "m42-health/Llama3-Med42-8B"
        self._out_dir   = Path(output_dir or paths.weights_base) / "adjudicator_dpo"
        self._device    = device
        self._beta      = beta

    # ------------------------------------------------------------------

    def train(
        self,
        dataset,
        lora_r: int = 128,
        lora_alpha: int = 256,
        learning_rate: float = 5e-6,
        num_epochs: int = 1,
        per_device_batch_size: int = 2,
        gradient_accumulation_steps: int = 16,
        warmup_ratio: float = 0.05,
        weight_decay: float = 0.05,
        max_length: int = 4096,
        max_prompt_length: int = 2048,
    ) -> None:
        """
        Run DPO alignment on the adjudicator.

        Parameters
        ----------
        dataset : datasets.Dataset
            Must contain columns: 'prompt', 'chosen', 'rejected'.
            'chosen' = visually grounded report
            'rejected' = hallucinated alternative
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
        from peft import LoraConfig, TaskType, get_peft_model
        from trl import DPOTrainer, DPOConfig
        from medlingo.utils.quantization import nf4_config

        with StageTimer("Stage 4: DPO Adjudicator Alignment"):

            tokenizer = AutoTokenizer.from_pretrained(self._base_path)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"     # Required for DPO

            # Policy model (trainable)
            logger.info("Loading policy model (8B, NF4)...")
            policy_model = AutoModelForCausalLM.from_pretrained(
                self._base_path,
                quantization_config=nf4_config(),
                device_map=self._device,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
            )
            policy_model.gradient_checkpointing_enable()

            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            policy_model = get_peft_model(policy_model, lora_cfg)
            policy_model.print_trainable_parameters()

            # Reference model (frozen copy — π_ref in Eq. 6)
            logger.info("Loading frozen reference model (8B, NF4)...")
            ref_model = AutoModelForCausalLM.from_pretrained(
                self._base_path,
                quantization_config=nf4_config(),
                device_map=self._device,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
            )
            for p in ref_model.parameters():
                p.requires_grad_(False)

            # DPO training config
            dpo_config = DPOConfig(
                output_dir=str(self._out_dir / "checkpoints"),
                num_train_epochs=num_epochs,
                per_device_train_batch_size=per_device_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                lr_scheduler_type="linear",
                bf16=True,
                logging_steps=10,
                save_steps=50,
                beta=self._beta,
                max_length=max_length,
                max_prompt_length=max_prompt_length,
                loss_type="sigmoid",
                label_smoothing=0.0,
                report_to="none",
                remove_unused_columns=False,
            )

            dpo_trainer = DPOTrainer(
                model=policy_model,
                ref_model=ref_model,
                args=dpo_config,
                train_dataset=dataset,
                tokenizer=tokenizer,
            )

            dpo_trainer.train()

            # Save DPO-aligned LoRA adapter
            policy_model.save_pretrained(str(self._out_dir))
            tokenizer.save_pretrained(str(self._out_dir))
            logger.info("DPO-aligned adjudicator adapter saved to %s", self._out_dir)
