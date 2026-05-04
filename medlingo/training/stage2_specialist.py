"""
medlingo/training/stage2_specialist.py

Stage 2: Specialist Knowledge Injection — §IV-B.

Fine-tunes six domain-specific LoRA adapters on top of the frozen
Llama-3.2-1B backbone using PEFT.

LoRA config  : r=64, α=128 per Table II
Domains      : pathology, radiology, dermatology, cardiology, neurology, general_medicine
Datasets     : PathVQA, RadVQA, SLAKE, PMC-VQA (~300k multimodal pairs total)
LR           : 2e-4, cosine schedule
Epochs       : 3
Batch        : 128 (effective)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, List

import torch

from medlingo.utils.model_registry import resolve_paths
from medlingo.utils.logging_utils import StageTimer

logger = logging.getLogger(__name__)

SPECIALIST_DOMAINS = [
    "pathology",
    "radiology",
    "dermatology",
    "cardiology",
    "neurology",
    "general_medicine",
]


class Stage2Trainer:
    """
    Trains six domain LoRA adapters on the Llama-3.2-1B backbone.

    Each adapter is trained independently on its domain-specific VQA dataset
    and saved to a separate adapter directory. Enables Parallel Hot-Swapping
    during inference.

    Parameters
    ----------
    base_model_path : str, optional
        Llama-3.2-1B HF Hub ID or local path.
    output_dir : str or Path, optional
        Root directory for adapter checkpoints.
    device : str
    """

    def __init__(
        self,
        base_model_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        device: str = "cuda",
    ) -> None:
        paths = resolve_paths()
        self._base_path = base_model_path or "meta-llama/Llama-3.2-1B-Instruct"
        self._out_root  = Path(output_dir or paths.weights_base) / "specialists"
        self._device    = device

    # ------------------------------------------------------------------

    def train_all(
        self,
        domain_datasets: Dict[str, object],
        lora_r: int = 64,
        lora_alpha: int = 128,
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        per_device_batch_size: int = 8,
        gradient_accumulation_steps: int = 16,
        warmup_ratio: float = 0.03,
        weight_decay: float = 0.1,
        max_seq_length: int = 2048,
    ) -> None:
        """
        Train one LoRA adapter per domain sequentially.

        Parameters
        ----------
        domain_datasets : dict
            Maps domain name → HuggingFace/torch Dataset of VQA pairs.
        """
        for domain in SPECIALIST_DOMAINS:
            if domain not in domain_datasets:
                logger.warning("No dataset for domain '%s' — skipping.", domain)
                continue

            self.train_domain(
                domain=domain,
                dataset=domain_datasets[domain],
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                per_device_batch_size=per_device_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_ratio=warmup_ratio,
                weight_decay=weight_decay,
                max_seq_length=max_seq_length,
            )

    def train_domain(
        self,
        domain: str,
        dataset,
        lora_r: int = 64,
        lora_alpha: int = 128,
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        per_device_batch_size: int = 8,
        gradient_accumulation_steps: int = 16,
        warmup_ratio: float = 0.03,
        weight_decay: float = 0.1,
        max_seq_length: int = 2048,
    ) -> None:
        """Train a single domain specialist adapter."""
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM,
            TrainingArguments, Trainer, DataCollatorForSeq2Seq,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from medlingo.utils.quantization import nf4_config

        save_path = self._out_root / domain

        with StageTimer(f"Stage 2: {domain.replace('_', ' ').title()} Specialist"):
            logger.info("Loading base backbone for domain: %s", domain)

            tokenizer = AutoTokenizer.from_pretrained(self._base_path)
            tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                self._base_path,
                quantization_config=nf4_config(),
                device_map=self._device,
                torch_dtype=torch.float16,
            )

            # Enable gradient checkpointing before LoRA wrapping
            model.gradient_checkpointing_enable()

            lora_config = LoraConfig(
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

            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

            # Domain-specific system prompt injected during tokenization
            def tokenize_fn(examples):
                system = (
                    f"You are a board-certified {domain.replace('_', ' ')} specialist. "
                    f"Answer the following clinical question precisely."
                )
                prompts = [
                    f"<|system|>{system}<|user|>{q}<|assistant|>{a}"
                    for q, a in zip(examples["question"], examples["answer"])
                ]
                enc = tokenizer(
                    prompts,
                    max_length=max_seq_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                enc["labels"] = enc["input_ids"].clone()
                return enc

            tokenized = dataset.map(
                tokenize_fn,
                batched=True,
                remove_columns=dataset.column_names,
            )

            args = TrainingArguments(
                output_dir=str(save_path / "checkpoints"),
                num_train_epochs=num_epochs,
                per_device_train_batch_size=per_device_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                lr_scheduler_type="cosine",
                bf16=True,
                logging_steps=25,
                save_steps=250,
                eval_strategy="no",
                remove_unused_columns=False,
                report_to="none",
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=tokenized,
                data_collator=DataCollatorForSeq2Seq(
                    tokenizer, model=model, padding=True
                ),
            )

            trainer.train()

            # Save LoRA adapter only (not base weights)
            model.save_pretrained(str(save_path))
            tokenizer.save_pretrained(str(save_path))
            logger.info("Adapter saved: %s", save_path)
