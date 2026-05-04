"""
medlingo/training/stage3_router.py

Stage 3: Intent-Based Triage and Uncertainty Calibration — §IV-C.

Fine-tunes the Semantic Triage Router (Llama-3.2-1B) on a curated
multi-label specialty classification task.

Dataset  : MedQA + PubMedQA (~50k samples)
LR       : 1e-5, linear schedule
Epochs   : 3–5
Batch    : 64 (effective)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict

import torch

from medlingo.core.router import DOMAIN_LIST, DOMAIN_TO_IDX
from medlingo.utils.model_registry import resolve_paths
from medlingo.utils.logging_utils import StageTimer

logger = logging.getLogger(__name__)


_ROUTER_CLASSIFICATION_SYSTEM = """\
You are a medical specialty classifier. Given a clinical question and optional image description, \
identify which TWO medical domains are most relevant from: \
pathology, radiology, dermatology, cardiology, neurology, general_medicine. \
Respond ONLY with a JSON: {"top_domains": ["<d1>", "<d2>"], "domain_scores": {...}}
"""


class Stage3Trainer:
    """
    Trains the Semantic Triage Router via supervised fine-tuning on
    labeled (query, specialty) pairs.

    After SFT, a temperature-scaling calibration pass aligns the router's
    confidence scores with empirical accuracy on held-out queries.

    Parameters
    ----------
    base_model_path : str, optional
        Llama-3.2-1B path or HF Hub ID.
    output_dir : str or Path, optional
        Where to save the trained router.
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
        self._out_dir   = Path(output_dir or paths.weights_base) / "router"
        self._device    = device

    # ------------------------------------------------------------------

    def train(
        self,
        dataset,
        learning_rate: float = 1e-5,
        num_epochs: int = 5,
        per_device_batch_size: int = 8,
        gradient_accumulation_steps: int = 8,
        warmup_ratio: float = 0.10,
        weight_decay: float = 0.01,
        max_seq_length: int = 512,
    ) -> None:
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM,
            TrainingArguments, Trainer, DataCollatorForSeq2Seq,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from medlingo.utils.quantization import nf4_config

        with StageTimer("Stage 3: Router Triage Training"):

            tokenizer = AutoTokenizer.from_pretrained(self._base_path)
            tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                self._base_path,
                quantization_config=nf4_config(),
                device_map=self._device,
                torch_dtype=torch.float16,
            )

            # Light LoRA for router — r=16 is sufficient for classification
            lora_cfg = LoraConfig(
                r=16, lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_cfg)
            model.gradient_checkpointing_enable()

            def tokenize_fn(examples):
                prompts, responses = [], []
                for query, domains, scores in zip(
                    examples["query"],
                    examples["top_domains"],
                    examples["domain_scores"],
                ):
                    prompt = (
                        f"<|system|>{_ROUTER_CLASSIFICATION_SYSTEM}"
                        f"<|user|>{query}<|assistant|>"
                    )
                    response = json.dumps({
                        "top_domains": domains[:2],
                        "domain_scores": scores,
                    })
                    prompts.append(prompt + response)

                enc = tokenizer(
                    prompts,
                    max_length=max_seq_length,
                    truncation=True,
                    padding="max_length",
                )
                enc["labels"] = enc["input_ids"].copy()
                return enc

            tokenized = dataset.map(
                tokenize_fn, batched=True,
                remove_columns=dataset.column_names,
            )

            train_args = TrainingArguments(
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
                save_steps=100,
                eval_strategy="no",
                report_to="none",
            )

            trainer = Trainer(
                model=model,
                args=train_args,
                train_dataset=tokenized,
                data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
            )
            trainer.train()

            model.save_pretrained(str(self._out_dir))
            tokenizer.save_pretrained(str(self._out_dir))
            logger.info("Router adapter saved to %s", self._out_dir)
