"""
medlingo/training/train_pipeline.py

Master Training Orchestrator.

Executes all four MedLingo training stages in the correct sequential order,
enforcing dependencies between stages (e.g. bridges must be trained before
specialists can use visual tokens).

Stage dependency chain:
    Stage 1 (Alignment)
        └─► Stage 2 (Specialists)   <- uses trained bridges
                └─► Stage 3 (Router)
                        └─► Stage 4 (DPO Adjudicator)  <- uses SPR from stages 2+3

Usage (CLI):
    python -m medlingo.training.train_pipeline --stages 1 2 3 4

Usage (Python):
    from medlingo.training.train_pipeline import MedLingoTrainer
    trainer = MedLingoTrainer()
    trainer.run(stages=[1, 2, 3, 4])
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

from medlingo.utils.model_registry import resolve_paths
from medlingo.utils.logging_utils import setup_logging, StageTimer
from medlingo.utils.memory_manager import get_memory_manager

logger = logging.getLogger(__name__)


class MedLingoTrainer:
    """
    Top-level orchestrator for the full MedLingo 4-stage training pipeline.

    Each stage is executed independently; completed stages are checkpointed
    so the pipeline can be resumed from any point after interruption.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to training_config.yaml. Uses package default if not provided.
    device : str
        "cuda" (recommended), "cpu" (slow, for testing only).
    resume_from_stage : int
        Skip already-completed stages and resume from this stage index.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        device: str = "cuda",
        resume_from_stage: int = 1,
    ) -> None:
        self._device = device
        self._resume = resume_from_stage
        self._paths  = resolve_paths()
        self._cfg    = self._load_config(config_path)
        self._mm     = get_memory_manager()

    def run(self, stages: List[int] = None) -> None:
        if stages is None:
            stages = [1, 2, 3, 4]

        stages = [s for s in sorted(stages) if s >= self._resume]

        logger.info(
            "MedLingo Training Pipeline | device=%s | stages=%s",
            self._device, stages,
        )
        logger.info("Weight output root: %s", self._paths.weights_base)

        with StageTimer("Full MedLingo Training Pipeline"):
            for stage_idx in stages:
                if   stage_idx == 1: self._run_stage1()
                elif stage_idx == 2: self._run_stage2()
                elif stage_idx == 3: self._run_stage3()
                elif stage_idx == 4: self._run_stage4()
                else:
                    logger.warning("Unknown stage index %d — skipping.", stage_idx)
                self._mm.log_snapshot(tag=f"post-stage{stage_idx}")

        logger.info("All requested stages complete.")
        logger.info("Weights directory: %s", self._paths.weights_base)

    def _run_stage1(self) -> None:
        from medlingo.training.stage1_alignment import Stage1Trainer
        from medlingo.data.loaders import build_stage1_dataset
        cfg = self._cfg.get("stage1", {})
        dataset = build_stage1_dataset(max_samples=cfg.get("dataset_size", 600_000))
        Stage1Trainer(output_dir=str(self._paths.weights_base), device=self._device).train(
            dataset=dataset,
            learning_rate=cfg.get("learning_rate", 1e-3),
            num_epochs=cfg.get("num_train_epochs", 1),
            per_device_batch_size=cfg.get("per_device_train_batch_size", 16),
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 16),
            warmup_ratio=cfg.get("warmup_ratio", 0.03),
            max_seq_length=cfg.get("max_seq_length", 2048),
        )

    def _run_stage2(self) -> None:
        from medlingo.training.stage2_specialist import Stage2Trainer
        from medlingo.data.loaders import build_stage2_datasets
        cfg = self._cfg.get("stage2", {})
        domain_datasets = build_stage2_datasets(
            max_samples_per_domain=cfg.get("combined_dataset_size", 300_000) // 6,
        )
        Stage2Trainer(output_dir=str(self._paths.weights_base), device=self._device).train_all(
            domain_datasets=domain_datasets,
            lora_r=cfg.get("lora_r", 64),
            lora_alpha=cfg.get("lora_alpha", 128),
            learning_rate=cfg.get("learning_rate", 2e-4),
            num_epochs=cfg.get("num_train_epochs", 3),
            per_device_batch_size=cfg.get("per_device_train_batch_size", 8),
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 16),
            warmup_ratio=cfg.get("warmup_ratio", 0.03),
            weight_decay=cfg.get("weight_decay", 0.1),
        )

    def _run_stage3(self) -> None:
        from medlingo.training.stage3_router import Stage3Trainer
        from medlingo.data.loaders import build_stage3_dataset
        cfg = self._cfg.get("stage3", {})
        dataset = build_stage3_dataset(max_samples=cfg.get("dataset_size", 50_000))
        Stage3Trainer(output_dir=str(self._paths.weights_base), device=self._device).train(
            dataset=dataset,
            learning_rate=cfg.get("learning_rate", 1e-5),
            num_epochs=cfg.get("num_train_epochs", 5),
            per_device_batch_size=cfg.get("per_device_train_batch_size", 8),
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
            warmup_ratio=cfg.get("warmup_ratio", 0.10),
            weight_decay=cfg.get("weight_decay", 0.01),
        )

    def _run_stage4(self) -> None:
        from medlingo.training.stage4_dpo import Stage4DPOTrainer
        from medlingo.data.loaders import build_stage4_dataset
        cfg = self._cfg.get("stage4", {})
        dataset = build_stage4_dataset()
        Stage4DPOTrainer(
            output_dir=str(self._paths.weights_base),
            device=self._device,
            beta=cfg.get("dpo", {}).get("beta", 0.1),
        ).train(
            dataset=dataset,
            lora_r=cfg.get("lora_r", 128),
            lora_alpha=cfg.get("lora_alpha", 256),
            learning_rate=cfg.get("learning_rate", 5e-6),
            num_epochs=cfg.get("num_train_epochs", 1),
            per_device_batch_size=cfg.get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 16),
            warmup_ratio=cfg.get("warmup_ratio", 0.05),
            weight_decay=cfg.get("weight_decay", 0.05),
            max_length=cfg.get("max_seq_length", 4096),
        )

    @staticmethod
    def _load_config(config_path) -> dict:
        import yaml
        from pathlib import Path as P
        if config_path is None:
            config_path = (
                P(__file__).parent.parent.parent / "configs" / "training_config.yaml"
            )
        config_path = P(config_path)
        if not config_path.exists():
            logger.warning("Training config not found at %s — using defaults.", config_path)
            return {}
        with open(config_path) as f:
            return yaml.safe_load(f) or {}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MedLingo Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m medlingo.training.train_pipeline --stages 1 2 3 4
  python -m medlingo.training.train_pipeline --stages 2
  python -m medlingo.training.train_pipeline --stages 3 4 --resume-from 3
  python -m medlingo.training.train_pipeline --stages 4 --device cpu
        """,
    )
    parser.add_argument("--stages", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-from", type=int, default=1, dest="resume_from")
    parser.add_argument("--config", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    setup_logging(level=args.log_level)
    MedLingoTrainer(
        config_path=args.config,
        device=args.device,
        resume_from_stage=args.resume_from,
    ).run(stages=args.stages)


def main():
    """setuptools entry point — called by the `medlingo-train` console script."""
    args = _parse_args()
    setup_logging(level=args.log_level)
    MedLingoTrainer(
        config_path=args.config,
        device=args.device,
        resume_from_stage=args.resume_from,
    ).run(stages=args.stages)
