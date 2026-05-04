"""
medlingo/utils/model_registry.py

Centralized model path resolution. All weight locations are resolved from
environment variables, never hardcoded. This allows the same codebase to
run in local dev, staging, and production environments by swapping .env files.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable contract
# ---------------------------------------------------------------------------
_ENV_WEIGHTS_DIR = "MEDLINGO_WEIGHTS_DIR"
_ENV_HF_TOKEN    = "HUGGINGFACE_TOKEN"
_ENV_CACHE_DIR   = "MEDLINGO_CACHE_DIR"

# Fallback HuggingFace Hub IDs used when local weights are absent.
# These are the exact backbones referenced in the MedLingo paper (Table I).
_HF_BACKBONE_IDS: Dict[str, str] = {
    "vision_encoder":   "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "specialist_base":  "meta-llama/Llama-3.2-1B-Instruct",
    "router_base":      "meta-llama/Llama-3.2-1B-Instruct",
    "adjudicator_base": "m42-health/Llama3-Med42-8B",
}


@dataclass
class ModelPaths:
    """Resolved paths for all MedLingo model components."""
    weights_base: Path
    vision_encoder: Path
    bridge_a: Path
    bridge_b: Path
    router: Path
    adjudicator: Path
    specialists: Dict[str, Path] = field(default_factory=dict)

    def all_trained_weights_exist(self) -> bool:
        """Returns True only when all four training stages have been completed."""
        required = [
            self.vision_encoder, self.bridge_a, self.bridge_b,
            self.router, self.adjudicator,
            *self.specialists.values(),
        ]
        return all(p.exists() for p in required)


class ModelRegistry:
    """
    Resolves model asset locations from the environment.

    Priority order for each component:
      1. Local weights under $MEDLINGO_WEIGHTS_DIR/<component>
      2. HuggingFace Hub (requires valid $HUGGINGFACE_TOKEN for gated models)

    Usage::

        registry = ModelRegistry()
        paths = registry.resolve()
        print(paths.adjudicator)
    """

    SPECIALIST_DOMAINS = [
        "pathology", "radiology", "dermatology",
        "cardiology", "neurology", "general_medicine",
    ]

    def __init__(self) -> None:
        self._weights_base = self._resolve_weights_base()
        self._hf_token: Optional[str] = os.getenv(_ENV_HF_TOKEN)
        self._cache_dir = Path(
            os.getenv(_ENV_CACHE_DIR, Path.home() / ".cache" / "medlingo")
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self) -> ModelPaths:
        """Return fully resolved ModelPaths object."""
        wb = self._weights_base
        specialists = {
            domain: wb / "specialists" / domain
            for domain in self.SPECIALIST_DOMAINS
        }

        paths = ModelPaths(
            weights_base=wb,
            vision_encoder=wb / "vision_encoder",
            bridge_a=wb / "bridge_pathway_a",
            bridge_b=wb / "bridge_pathway_b",
            router=wb / "router",
            adjudicator=wb / "adjudicator_dpo",
            specialists=specialists,
        )

        if not paths.all_trained_weights_exist():
            logger.warning(
                "⚠  Trained weights not found under %s. "
                "Backbone models will be fetched from HuggingFace Hub. "
                "Run the full training pipeline first to use fine-tuned weights.",
                wb,
            )
        else:
            logger.info("✓  All MedLingo weights resolved from %s", wb)

        return paths

    def backbone_id(self, component: str) -> str:
        """Return the HuggingFace Hub ID for a backbone component."""
        if component not in _HF_BACKBONE_IDS:
            raise KeyError(
                f"Unknown component '{component}'. "
                f"Valid keys: {list(_HF_BACKBONE_IDS)}"
            )
        return _HF_BACKBONE_IDS[component]

    @property
    def hf_token(self) -> Optional[str]:
        return self._hf_token

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_weights_base() -> Path:
        raw = os.getenv(_ENV_WEIGHTS_DIR)
        if raw:
            path = Path(raw).expanduser().resolve()
            path.mkdir(parents=True, exist_ok=True)
            logger.debug("Weight directory: %s", path)
            return path

        # Graceful fallback so the codebase runs without any .env setup
        fallback = Path.home() / ".medlingo" / "weights"
        fallback.mkdir(parents=True, exist_ok=True)
        logger.warning(
            "%s not set. Using fallback: %s. "
            "Set this variable in your .env file for production use.",
            _ENV_WEIGHTS_DIR,
            fallback,
        )
        return fallback


# Module-level singleton
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Return (or lazily create) the global ModelRegistry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def resolve_paths() -> ModelPaths:
    """Convenience wrapper around get_registry().resolve()."""
    return get_registry().resolve()
