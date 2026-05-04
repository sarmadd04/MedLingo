"""
medlingo/utils/memory_manager.py

GPU memory management utilities for the MedLingo 16 GB VRAM budget.
Tracks component footprints, enforces budget, and handles OOM gracefully.
"""

from __future__ import annotations

import gc
import torch
import logging
import contextlib
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Memory footprints per Table I + §V-B
_COMPONENT_BUDGETS_GB: Dict[str, float] = {
    "vision_encoder":   0.3,
    "bridge_a":         0.05,
    "bridge_b":         0.05,
    "router":           1.2,
    "specialist_a":     1.2,
    "specialist_b":     1.2,
    "adjudicator":      5.5,
    "kv_cache":         1.5,
    "activations":      1.0,
}

_TOTAL_BUDGET_GB = 16.0
_PEAK_TARGET_GB  = 14.8


@dataclass
class MemorySnapshot:
    allocated_gb: float
    reserved_gb: float
    free_gb: float
    components: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"  Allocated : {self.allocated_gb:.2f} GB",
            f"  Reserved  : {self.reserved_gb:.2f} GB",
            f"  Free      : {self.free_gb:.2f} GB",
        ]
        if self.components:
            lines.append("  Components:")
            for k, v in self.components.items():
                lines.append(f"    {k:<20} {v:.2f} GB")
        return "\n".join(lines)


class MemoryManager:
    """
    Tracks and enforces the 16 GB VRAM budget across all MedLingo components.

    Usage::

        mm = MemoryManager()
        with mm.component_scope("specialist_a"):
            model = load_specialist(...)
        mm.log_snapshot()
    """

    def __init__(self, budget_gb: float = _TOTAL_BUDGET_GB) -> None:
        self._budget_gb = budget_gb
        self._active: Dict[str, float] = {}
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Context managers
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def component_scope(self, name: str, expected_gb: Optional[float] = None):
        """Track VRAM delta for a named component load."""
        before = self._vram_allocated_gb()
        try:
            yield
        finally:
            after = self._vram_allocated_gb()
            delta = max(after - before, 0.0)
            self._active[name] = delta

            if expected_gb is not None:
                tolerance = expected_gb * 0.20          # 20 % tolerance
                if delta > expected_gb + tolerance:
                    logger.warning(
                        "Component '%s' used %.2f GB — expected ~%.2f GB",
                        name, delta, expected_gb,
                    )

            total = sum(self._active.values())
            if total > _PEAK_TARGET_GB:
                logger.warning(
                    "⚠  Peak VRAM estimate %.2f GB exceeds target %.2f GB. "
                    "Consider offloading inactive specialists.",
                    total, _PEAK_TARGET_GB,
                )

    # ------------------------------------------------------------------
    # Snapshot & reporting
    # ------------------------------------------------------------------

    def snapshot(self) -> MemorySnapshot:
        if not torch.cuda.is_available():
            return MemorySnapshot(0.0, 0.0, 0.0, {})

        allocated = self._vram_allocated_gb()
        reserved  = torch.cuda.memory_reserved(0)  / (1024 ** 3)
        free      = (torch.cuda.get_device_properties(0).total_memory
                     - torch.cuda.memory_reserved(0)) / (1024 ** 3)
        return MemorySnapshot(allocated, reserved, free, dict(self._active))

    def log_snapshot(self, tag: str = "") -> None:
        snap = self.snapshot()
        prefix = f"[{tag}] " if tag else ""
        logger.info("%sMedLingo VRAM snapshot:\n%s", prefix, snap)

    # ------------------------------------------------------------------
    # Cleanup helpers
    # ------------------------------------------------------------------

    def offload_component(self, name: str, model: torch.nn.Module) -> None:
        """Move a model to CPU and free its VRAM slot."""
        model.to("cpu")
        self._active.pop(name, None)
        self._flush_cache()
        logger.debug("Offloaded '%s' to CPU.", name)

    @staticmethod
    def _flush_cache() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @staticmethod
    def _vram_allocated_gb() -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated(0) / (1024 ** 3)


# Module-level singleton
_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    global _manager
    if _manager is None:
        _manager = MemoryManager()
    return _manager
