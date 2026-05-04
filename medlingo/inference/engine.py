"""
medlingo/inference/engine.py

MedLingo Inference Engine.

Wraps the full pipeline behind a clean, serializable interface that can
be instantiated once and queried repeatedly — suitable for REST APIs,
batch evaluation, or interactive diagnostic sessions.

The engine enforces the single-load / reuse pattern: all models are loaded
once on `initialize()`, then kept resident in VRAM for subsequent queries.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

from medlingo.pipeline.medlingo_pipeline import MedLingoPipeline
from medlingo.core.adjudicator import AdjudicationVerdict
from medlingo.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticRequest:
    """A single diagnostic query to the MedLingo engine."""
    query: str
    image_path: Optional[str] = None
    image_description: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosticResponse:
    """Structured response returned by the inference engine."""
    request_id: Optional[str]
    verdict: str
    visual_grounding: str
    conflict_resolution: Optional[str]
    expert_domains: List[str]
    expert_weights: Dict[str, float]
    overall_confidence: str
    latency_ms: float
    raw_verdict: AdjudicationVerdict = field(repr=False, default=None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id":         self.request_id,
            "verdict":            self.verdict,
            "visual_grounding":   self.visual_grounding,
            "conflict_resolution":self.conflict_resolution,
            "expert_domains":     self.expert_domains,
            "expert_weights":     self.expert_weights,
            "overall_confidence": self.overall_confidence,
            "latency_ms":         round(self.latency_ms, 1),
        }

    def __str__(self) -> str:
        if self.raw_verdict:
            return self.raw_verdict.format_response()
        return self.verdict


class MedLingoEngine:
    """
    Production inference engine for MedLingo.

    Initializes all pipeline components once and provides a thread-safe
    `diagnose()` method for on-demand inference.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to inference_config.yaml.
    device : str
        "auto", "cuda", or "cpu".
    verbose : bool
        If True, log routing and SPR details for each request.

    Example
    -------
    >>> engine = MedLingoEngine()
    >>> engine.initialize()
    >>> response = engine.diagnose(DiagnosticRequest(
    ...     query="Is there evidence of pneumothorax?",
    ...     image_path="chest_xray.jpg",
    ... ))
    >>> print(response)
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
        verbose: bool = False,
    ) -> None:
        self._pipeline: Optional[MedLingoPipeline] = None
        self._config_path = config_path
        self._device = device
        self._verbose = verbose
        self._request_count = 0
        self._total_latency_ms = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Load all pipeline components into VRAM.
        Call once before the first `diagnose()` invocation.
        """
        logger.info("Initializing MedLingo inference engine...")
        self._pipeline = MedLingoPipeline(
            config_path=self._config_path,
            device=self._device,
        )
        self._pipeline.warm_up()
        logger.info("MedLingo engine ready.")

    def shutdown(self) -> None:
        """Release GPU memory and clean up."""
        import gc
        import torch

        self._pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("MedLingo engine shut down.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def diagnose(self, request: DiagnosticRequest) -> DiagnosticResponse:
        """
        Run a full MedLingo diagnostic pass on a single request.

        Parameters
        ----------
        request : DiagnosticRequest
            Encapsulates the query and optional image path.

        Returns
        -------
        DiagnosticResponse
            Structured response with verdict, grounding, and metadata.
        """
        if self._pipeline is None:
            logger.warning("Engine not initialized — calling initialize() now.")
            self.initialize()

        t0 = time.perf_counter()
        verdict = self._pipeline.diagnose(
            image=request.image_path,
            query=request.query,
            image_description=request.image_description,
            verbose=self._verbose,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        self._request_count += 1
        self._total_latency_ms += latency_ms

        response = DiagnosticResponse(
            request_id=request.request_id,
            verdict=verdict.verdict,
            visual_grounding=verdict.grounding_summary,
            conflict_resolution=verdict.conflict_resolution,
            expert_domains=list(verdict.expert_weights.keys()),
            expert_weights=verdict.expert_weights,
            overall_confidence=verdict.overall_confidence,
            latency_ms=latency_ms,
            raw_verdict=verdict,
        )

        logger.info(
            "Request %s completed in %.0f ms | confidence=%s",
            request.request_id or self._request_count,
            latency_ms,
            verdict.overall_confidence,
        )

        return response

    def diagnose_batch(
        self, requests: List[DiagnosticRequest]
    ) -> List[DiagnosticResponse]:
        """Process a list of diagnostic requests sequentially."""
        return [self.diagnose(req) for req in requests]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def stats(self) -> Dict[str, Any]:
        avg_latency = (
            self._total_latency_ms / self._request_count
            if self._request_count else 0.0
        )
        return {
            "requests_served": self._request_count,
            "average_latency_ms": round(avg_latency, 1),
            "total_latency_ms": round(self._total_latency_ms, 1),
        }

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, *_):
        self.shutdown()
