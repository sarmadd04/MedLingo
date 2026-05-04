"""
medlingo/pipeline/parallel_executor.py

Parallel Expert Executor — §III-B.

Manages the two parallel Llama-3.2-1B instances, hot-swapping LoRA adapters
between domains with < 150ms switching latency as stated in §V-D.

"The architecture initializes two parallel instances of the Llama-3.2-1B
model backbone. At runtime, the system performs a Parallel Hot-Swap,
loading the router-selected LoRA adapters into each respective model instance."
"""

from __future__ import annotations

import time
import logging
import threading
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from medlingo.core.specialist import SpecialistExpert, ExpertFinding
from medlingo.core.router import RoutingDecision
from medlingo.pipeline.spr import SemanticPeerReview, SPRTranscript

logger = logging.getLogger(__name__)

# Target hot-swap latency per §V-D
_HOT_SWAP_TARGET_MS = 150.0


class ParallelExpertExecutor:
    """
    Manages two persistent specialist model instances and orchestrates
    domain-specific adapter swapping, parallel inference, and SPR.

    Memory note: Two 1B NF4 instances ≈ 2.4 GB combined (§V-B).

    Parameters
    ----------
    adapter_registry : dict
        Maps domain name → local path to trained LoRA adapter.
    base_model_path : str, optional
        Shared Llama-3.2-1B base weights path (or HF Hub ID).
    device : str
        GPU device string.
    spr_max_rounds : int
        Passed through to the SemanticPeerReview orchestrator.
    """

    def __init__(
        self,
        adapter_registry: Dict[str, Optional[str]],
        base_model_path: Optional[str] = None,
        device: str = "cuda",
        spr_max_rounds: int = 2,
    ) -> None:
        self._registry = adapter_registry
        self._base_path = base_model_path
        self._device = device

        # Two persistent expert instances (Instance 1 and Instance 2)
        self._instance_1 = SpecialistExpert(
            domain="__unloaded__",
            base_model_path=base_model_path,
            device=device,
            instance_id=1,
        )
        self._instance_2 = SpecialistExpert(
            domain="__unloaded__",
            base_model_path=base_model_path,
            device=device,
            instance_id=2,
        )
        self._spr = SemanticPeerReview(max_rounds=spr_max_rounds)
        self._swap_times: List[float] = []

    # ------------------------------------------------------------------
    # Main execution entry point
    # ------------------------------------------------------------------

    def execute(
        self,
        routing: RoutingDecision,
        query: str,
        visual_description: Optional[str] = None,
        visual_tokens=None,
    ) -> Tuple[SPRTranscript, List[ExpertFinding]]:
        """
        Full parallel expert execution cycle:
          1. Hot-swap LoRA adapters for the two routed domains.
          2. Run initial analysis in parallel (threads).
          3. Execute SPR loop.
          4. Return refined findings + full transcript.

        Parameters
        ----------
        routing : RoutingDecision
            Output of the Semantic Triage Router.
        query : str
            Clinical question.
        visual_description : str, optional
            Text description of image features.
        visual_tokens : torch.Tensor, optional
            PathwayA-projected visual tokens (B, 2048).

        Returns
        -------
        SPRTranscript, list of ExpertFinding (refined)
        """
        domain_a, domain_b = routing.top_domains[:2]
        logger.info(
            "Parallel execution: Instance 1 → %s | Instance 2 → %s",
            domain_a, domain_b,
        )

        # Hot-swap adapters
        self._hot_swap(self._instance_1, domain_a)
        self._hot_swap(self._instance_2, domain_b)

        # Parallel initial inference
        finding_a, finding_b = self._parallel_analyze(
            query, visual_description, visual_tokens
        )

        logger.info(
            "Initial findings — %s: U=%.3f | %s: U=%.3f",
            domain_a, finding_a.uncertainty_score,
            domain_b, finding_b.uncertainty_score,
        )

        # SPR loop
        transcript = self._spr.run(
            self._instance_1, self._instance_2,
            finding_a, finding_b,
        )

        refined_findings = [
            transcript.final_finding_a,
            transcript.final_finding_b,
        ]

        return transcript, refined_findings

    # ------------------------------------------------------------------
    # Hot-swap helper
    # ------------------------------------------------------------------

    def _hot_swap(self, instance: SpecialistExpert, domain: str) -> None:
        t0 = time.perf_counter()
        adapter_path = self._registry.get(domain)

        instance.domain = domain
        instance.unload_adapter()
        instance.load(adapter_path)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._swap_times.append(elapsed_ms)

        if elapsed_ms > _HOT_SWAP_TARGET_MS:
            logger.warning(
                "LoRA hot-swap for '%s' took %.0f ms (target: %d ms).",
                domain, elapsed_ms, _HOT_SWAP_TARGET_MS,
            )
        else:
            logger.debug("LoRA hot-swap '%s': %.0f ms ✓", domain, elapsed_ms)

    # ------------------------------------------------------------------
    # Parallel inference
    # ------------------------------------------------------------------

    def _parallel_analyze(
        self,
        query: str,
        visual_description: Optional[str],
        visual_tokens,
    ) -> Tuple[ExpertFinding, ExpertFinding]:
        results: Dict[int, ExpertFinding] = {}

        def _run(instance: SpecialistExpert, idx: int):
            results[idx] = instance.analyze(
                query=query,
                visual_tokens=visual_tokens,
                image_description=visual_description,
            )

        # Run on two threads — both reside on the same GPU (as per paper design)
        t1 = threading.Thread(target=_run, args=(self._instance_1, 1))
        t2 = threading.Thread(target=_run, args=(self._instance_2, 2))

        t1.start(); t2.start()
        t1.join();  t2.join()

        return results[1], results[2]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def average_swap_latency_ms(self) -> float:
        return sum(self._swap_times) / len(self._swap_times) if self._swap_times else 0.0
