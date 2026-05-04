"""
medlingo/agents/__init__.py

MedLingo Agent Layer.

Exposes the three agent roles described in §III as importable classes:

  TriageAgent     — wraps SemanticTriageRouter behind an agent interface
  ResidentAgent   — wraps SpecialistExpert; owns a domain and an adapter slot
  AdjudicatorAgent— wraps ChiefAdjudicator; synthesises the final verdict

These thin wrappers add identity metadata (agent_id, role) and a unified
`.run(context)` interface so agents can be composed into arbitrary
orchestration graphs without modifying core logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """
    Shared execution context passed between agents in a single diagnostic run.

    Populated incrementally: the Triage agent fills `routing`, the Resident
    agents fill `findings` and `spr_transcript`, and the Adjudicator agent
    fills `verdict`.
    """
    query: str
    image_path: Optional[str] = None
    image_description: Optional[str] = None
    visual_tokens_specialist: Any = None       # z_s — shape (1, 2048)
    visual_tokens_orchestrator: Any = None     # z_o — shape (1, 4096)
    routing: Optional[Any] = None              # RoutingDecision
    findings: list = field(default_factory=list)
    spr_transcript: Optional[Any] = None       # SPRTranscript
    verdict: Optional[Any] = None              # AdjudicationVerdict
    metadata: Dict[str, Any] = field(default_factory=dict)


class TriageAgent:
    """
    Agent wrapper around SemanticTriageRouter.

    Role : semantic routing — determines which two specialist domains
           are most relevant for the incoming query and image.
    """

    role = "triage"

    def __init__(self, router) -> None:
        self._router = router

    def run(self, context: AgentContext) -> AgentContext:
        """Populate context.routing and return updated context."""
        routing = self._router.route(
            query=context.query,
            visual_context=context.image_description,
            visual_tokens=context.visual_tokens_specialist,
        )
        context.routing = routing
        logger.info(
            "[TriageAgent] Routed to: %s (confidence=%.2f)",
            routing.top_domains, routing.routing_confidence,
        )
        return context


class ResidentAgent:
    """
    Agent wrapper around ParallelExpertExecutor.

    Role : parallel domain-specific analysis + SPR.
           Activates two LoRA-specialised model instances and executes
           the full Semantic Peer Review cycle.
    """

    role = "resident"

    def __init__(self, executor) -> None:
        self._executor = executor

    def run(self, context: AgentContext) -> AgentContext:
        """Populate context.findings and context.spr_transcript."""
        if context.routing is None:
            raise RuntimeError("TriageAgent must run before ResidentAgent.")

        transcript, findings = self._executor.execute(
            routing=context.routing,
            query=context.query,
            visual_description=context.image_description,
            visual_tokens=context.visual_tokens_specialist,
        )
        context.findings = findings
        context.spr_transcript = transcript
        logger.info(
            "[ResidentAgent] SPR complete — %d round(s), converged=%s",
            transcript.rounds_completed, transcript.converged,
        )
        return context


class AdjudicatorAgent:
    """
    Agent wrapper around ChiefAdjudicator.

    Role : grounded adjudication — synthesises specialist findings with
           Pathway B visual tokens to produce the final clinical verdict.
    """

    role = "adjudicator"

    def __init__(self, adjudicator) -> None:
        self._adjudicator = adjudicator

    def run(self, context: AgentContext) -> AgentContext:
        """Populate context.verdict with the final AdjudicationVerdict."""
        if not context.findings:
            raise RuntimeError("ResidentAgent must run before AdjudicatorAgent.")

        verdict = self._adjudicator.adjudicate(
            query=context.query,
            findings=context.findings,
            spr_transcript=context.spr_transcript.to_text()
            if context.spr_transcript else "",
            visual_tokens=context.visual_tokens_orchestrator,
            image_description=context.image_description,
        )
        context.verdict = verdict
        logger.info(
            "[AdjudicatorAgent] Verdict issued — confidence=%s, latency=%.0f ms",
            verdict.overall_confidence, verdict.latency_ms,
        )
        return context


__all__ = [
    "AgentContext",
    "TriageAgent",
    "ResidentAgent",
    "AdjudicatorAgent",
]
