"""
tests/test_pipeline.py

Integration tests for the MedLingo pipeline.

All LLM inference calls are mocked so these tests run on CPU without
any downloaded weights. They validate that:
  - The pipeline wires components together correctly
  - Data flows through each stage without shape errors
  - The SPR transcript is properly constructed
  - The adjudicator verdict is correctly structured
"""

from __future__ import annotations

import torch
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from medlingo.core.specialist import ExpertFinding
from medlingo.core.adjudicator import AdjudicationVerdict
from medlingo.core.router import RoutingDecision
from medlingo.pipeline.spr import SemanticPeerReview, SPRTranscript
from medlingo.inference.engine import DiagnosticRequest, DiagnosticResponse


# ---------------------------------------------------------------------------
# Helpers — deterministic mock components
# ---------------------------------------------------------------------------

def _make_finding(domain: str, u: float = 0.25, refined: bool = False) -> ExpertFinding:
    return ExpertFinding(
        domain=domain,
        report=f"[{domain}] Clinical finding with uncertainty {u:.2f}.",
        uncertainty_score=u,
        confidence_tier="MODERATE",
        raw_logits_entropy=u,
        refined=refined,
    )


def _make_routing(domains=None) -> RoutingDecision:
    domains = domains or ["radiology", "pathology"]
    return RoutingDecision(
        top_domains=domains,
        domain_scores={d: 0.5 / len(domains) for d in domains},
        routing_confidence=0.65,
    )


# ---------------------------------------------------------------------------
# SPR tests
# ---------------------------------------------------------------------------

class TestSemanticPeerReview:

    def _make_mock_expert(self, domain: str, u: float = 0.25) -> MagicMock:
        expert = MagicMock()
        expert.domain = domain
        expert.refine_with_peer_feedback.return_value = _make_finding(
            domain, u=u * 0.9, refined=True
        )
        return expert

    def test_spr_runs_correct_number_of_rounds(self):
        spr = SemanticPeerReview(max_rounds=2, convergence_threshold=0.0)
        expert_a = self._make_mock_expert("radiology",  u=0.30)
        expert_b = self._make_mock_expert("pathology",  u=0.40)
        fa = _make_finding("radiology",  u=0.30)
        fb = _make_finding("pathology",  u=0.40)

        transcript = spr.run(expert_a, expert_b, fa, fb)
        assert transcript.rounds_completed == 2

    def test_spr_early_convergence(self):
        """When delta < threshold from round 1, SPR should stop early."""
        spr = SemanticPeerReview(max_rounds=3, convergence_threshold=0.99)
        expert_a = self._make_mock_expert("radiology", u=0.30)
        expert_b = self._make_mock_expert("pathology", u=0.30)
        fa = _make_finding("radiology", u=0.30)
        fb = _make_finding("pathology", u=0.30)

        transcript = spr.run(expert_a, expert_b, fa, fb)
        assert transcript.converged is True
        assert transcript.rounds_completed == 1

    def test_spr_transcript_contains_both_domains(self):
        spr = SemanticPeerReview(max_rounds=1)
        expert_a = self._make_mock_expert("radiology")
        expert_b = self._make_mock_expert("pathology")
        fa = _make_finding("radiology")
        fb = _make_finding("pathology")

        transcript = spr.run(expert_a, expert_b, fa, fb)
        text = transcript.to_text()
        assert "RADIOLOGY" in text
        assert "PATHOLOGY" in text

    def test_spr_final_findings_are_refined(self):
        spr = SemanticPeerReview(max_rounds=2)
        expert_a = self._make_mock_expert("radiology")
        expert_b = self._make_mock_expert("pathology")
        fa = _make_finding("radiology")
        fb = _make_finding("pathology")

        transcript = spr.run(expert_a, expert_b, fa, fb)
        assert transcript.final_finding_a is not None
        assert transcript.final_finding_b is not None

    def test_spr_transcript_text_round_structure(self):
        spr = SemanticPeerReview(max_rounds=2, convergence_threshold=0.0)
        expert_a = self._make_mock_expert("cardiology")
        expert_b = self._make_mock_expert("neurology")
        fa = _make_finding("cardiology")
        fb = _make_finding("neurology")

        transcript = spr.run(expert_a, expert_b, fa, fb)
        text = transcript.to_text()
        assert "Round 1" in text
        assert "Round 2" in text


# ---------------------------------------------------------------------------
# Adjudication verdict tests
# ---------------------------------------------------------------------------

class TestAdjudicationVerdict:

    def _make_verdict(self) -> AdjudicationVerdict:
        return AdjudicationVerdict(
            verdict="Right lower lobe pneumonia confirmed.",
            grounding_summary="Opacity visible in right lower zone on PA chest X-ray.",
            conflict_resolution="Radiology and pathology experts agreed on consolidation.",
            expert_weights={"radiology": 0.60, "pathology": 0.40},
            overall_confidence="HIGH",
            latency_ms=3800.0,
        )

    def test_format_response_contains_verdict(self):
        v = self._make_verdict()
        s = v.format_response()
        assert "Right lower lobe pneumonia confirmed." in s

    def test_format_response_contains_confidence(self):
        v = self._make_verdict()
        s = v.format_response()
        assert "HIGH" in s

    def test_format_response_contains_grounding(self):
        v = self._make_verdict()
        s = v.format_response()
        assert "Opacity visible" in s


# ---------------------------------------------------------------------------
# DiagnosticRequest / DiagnosticResponse
# ---------------------------------------------------------------------------

class TestDiagnosticDataclasses:

    def test_request_defaults(self):
        req = DiagnosticRequest(query="Is there consolidation?")
        assert req.image_path is None
        assert req.request_id is None
        assert req.metadata == {}

    def test_response_to_dict_keys(self):
        verdict = AdjudicationVerdict(
            verdict="Test verdict.",
            grounding_summary="Visual anchor.",
            conflict_resolution=None,
            expert_weights={"radiology": 1.0},
            overall_confidence="MODERATE",
        )
        resp = DiagnosticResponse(
            request_id="r1",
            verdict=verdict.verdict,
            visual_grounding=verdict.grounding_summary,
            conflict_resolution=verdict.conflict_resolution,
            expert_domains=["radiology"],
            expert_weights={"radiology": 1.0},
            overall_confidence="MODERATE",
            latency_ms=1234.5,
            raw_verdict=verdict,
        )
        d = resp.to_dict()
        assert "verdict" in d
        assert "latency_ms" in d
        assert "overall_confidence" in d
        assert d["latency_ms"] == 1234.5
