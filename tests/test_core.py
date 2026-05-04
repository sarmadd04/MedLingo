"""
tests/test_core.py

Unit tests for MedLingo core components.

Tests are designed to run without GPU and without any trained weights.
All LLM calls are mocked; only the pure-Python and math-heavy components
(projection bridges, uncertainty scorer, SPR convergence logic) run
against real code.
"""

from __future__ import annotations

import json
import math
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch, PropertyMock

from medlingo.core.projection_bridges import PathwayA, PathwayB, DualPathBridges
from medlingo.core.uncertainty import UncertaintyScorer, shannon_entropy, normalized_sequence_entropy
from medlingo.core.router import SemanticTriageRouter, RoutingDecision, DOMAIN_LIST
from medlingo.core.specialist import ExpertFinding


# ===========================================================================
# Projection Bridges
# ===========================================================================

class TestProjectionBridges:

    def test_pathway_a_output_shape(self):
        bridge = PathwayA()
        x = torch.randn(4, 768)         # Batch of 4, BioMedCLIP output dim
        out = bridge(x)
        assert out.shape == (4, 2048), f"Expected (4, 2048), got {out.shape}"

    def test_pathway_b_output_shape(self):
        bridge = PathwayB()
        x = torch.randn(4, 768)
        out = bridge(x)
        assert out.shape == (4, 4096), f"Expected (4, 4096), got {out.shape}"

    def test_dual_path_forward(self):
        bridges = DualPathBridges()
        x = torch.randn(2, 768)
        z_s, z_o = bridges(x)
        assert z_s.shape == (2, 2048)
        assert z_o.shape == (2, 4096)

    def test_dual_path_freeze(self):
        bridges = DualPathBridges()
        bridges.freeze_pathways()
        for param in bridges.parameters():
            assert not param.requires_grad, "Parameter should be frozen after freeze_pathways()"

    def test_pathway_a_gradient_flow(self):
        bridge = PathwayA()
        x = torch.randn(2, 768, requires_grad=False)
        out = bridge(x)
        loss = out.sum()
        loss.backward()
        # Verify gradients exist in bridge parameters
        for p in bridge.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_pathway_b_gradient_flow(self):
        bridge = PathwayB()
        x = torch.randn(2, 768)
        out = bridge(x)
        out.sum().backward()
        for p in bridge.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_bridges_save_load(self, tmp_path):
        bridges = DualPathBridges()
        # Modify weights to a known state
        with torch.no_grad():
            for p in bridges.parameters():
                nn.init.ones_(p)

        bridges.save(tmp_path)
        assert (tmp_path / "bridge_a.pt").exists()
        assert (tmp_path / "bridge_b.pt").exists()

        bridges2 = DualPathBridges()
        bridges2.load(tmp_path)

        for p1, p2 in zip(bridges.parameters(), bridges2.parameters()):
            assert torch.allclose(p1, p2), "Loaded weights do not match saved weights"


# ===========================================================================
# Uncertainty Scoring
# ===========================================================================

class TestUncertaintyScoring:

    def test_shannon_entropy_uniform(self):
        """Uniform distribution should yield maximum entropy."""
        n = 10
        probs = torch.full((n,), 1.0 / n)
        h = shannon_entropy(probs).item()
        expected = math.log(n)
        assert abs(h - expected) < 1e-5, f"Expected {expected}, got {h}"

    def test_shannon_entropy_degenerate(self):
        """Peaked distribution (one-hot) should yield near-zero entropy."""
        probs = torch.zeros(10)
        probs[0] = 1.0
        h = shannon_entropy(probs).item()
        assert h < 1e-4, f"Expected near-zero entropy, got {h}"

    def test_normalized_entropy_range(self):
        """Normalized sequence entropy must lie in [0, 1]."""
        vocab_size = 32000
        # Random logits simulating a generation step
        logits = [torch.randn(vocab_size) for _ in range(20)]
        score = normalized_sequence_entropy(logits, vocab_size)
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_uncertainty_scorer_empty(self):
        scorer = UncertaintyScorer(vocab_size=32000)
        score = normalized_sequence_entropy([], vocab_size=32000)
        assert score == 1.0, "Empty sequence should return maximum uncertainty"

    def test_calibration_tiers(self):
        scorer = UncertaintyScorer()
        assert scorer.calibrate_confidence(0.10) == "HIGH"
        assert scorer.calibrate_confidence(0.30) == "MODERATE"
        assert scorer.calibrate_confidence(0.60) == "LOW"
        assert scorer.calibrate_confidence(0.85) == "VERY_LOW"

    def test_running_average(self):
        scorer = UncertaintyScorer()
        scorer._score_history = [0.2, 0.4, 0.6]
        assert abs(scorer.running_average() - 0.4) < 1e-6

    def test_reset_clears_history(self):
        scorer = UncertaintyScorer()
        scorer._score_history = [0.1, 0.2, 0.3]
        scorer.reset()
        assert scorer._score_history == []


# ===========================================================================
# Semantic Triage Router (mocked LLM)
# ===========================================================================

class TestSemanticTriageRouter:

    def _mock_router(self) -> SemanticTriageRouter:
        router = SemanticTriageRouter.__new__(SemanticTriageRouter)
        router._model_path = None
        router._device = "cpu"
        router._top_k = 2
        router._threshold = 0.35
        router._model = MagicMock()
        router._tokenizer = MagicMock()
        return router

    def test_parse_valid_json_response(self):
        router = self._mock_router()
        response = json.dumps({
            "top_domains": ["radiology", "pathology"],
            "domain_scores": {
                "pathology": 0.35, "radiology": 0.40, "dermatology": 0.02,
                "cardiology": 0.08, "neurology": 0.05, "general_medicine": 0.10,
            },
            "visual_hint": "Bilateral infiltrates visible.",
        })
        decision = router._parse_routing_response(response, "chest X-ray query")
        assert decision.top_domains == ["radiology", "pathology"]
        assert abs(decision.domain_scores["radiology"] - 0.40) < 1e-6

    def test_keyword_fallback_radiology(self):
        router = self._mock_router()
        decision = router._keyword_fallback("Is there an X-ray showing lung consolidation?")
        assert "radiology" in decision.top_domains

    def test_keyword_fallback_dermatology(self):
        router = self._mock_router()
        decision = router._keyword_fallback("Describe this skin lesion.")
        assert "dermatology" in decision.top_domains

    def test_routing_decision_top_k(self):
        router = self._mock_router()
        decision = router._keyword_fallback("Brain MRI showing white matter changes.")
        assert len(decision.top_domains) == 2

    def test_all_domains_covered_in_keyword_fallback(self):
        router = self._mock_router()
        decision = router._keyword_fallback("General patient clinical review.")
        assert all(d in DOMAIN_LIST for d in decision.top_domains)


# ===========================================================================
# Expert Findings (dataclass contract)
# ===========================================================================

class TestExpertFinding:

    def _make_finding(self, domain="radiology", u=0.25) -> ExpertFinding:
        return ExpertFinding(
            domain=domain,
            report="Right lower lobe opacity consistent with pneumonia.",
            uncertainty_score=u,
            confidence_tier="MODERATE",
            raw_logits_entropy=u,
        )

    def test_to_prompt_str_contains_domain(self):
        f = self._make_finding("radiology")
        s = f.to_prompt_str()
        assert "RADIOLOGY" in s

    def test_to_prompt_str_contains_uncertainty(self):
        f = self._make_finding(u=0.312)
        s = f.to_prompt_str()
        assert "0.312" in s

    def test_refined_flag_defaults_false(self):
        f = self._make_finding()
        assert f.refined is False

    def test_spr_notes_empty_by_default(self):
        f = self._make_finding()
        assert f.spr_notes == ""
