"""
medlingo/core/uncertainty.py

Entropy-based uncertainty scoring for MedLingo specialist outputs.

Implements two uncertainty metrics from the paper:

  Eq. 3 — Basic Shannon entropy over class probabilities:
      H(p) = -Σ p_i · log(p_i)

  Eq. 5 — Normalized token-level sequence entropy (used in §III-C):
      U(y) = -(1 / L·log|V|) · Σ_l Σ_v P(y_l=v) · log P(y_l=v)

Both metrics range in [0, 1] after normalization. Higher values signal
lower confidence and trigger closer scrutiny by the Chief Adjudicator.
"""

from __future__ import annotations

import math
import torch
import logging
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)

_EPS = 1e-9   # Numerical stability floor


def shannon_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute Shannon entropy per Eq. 3.

    Parameters
    ----------
    probs : torch.Tensor
        Shape (..., C) — softmax probability distribution over C classes.

    Returns
    -------
    torch.Tensor
        Shape (...,) — scalar entropy per sample in the batch.
    """
    probs = probs.clamp(min=_EPS)
    return -(probs * probs.log()).sum(dim=-1)


def normalized_sequence_entropy(
    logits_sequence: List[torch.Tensor],
    vocab_size: int,
) -> float:
    """
    Compute per-sequence normalized entropy per Eq. 5.

    Used to produce the U(y) score that weights expert testimony
    during adjudication.

    Parameters
    ----------
    logits_sequence : list of torch.Tensor
        Each element is shape (V,) — raw logits at each generated token.
    vocab_size : int
        |V| — vocabulary size used for normalization.

    Returns
    -------
    float
        Uncertainty score U ∈ [0, 1]. Higher means less confident.
    """
    if not logits_sequence:
        return 1.0                                # Maximum uncertainty if empty

    seq_len = len(logits_sequence)
    log_norm = math.log(vocab_size + _EPS)
    total_entropy = 0.0

    for logits in logits_sequence:
        probs = torch.softmax(logits.float(), dim=-1)
        token_entropy = -(probs * (probs + _EPS).log()).sum().item()
        total_entropy += token_entropy

    # Normalize by L · log|V| as in Eq. 5
    return total_entropy / (seq_len * log_norm)


class UncertaintyScorer:
    """
    Stateful scorer that processes model generation outputs and returns
    calibrated uncertainty scores for use in the SPR and adjudication phases.
    """

    def __init__(self, vocab_size: int = 32000) -> None:
        self.vocab_size = vocab_size
        self._score_history: List[float] = []

    def score_from_generation(
        self,
        scores: tuple[torch.Tensor, ...],
    ) -> float:
        """
        Compute uncertainty from a HuggingFace `generate()` scores tuple.

        Parameters
        ----------
        scores : tuple of torch.Tensor
            Each element is shape (B, V) — the logit distributions returned
            when `output_scores=True` is passed to `model.generate()`.

        Returns
        -------
        float
            Normalized sequence entropy U ∈ [0, 1].
        """
        # Use the first batch element
        logits_list = [s[0].detach().cpu() for s in scores]
        score = normalized_sequence_entropy(logits_list, self.vocab_size)
        self._score_history.append(score)
        logger.debug("Uncertainty score: %.4f", score)
        return score

    def score_from_logits(self, logits: torch.Tensor) -> float:
        """
        Quick single-tensor scoring for classification heads.

        Parameters
        ----------
        logits : torch.Tensor
            Shape (V,) or (1, V).
        """
        if logits.dim() == 2:
            logits = logits[0]
        probs = torch.softmax(logits.float(), dim=-1)
        h = shannon_entropy(probs).item()
        normalized = h / math.log(self.vocab_size + _EPS)
        self._score_history.append(normalized)
        return normalized

    def calibrate_confidence(self, score: float) -> str:
        """Map numerical uncertainty to a human-readable confidence tier."""
        if score < 0.20:
            return "HIGH"
        elif score < 0.45:
            return "MODERATE"
        elif score < 0.70:
            return "LOW"
        else:
            return "VERY_LOW"

    def running_average(self) -> float:
        if not self._score_history:
            return 0.5
        return sum(self._score_history) / len(self._score_history)

    def reset(self) -> None:
        self._score_history.clear()
