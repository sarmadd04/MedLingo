"""
medlingo/core/router.py

Semantic Triage Router — §III-B.

A quantized Llama-3.2-1B model performs multi-label classification
to identify the top-2 most relevant diagnostic domains for a given
image-query pair. This governs which LoRA specialist adapters are
hot-swapped into the parallel expert instances.

"The router analyzes the semantic embeddings of the text and the
visual modality to perform a multi-label classification.
It identifies the two diagnostic domains with the highest relevance."
"""

from __future__ import annotations

import json
import torch
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

DOMAIN_LIST = [
    "pathology",
    "radiology",
    "dermatology",
    "cardiology",
    "neurology",
    "general_medicine",
]

DOMAIN_TO_IDX: Dict[str, int] = {d: i for i, d in enumerate(DOMAIN_LIST)}


@dataclass
class RoutingDecision:
    top_domains: List[str]                # e.g. ["radiology", "pathology"]
    domain_scores: Dict[str, float]       # Softmax-normalized relevance scores
    visual_hint: Optional[str] = None     # Short description of salient features
    routing_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "top_domains": self.top_domains,
            "domain_scores": self.domain_scores,
            "routing_confidence": self.routing_confidence,
        }


_ROUTER_SYSTEM_PROMPT = """\
You are MedLingo's Semantic Triage Router. Your task is to analyze a medical \
query and image description, then classify which TWO of the following diagnostic \
domains are most relevant:

Domains: pathology, radiology, dermatology, cardiology, neurology, general_medicine

Respond ONLY with a JSON object in this exact format:
{
  "top_domains": ["<domain1>", "<domain2>"],
  "domain_scores": {
    "pathology": <float 0-1>,
    "radiology": <float 0-1>,
    "dermatology": <float 0-1>,
    "cardiology": <float 0-1>,
    "neurology": <float 0-1>,
    "general_medicine": <float 0-1>
  },
  "visual_hint": "<one-sentence description of key visual features>"
}
All six domain_scores must sum to 1.0.
"""


class SemanticTriageRouter:
    """
    Lightweight NF4-quantized Llama-3.2-1B router for domain selection.

    The router accepts:
      - A text query (the clinical question)
      - Optional visual tokens z_s from PathwayA (prepended as soft prompts)

    It returns a RoutingDecision indicating the top-2 activated domains.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        top_k: int = 2,
        confidence_threshold: float = 0.35,
    ) -> None:
        self._model_path = model_path
        self._device = device
        self._top_k = top_k
        self._threshold = confidence_threshold
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the router model (call once before inference)."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from medlingo.utils.quantization import nf4_config

        model_id = self._model_path or "meta-llama/Llama-3.2-1B-Instruct"
        logger.info("Loading router from: %s", model_id)

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=nf4_config(),
            device_map=self._device,
            trust_remote_code=True,
        )
        self._model.eval()
        logger.info("Router loaded ✓")

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    @torch.no_grad()
    def route(
        self,
        query: str,
        visual_context: Optional[str] = None,
        visual_tokens: Optional[torch.Tensor] = None,
    ) -> RoutingDecision:
        """
        Determine which two specialist domains handle this query.

        Parameters
        ----------
        query : str
            The clinical question from the user.
        visual_context : str, optional
            A textual description of the image (used when visual_tokens not provided).
        visual_tokens : torch.Tensor, optional
            Shape (B, 2048) — PathwayA-projected tokens for multimodal routing.

        Returns
        -------
        RoutingDecision
        """
        if self._model is None:
            self.load()

        user_content = query
        if visual_context:
            user_content = f"[Image context: {visual_context}]\n\nQuery: {query}"

        messages = [
            {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        input_ids = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self._device)

        outputs = self._model.generate(
            input_ids,
            max_new_tokens=256,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        generated = outputs[0][input_ids.shape[1]:]
        response_text = self._tokenizer.decode(generated, skip_special_tokens=True)

        return self._parse_routing_response(response_text, query)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_routing_response(
        self, response: str, query: str
    ) -> RoutingDecision:
        try:
            # Extract JSON block
            start = response.find("{")
            end   = response.rfind("}") + 1
            data  = json.loads(response[start:end])

            top_domains = data.get("top_domains", [])[:self._top_k]
            domain_scores = data.get("domain_scores", {})
            visual_hint   = data.get("visual_hint", "")

            # Validate domains
            top_domains = [d for d in top_domains if d in DOMAIN_TO_IDX]
            if len(top_domains) < self._top_k:
                top_domains = self._fallback_routing(domain_scores)

            confidence = sum(domain_scores.get(d, 0.0) for d in top_domains)

            return RoutingDecision(
                top_domains=top_domains,
                domain_scores=domain_scores,
                visual_hint=visual_hint,
                routing_confidence=confidence,
            )

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Router parse failed (%s) — using keyword fallback.", exc)
            return self._keyword_fallback(query)

    def _fallback_routing(self, scores: Dict[str, float]) -> List[str]:
        """Sort by score and return top-k domain names."""
        sorted_domains = sorted(scores, key=lambda d: scores.get(d, 0), reverse=True)
        return sorted_domains[:self._top_k]

    def _keyword_fallback(self, query: str) -> RoutingDecision:
        """Deterministic keyword-based routing when the LLM parse fails."""
        q = query.lower()
        keyword_map = {
            "pathology":        ["biopsy", "histology", "patholog", "cell", "tissue"],
            "radiology":        ["xray", "x-ray", "ct scan", "mri", "radiolog", "scan"],
            "dermatology":      ["skin", "lesion", "rash", "melanoma", "dermato"],
            "cardiology":       ["heart", "cardiac", "ecg", "echocardiogram", "cardio"],
            "neurology":        ["brain", "neuro", "stroke", "spine", "alzheimer"],
            "general_medicine": ["symptom", "diagnosis", "patient", "treat", "general"],
        }
        scores = {d: 0.0 for d in DOMAIN_LIST}
        for domain, keywords in keyword_map.items():
            scores[domain] = sum(1.0 for kw in keywords if kw in q)
        total = sum(scores.values()) or 1.0
        scores = {d: v / total for d, v in scores.items()}
        top_domains = sorted(scores, key=lambda d: scores[d], reverse=True)[:self._top_k]
        return RoutingDecision(
            top_domains=top_domains,
            domain_scores=scores,
            routing_confidence=sum(scores[d] for d in top_domains),
        )
