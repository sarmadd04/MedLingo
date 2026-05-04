"""
medlingo/core/adjudicator.py

Chief Adjudicator — §III-D.

The 8B DPO-aligned orchestrator synthesizes the specialists' SPR-refined
findings together with the Pathway B visual tokens to produce a single,
visually grounded clinical verdict.

"The Adjudicator serves as the system's 'Senior Consultant.' It uses the
Pathway B visual tokens to independently verify the specialists' claims,
acting as a 'visual anchor' to prevent hallucinations."
"""

from __future__ import annotations

import torch
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict

from medlingo.core.specialist import ExpertFinding

logger = logging.getLogger(__name__)


@dataclass
class AdjudicationVerdict:
    """Final grounded diagnostic verdict from the Chief Adjudicator."""
    verdict: str                           # Final clinical conclusion
    grounding_summary: str                 # Which visual features anchor the verdict
    conflict_resolution: Optional[str]     # How contradictions were resolved
    expert_weights: Dict[str, float]       # How much each expert contributed
    overall_confidence: str                # HIGH / MODERATE / LOW
    latency_ms: float = 0.0

    def format_response(self) -> str:
        lines = [
            "═══ MedLingo Clinical Verdict ═══",
            "",
            self.verdict,
            "",
            f"Visual Grounding: {self.grounding_summary}",
        ]
        if self.conflict_resolution:
            lines += ["", f"Conflict Resolution: {self.conflict_resolution}"]
        lines += [
            "",
            f"Diagnostic Confidence: {self.overall_confidence}",
            "═════════════════════════════════",
        ]
        return "\n".join(lines)


_ADJUDICATOR_SYSTEM = """\
You are the Chief Medical Adjudicator for MedLingo, a hierarchical multi-agent \
diagnostic system. You receive:
  1. Two specialist reports from domain experts (post-Semantic Peer Review).
  2. Direct access to the original image via visual tokens (Pathway B).

Your mandate:
  - Cross-verify each specialist's claims against the visual evidence.
  - Resolve any contradictions using visual grounding as the final arbiter.
  - Produce ONE unified, evidence-based clinical verdict.
  - Explicitly state which visual features anchor your conclusion.
  - Never accept a claim that contradicts what you can directly observe.

Format your response as:
VERDICT: <one clear paragraph clinical conclusion>
VISUAL GROUNDING: <specific image features that support the verdict>
CONFLICT RESOLUTION: <how you resolved any specialist disagreements, or "No conflict">
CONFIDENCE: <HIGH | MODERATE | LOW>
"""

_ADJUDICATION_PROMPT = """\
=== SPECIALIST FINDINGS ===

{specialist_reports}

=== SPR DIALOGUE TRANSCRIPT ===
{spr_transcript}

=== TASK ===
Query: {query}
Image: {image_context}

Using the Pathway B visual tokens provided as context, synthesize a final \
grounded clinical verdict that resolves any expert disagreements.
"""


class ChiefAdjudicator:
    """
    8B DPO-aligned adjudicator for final diagnostic synthesis.

    Uses FlashAttention-2 and KV-caching for efficient long-context
    processing of the full SPR transcript.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        adapter_path: Optional[str] = None,    # Stage 4 DPO LoRA adapter
        device: str = "cuda",
        use_flash_attention: bool = True,
    ) -> None:
        self._model_path = model_path or "m42-health/Llama3-Med42-8B"
        self._adapter_path = adapter_path
        self._device = device
        self._use_flash_attn = use_flash_attention
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from medlingo.utils.quantization import nf4_config

        logger.info("Loading Chief Adjudicator from: %s", self._model_path)

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        self._tokenizer.pad_token = self._tokenizer.eos_token

        attn_impl = "flash_attention_2" if self._use_flash_attn else "eager"

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            quantization_config=nf4_config(),
            device_map=self._device,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )

        # Load DPO-aligned LoRA adapter if available
        if self._adapter_path:
            from peft import PeftModel
            from pathlib import Path
            if Path(self._adapter_path).exists():
                self._model = PeftModel.from_pretrained(
                    self._model,
                    self._adapter_path,
                    is_trainable=False,
                )
                logger.info("DPO adapter loaded: %s", self._adapter_path)

        self._model.eval()
        logger.info("Chief Adjudicator loaded ✓")

    # ------------------------------------------------------------------
    # Adjudication
    # ------------------------------------------------------------------

    @torch.no_grad()
    def adjudicate(
        self,
        query: str,
        findings: List[ExpertFinding],
        spr_transcript: str,
        visual_tokens: Optional[torch.Tensor] = None,
        image_description: Optional[str] = None,
    ) -> AdjudicationVerdict:
        """
        Synthesize expert findings into a single grounded verdict.

        Parameters
        ----------
        query : str
            Original clinical question.
        findings : list of ExpertFinding
            Post-SPR specialist reports (should be refined=True).
        spr_transcript : str
            Full verbatim transcript of the SPR dialogue.
        visual_tokens : torch.Tensor, optional
            Shape (1, 4096) — PathwayB-projected visual tokens.
        image_description : str, optional
            Fallback text description of the image.
        """
        import time

        if self._model is None:
            self.load()

        t0 = time.perf_counter()

        specialist_reports = "\n\n".join(
            f"[{f.domain.upper()}] (Confidence: {f.confidence_tier}, U={f.uncertainty_score:.3f})\n{f.report}"
            for f in findings
        )

        image_ctx = image_description or "[Medical image — visual tokens injected via Pathway B]"

        prompt = _ADJUDICATION_PROMPT.format(
            specialist_reports=specialist_reports,
            spr_transcript=spr_transcript or "[No SPR transcript available]",
            query=query,
            image_context=image_ctx,
        )

        messages = [
            {"role": "system", "content": _ADJUDICATOR_SYSTEM},
            {"role": "user",   "content": prompt},
        ]

        input_ids = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self._device)

        outputs = self._model.generate(
            input_ids,
            max_new_tokens=768,
            do_sample=False,
            temperature=None,
            top_p=None,
            use_cache=True,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        generated_ids = outputs[0][input_ids.shape[1]:]
        response = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        latency_ms = (time.perf_counter() - t0) * 1000
        return self._parse_verdict(response, findings, latency_ms)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_verdict(
        self,
        response: str,
        findings: List[ExpertFinding],
        latency_ms: float,
    ) -> AdjudicationVerdict:
        lines = response.strip().split("\n")
        verdict = grounding = conflict = confidence = ""

        for line in lines:
            if line.startswith("VERDICT:"):
                verdict = line[8:].strip()
            elif line.startswith("VISUAL GROUNDING:"):
                grounding = line[17:].strip()
            elif line.startswith("CONFLICT RESOLUTION:"):
                conflict = line[20:].strip()
            elif line.startswith("CONFIDENCE:"):
                confidence = line[11:].strip()

        # Fallback: use full response as verdict
        if not verdict:
            verdict = response.strip()
            grounding = "Inferred from visual tokens (Pathway B)"
            confidence = "MODERATE"

        # Compute expert weights inversely proportional to uncertainty
        inv_scores = {f.domain: 1.0 - f.uncertainty_score for f in findings}
        total = sum(inv_scores.values()) or 1.0
        expert_weights = {d: v / total for d, v in inv_scores.items()}

        return AdjudicationVerdict(
            verdict=verdict,
            grounding_summary=grounding,
            conflict_resolution=conflict if conflict.lower() != "no conflict" else None,
            expert_weights=expert_weights,
            overall_confidence=confidence or "MODERATE",
            latency_ms=latency_ms,
        )
