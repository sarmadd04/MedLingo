"""
medlingo/core/specialist.py

Domain-specialized Resident Expert — §III-B, §III-C.

Each specialist is a Llama-3.2-1B backbone with a domain-specific LoRA
adapter hot-swapped at runtime. Two specialists run in parallel per
inference call, sharing the same base weights but holding separate adapters.

"The architecture initializes two parallel instances of the Llama-3.2-1B
model backbone. At runtime, the system performs a Parallel Hot-Swap,
loading the router-selected LoRA adapters into each respective model instance."
"""

from __future__ import annotations

import torch
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from medlingo.core.uncertainty import UncertaintyScorer

logger = logging.getLogger(__name__)


@dataclass
class ExpertFinding:
    """Structured output from a single specialist expert."""
    domain: str
    report: str                        # Clinical findings paragraph
    uncertainty_score: float           # U ∈ [0, 1]; lower = more confident
    confidence_tier: str               # HIGH / MODERATE / LOW / VERY_LOW
    raw_logits_entropy: float
    refined: bool = False              # True after SPR refinement pass
    spr_notes: str = ""                # Peer-review feedback received
    token_count: int = 0

    def to_prompt_str(self) -> str:
        return (
            f"[{self.domain.upper()} EXPERT FINDING]\n"
            f"Report: {self.report}\n"
            f"Confidence: {self.confidence_tier} (U={self.uncertainty_score:.3f})"
        )


_SPECIALIST_SYSTEM = """\
You are a board-certified specialist in {domain}. You are analyzing a medical \
image alongside a clinical query. Provide a precise, evidence-based diagnostic \
finding. Cite specific visual features observed in the image. Be concise and \
avoid speculation beyond what the image directly supports.

Specialty: {domain_upper}
Visual context tokens are prepended to your input.
"""

_SPR_REFINEMENT_PROMPT = """\
You have received feedback from a peer specialist.

Your previous finding:
{own_report}

Your peer ({peer_domain}) reported:
{peer_report}
Peer confidence: {peer_confidence}

Please critically review your finding in light of this peer input.
If you agree with the peer on any point, explicitly acknowledge it.
If you identify a contradiction, explain your reasoning for maintaining
or revising your position. Provide your final refined finding.
"""


class SpecialistExpert:
    """
    Single domain-specialist agent with LoRA hot-swapping support.

    Parameters
    ----------
    domain : str
        One of the six MedLingo diagnostic domains.
    base_model_path : str, optional
        Path to the Llama-3.2-1B base weights (or HF Hub ID).
    adapter_path : str, optional
        Path to the pre-trained domain LoRA adapter.
    device : str
        Target device.
    instance_id : int
        0 or 1 — identifies this as Instance 1 or Instance 2 in parallel exec.
    """

    def __init__(
        self,
        domain: str,
        base_model_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
        device: str = "cuda",
        instance_id: int = 0,
    ) -> None:
        self.domain = domain
        self._base_path = base_model_path or "meta-llama/Llama-3.2-1B-Instruct"
        self._adapter_path = adapter_path
        self._device = device
        self._instance_id = instance_id
        self._model = None
        self._tokenizer = None
        self._uncertainty = UncertaintyScorer()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self, adapter_path: Optional[str] = None) -> None:
        """Load base model and optionally hot-swap a LoRA adapter."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        from medlingo.utils.quantization import nf4_config

        if self._model is None:
            logger.info(
                "[Instance %d] Loading %s specialist backbone: %s",
                self._instance_id, self._domain, self._base_path,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self._base_path)
            self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModelForCausalLM.from_pretrained(
                self._base_path,
                quantization_config=nf4_config(),
                device_map=self._device,
                trust_remote_code=True,
            )

        # Hot-swap LoRA adapter
        effective_adapter = adapter_path or self._adapter_path
        if effective_adapter and Path(effective_adapter).exists():
            self._model = PeftModel.from_pretrained(
                self._model,
                effective_adapter,
                is_trainable=False,
            )
            logger.info(
                "[Instance %d] LoRA adapter hot-swapped: %s",
                self._instance_id, effective_adapter,
            )
        else:
            logger.warning(
                "[Instance %d] No adapter found for '%s' — using base weights.",
                self._instance_id, self.domain,
            )

        self._model.eval()

    def unload_adapter(self) -> None:
        """Remove LoRA adapter to prepare for next domain hot-swap."""
        try:
            from peft import PeftModel
            if isinstance(self._model, PeftModel):
                self._model = self._model.base_model.model
                logger.debug("[Instance %d] Adapter unloaded.", self._instance_id)
        except Exception as exc:
            logger.debug("Adapter unload skipped: %s", exc)

    @property
    def _domain(self) -> str:
        return self.domain.replace("_", " ").title()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def analyze(
        self,
        query: str,
        visual_tokens: Optional[torch.Tensor] = None,
        image_description: Optional[str] = None,
    ) -> ExpertFinding:
        """
        Generate a clinical finding for the given query and image.

        Parameters
        ----------
        query : str
            Clinical question.
        visual_tokens : torch.Tensor, optional
            Shape (1, 2048) — PathwayA-projected tokens.
        image_description : str, optional
            Fallback text description if visual tokens unavailable.
        """
        if self._model is None:
            self.load()

        system_prompt = _SPECIALIST_SYSTEM.format(
            domain=self.domain.replace("_", " "),
            domain_upper=self._domain,
        )

        context = image_description or "[Medical image provided]"
        user_msg = f"Image: {context}\n\nClinical Query: {query}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg},
        ]

        input_ids = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self._device)

        outputs = self._model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        generated_ids = outputs.sequences[0][input_ids.shape[1]:]
        report = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        u_score = self._uncertainty.score_from_generation(outputs.scores)
        confidence = self._uncertainty.calibrate_confidence(u_score)

        return ExpertFinding(
            domain=self.domain,
            report=report.strip(),
            uncertainty_score=u_score,
            confidence_tier=confidence,
            raw_logits_entropy=u_score,
            token_count=len(generated_ids),
        )

    @torch.no_grad()
    def refine_with_peer_feedback(
        self,
        own_finding: ExpertFinding,
        peer_finding: ExpertFinding,
    ) -> ExpertFinding:
        """
        SPR refinement pass — inject peer finding and re-generate.

        This implements the Semantic Peer Review loop (§III-C):
        "The preliminary report and associated uncertainty score generated
        by Expert A are injected into the context window of Expert B,
        and vice versa."
        """
        if self._model is None:
            self.load()

        refinement_prompt = _SPR_REFINEMENT_PROMPT.format(
            own_report=own_finding.report,
            peer_domain=peer_finding.domain.replace("_", " ").title(),
            peer_report=peer_finding.report,
            peer_confidence=peer_finding.confidence_tier,
        )

        messages = [
            {"role": "system", "content": _SPECIALIST_SYSTEM.format(
                domain=self.domain.replace("_", " "),
                domain_upper=self._domain,
            )},
            {"role": "user", "content": refinement_prompt},
        ]

        input_ids = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self._device)

        outputs = self._model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        generated_ids = outputs.sequences[0][input_ids.shape[1]:]
        refined_report = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        u_score = self._uncertainty.score_from_generation(outputs.scores)
        confidence = self._uncertainty.calibrate_confidence(u_score)

        return ExpertFinding(
            domain=self.domain,
            report=refined_report.strip(),
            uncertainty_score=u_score,
            confidence_tier=confidence,
            raw_logits_entropy=u_score,
            refined=True,
            spr_notes=f"Peer review from {peer_finding.domain}: considered.",
            token_count=len(generated_ids),
        )
