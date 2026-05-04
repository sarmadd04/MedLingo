"""
MedLingo — Resource-Efficient Hierarchical Multi-Agent Intelligence
for Autonomous Clinical and Robotic Diagnostics.

A hierarchical multi-agent framework for resource-efficient medical
image-language diagnostics within a 16 GB VRAM budget.

Components:
  - Frozen BioMedCLIP vision encoder
  - Dual-path linear projection bridges (Pathway A: 768→2048, Pathway B: 768→4096)
  - Semantic Triage Router (Llama-3.2-1B NF4, top-2 domain selection)
  - Six domain LoRA specialists (Llama-3.2-1B NF4) with Parallel Hot-Swapping
  - Semantic Peer Review (SPR) loop with entropy-based uncertainty scoring
  - DPO-aligned Chief Adjudicator (Llama-3.1-8B NF4) for grounded verdicts

Paper:
  Sultan et al., "MedLingo: Resource-Efficient Hierarchical Multi-Agent
  Intelligence for Autonomous Clinical and Robotic Diagnostics", ICRAI 2026.

Quick start:
  >>> from medlingo import MedLingoEngine, DiagnosticRequest
  >>> engine = MedLingoEngine()
  >>> engine.initialize()
  >>> response = engine.diagnose(DiagnosticRequest(
  ...     query="Is there evidence of pneumothorax?",
  ...     image_path="chest_xray.jpg",
  ... ))
  >>> print(response)
"""

__version__ = "1.0.0"
__authors__ = [
    "Sarmad Sultan",
    "Muhammad Sami Ullah",
    "Ahmad Jan",
    "Muhammad Dawood Rizwan",
    "Muhammad Naseer Bajwa",
    "Muhammad Moazam Fraz",
]
__institution__ = "National University of Sciences and Technology (NUST)"

from medlingo.inference.engine import MedLingoEngine, DiagnosticRequest, DiagnosticResponse
from medlingo.pipeline.medlingo_pipeline import MedLingoPipeline
from medlingo.core.adjudicator import AdjudicationVerdict
from medlingo.utils.logging_utils import setup_logging

__all__ = [
    "MedLingoEngine",
    "DiagnosticRequest",
    "DiagnosticResponse",
    "MedLingoPipeline",
    "AdjudicationVerdict",
    "setup_logging",
]
