"""
medlingo/pipeline/medlingo_pipeline.py

MedLingo Pipeline — top-level orchestrator.

Implements the full inference flow described in §III:
  Image + Query
       ↓
  VisionEncoder  → (768-dim features)
       ↓
  DualPathBridges → z_s (2048) for specialists
                  → z_o (4096) for adjudicator
       ↓
  SemanticTriageRouter → top-2 domains
       ↓
  ParallelExpertExecutor → [Instance1 ↔ Instance2] + SPR
       ↓
  ChiefAdjudicator (DPO-aligned) → grounded verdict
"""

from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Optional, Union, List
from PIL import Image

from medlingo.core.vision_encoder import VisionEncoder
from medlingo.core.projection_bridges import DualPathBridges
from medlingo.core.router import SemanticTriageRouter
from medlingo.core.adjudicator import ChiefAdjudicator, AdjudicationVerdict
from medlingo.core.specialist import ExpertFinding
from medlingo.pipeline.parallel_executor import ParallelExpertExecutor
from medlingo.pipeline.spr import SPRTranscript
from medlingo.utils.memory_manager import get_memory_manager
from medlingo.utils.model_registry import resolve_paths

logger = logging.getLogger(__name__)


class MedLingoPipeline:
    """
    End-to-end MedLingo inference pipeline.

    All models are lazy-loaded on first call to `diagnose()`.
    Component loading order respects the 16 GB VRAM budget.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to inference_config.yaml. Uses package default if not provided.
    device : str
        "cuda", "cpu", or "auto" (auto-selects GPU if available).

    Example
    -------
    >>> pipeline = MedLingoPipeline()
    >>> result = pipeline.diagnose(
    ...     image_path="chest_xray.jpg",
    ...     query="Is there evidence of pneumothorax?"
    ... )
    >>> print(result.format_response())
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
    ) -> None:
        import torch
        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        self._paths = resolve_paths()
        self._cfg = self._load_config(config_path)
        self._mm = get_memory_manager()

        # Components initialized lazily
        self._vision_encoder: Optional[VisionEncoder] = None
        self._bridges: Optional[DualPathBridges] = None
        self._router: Optional[SemanticTriageRouter] = None
        self._executor: Optional[ParallelExpertExecutor] = None
        self._adjudicator: Optional[ChiefAdjudicator] = None

        self._initialized = False

        logger.info(
            "MedLingoPipeline created | device=%s | weights=%s",
            self._device, self._paths.weights_base,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diagnose(
        self,
        image: Optional[Union[str, Path, "Image.Image"]] = None,
        query: str = "",
        image_description: Optional[str] = None,
        verbose: bool = False,
    ) -> AdjudicationVerdict:
        """
        Run the full MedLingo diagnostic pipeline on an image and query.

        Parameters
        ----------
        image : str, Path, or PIL Image, optional
            Medical image to analyze. If None, text-only mode is used.
        query : str
            Clinical question (e.g. "What abnormalities are present?").
        image_description : str, optional
            Manual text description of the image (overrides visual encoding).
        verbose : bool
            Log routing and SPR details to console.

        Returns
        -------
        AdjudicationVerdict
        """
        t_pipeline_start = time.perf_counter()

        if not self._initialized:
            self._initialize_components()

        # Step 1: Visual encoding
        z_s, z_o, img_desc = self._encode_image(image, image_description)

        # Step 2: Semantic triage routing
        routing = self._router.route(
            query=query,
            visual_context=img_desc,
            visual_tokens=z_s,
        )

        if verbose:
            logger.info(
                "Routing decision → %s (confidence=%.2f)",
                routing.top_domains, routing.routing_confidence,
            )

        # Step 3: Parallel expert execution + SPR
        transcript, refined_findings = self._executor.execute(
            routing=routing,
            query=query,
            visual_description=img_desc,
            visual_tokens=z_s,
        )

        if verbose:
            logger.info("SPR completed — %d round(s), converged=%s",
                        transcript.rounds_completed, transcript.converged)

        # Step 4: Grounded adjudication
        verdict = self._adjudicator.adjudicate(
            query=query,
            findings=refined_findings,
            spr_transcript=transcript.to_text(),
            visual_tokens=z_o,
            image_description=img_desc,
        )

        total_ms = (time.perf_counter() - t_pipeline_start) * 1000
        verdict.latency_ms = total_ms

        logger.info(
            "MedLingo inference complete — %.1f ms | confidence=%s",
            total_ms, verdict.overall_confidence,
        )

        return verdict

    def warm_up(self) -> None:
        """Pre-load all components to avoid cold-start latency."""
        if not self._initialized:
            self._initialize_components()
            logger.info("Pipeline warm-up complete.")

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------

    def _initialize_components(self) -> None:
        logger.info("Initializing MedLingo components...")

        with self._mm.component_scope("vision_encoder", expected_gb=0.3):
            ve_path = self._paths.vision_encoder
            self._vision_encoder = VisionEncoder(
                model_path=str(ve_path) if ve_path.exists() else None,
                device=self._device,
            )

        self._bridges = DualPathBridges()
        bridge_dir = self._paths.bridge_a.parent
        if bridge_dir.exists():
            self._bridges.load(bridge_dir)
            self._bridges.freeze_pathways()
        self._bridges.to(self._device)

        with self._mm.component_scope("router", expected_gb=1.2):
            self._router = SemanticTriageRouter(
                model_path=str(self._paths.router)
                if self._paths.router.exists() else None,
                device=self._device,
                top_k=self._cfg.get("spr", {}).get("top_k_experts", 2),
            )

        # Build adapter registry from resolved paths
        adapter_registry = {
            domain: (str(path) if path.exists() else None)
            for domain, path in self._paths.specialists.items()
        }

        with self._mm.component_scope("specialists", expected_gb=2.4):
            self._executor = ParallelExpertExecutor(
                adapter_registry=adapter_registry,
                base_model_path=str(self._paths.router)
                if self._paths.router.exists() else None,
                device=self._device,
                spr_max_rounds=self._cfg.get("spr", {}).get("max_rounds", 2),
            )

        with self._mm.component_scope("adjudicator", expected_gb=5.5):
            self._adjudicator = ChiefAdjudicator(
                model_path=str(self._paths.adjudicator)
                if self._paths.adjudicator.exists() else None,
                adapter_path=str(self._paths.adjudicator / "dpo_adapter")
                if (self._paths.adjudicator / "dpo_adapter").exists() else None,
                device=self._device,
                use_flash_attention=True,
            )

        self._initialized = True
        self._mm.log_snapshot(tag="post-init")

    # ------------------------------------------------------------------
    # Image processing
    # ------------------------------------------------------------------

    def _encode_image(self, image, image_description):
        import torch

        if image is None:
            dummy_z_s = torch.zeros(1, 2048, device=self._device)
            dummy_z_o = torch.zeros(1, 4096, device=self._device)
            return dummy_z_s, dummy_z_o, image_description or ""

        if isinstance(image, (str, Path)):
            pil_img = Image.open(str(image)).convert("RGB")
        else:
            pil_img = image

        pixel_values = self._vision_encoder.preprocess([pil_img])

        with torch.no_grad():
            visual_features = self._vision_encoder(pixel_values)   # (1, 768)
            z_s, z_o = self._bridges(visual_features)               # (1,2048), (1,4096)

        description = image_description or f"Medical image ({pil_img.size[0]}×{pil_img.size[1]})"
        return z_s, z_o, description

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(config_path) -> dict:
        import yaml
        from pathlib import Path as P

        if config_path is None:
            config_path = P(__file__).parent.parent.parent / "configs" / "inference_config.yaml"

        config_path = P(config_path)
        if not config_path.exists():
            logger.warning("Inference config not found at %s — using defaults.", config_path)
            return {}

        with open(config_path) as f:
            return yaml.safe_load(f) or {}
