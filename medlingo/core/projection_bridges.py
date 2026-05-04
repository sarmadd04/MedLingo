"""
medlingo/core/projection_bridges.py

Dual-path multimodal alignment bridges as defined in §III-A.

Pathway A  (768 → 2048): maps visual features into the specialist latent space
Pathway B  (768 → 4096): maps visual features into the adjudicator latent space

Equations from the paper:
    z_s = W_s · v + b_s,  W_s ∈ R^{2048×768}   [Eq. 1]
    z_o = W_o · v + b_o,  W_o ∈ R^{4096×768}   [Eq. 2]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


class _MLP2Layer(nn.Module):
    """Two-layer MLP bridge with GELU activation and residual-style design."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden = max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, out_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PathwayA(_MLP2Layer):
    """
    Specialist projection bridge: R^768 → R^2048.

    Aligns BioMedCLIP visual tokens to the Llama-3.2-1B hidden-state space.
    Trained exclusively in Stage 1; frozen in all subsequent stages.
    """

    IN_DIM  = 768
    OUT_DIM = 2048

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__(self.IN_DIM, self.OUT_DIM, dropout)
        logger.debug("PathwayA bridge initialized: %d → %d", self.IN_DIM, self.OUT_DIM)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        visual_features : torch.Tensor
            Shape (B, 768) — output of frozen BioMedCLIP encoder.

        Returns
        -------
        torch.Tensor
            Shape (B, 2048) — specialist-aligned visual tokens z_s.
        """
        return super().forward(visual_features.float())


class PathwayB(_MLP2Layer):
    """
    Adjudicator projection bridge: R^768 → R^4096.

    Aligns BioMedCLIP visual tokens to the Llama-3.1-8B hidden-state space.
    Provides the orchestrator with an independent visual anchor for grounded
    adjudication (prevents hallucination acceptance from specialists).
    """

    IN_DIM  = 768
    OUT_DIM = 4096

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__(self.IN_DIM, self.OUT_DIM, dropout)
        logger.debug("PathwayB bridge initialized: %d → %d", self.IN_DIM, self.OUT_DIM)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        visual_features : torch.Tensor
            Shape (B, 768) — output of frozen BioMedCLIP encoder.

        Returns
        -------
        torch.Tensor
            Shape (B, 4096) — adjudicator-aligned visual tokens z_o.
        """
        return super().forward(visual_features.float())


class DualPathBridges(nn.Module):
    """
    Container that runs both pathways in a single forward pass.

    Returns a namedtuple with specialist_tokens and orchestrator_tokens.
    """

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.pathway_a = PathwayA(dropout=dropout)
        self.pathway_b = PathwayB(dropout=dropout)

    def forward(
        self,
        visual_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        z_s : torch.Tensor  (B, 2048)  — for specialists
        z_o : torch.Tensor  (B, 4096)  — for adjudicator
        """
        z_s = self.pathway_a(visual_features)
        z_o = self.pathway_b(visual_features)
        return z_s, z_o

    def freeze_pathways(self) -> None:
        """Freeze both bridges after Stage 1 training completes."""
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()
        logger.info("Dual-path bridges frozen post Stage 1.")

    def save(self, directory: Union[str, Path]) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.pathway_a.state_dict(), directory / "bridge_a.pt")
        torch.save(self.pathway_b.state_dict(), directory / "bridge_b.pt")
        logger.info("Bridges saved to %s", directory)

    def load(self, directory: Union[str, Path]) -> None:
        directory = Path(directory)
        self.pathway_a.load_state_dict(
            torch.load(directory / "bridge_a.pt", weights_only=True)
        )
        self.pathway_b.load_state_dict(
            torch.load(directory / "bridge_b.pt", weights_only=True)
        )
        logger.info("Bridges loaded from %s", directory)
