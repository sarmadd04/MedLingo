"""
medlingo/utils/quantization.py

NF4 / BitsAndBytes quantization configuration helpers.
All MedLingo LLM backbones (Router, Specialists, Adjudicator) are loaded
in 4-bit NormalFloat (NF4) with double-quantization as described in §V-B.
"""

from __future__ import annotations

import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from transformers import BitsAndBytesConfig
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False
    logger.warning("bitsandbytes not available — quantization disabled.")


def nf4_config(
    double_quant: bool = True,
    compute_dtype: Optional[torch.dtype] = None,
) -> "BitsAndBytesConfig":
    """
    Return a BitsAndBytesConfig for 4-bit NormalFloat quantization.

    Parameters
    ----------
    double_quant : bool
        Enable double-quantization to further compress quantization constants.
    compute_dtype : torch.dtype, optional
        Dtype for compute operations. Defaults to bfloat16 on capable hardware,
        falls back to float16.
    """
    if not _BNB_AVAILABLE:
        raise RuntimeError(
            "bitsandbytes is required for NF4 quantization. "
            "Install with: pip install bitsandbytes>=0.43.0"
        )

    if compute_dtype is None:
        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

    logger.debug(
        "NF4 config: double_quant=%s, compute_dtype=%s",
        double_quant, compute_dtype,
    )

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def get_torch_dtype(prefer_bf16: bool = True) -> torch.dtype:
    """
    Resolve the best available floating-point dtype for this hardware.

    BF16 is preferred on Ampere (RTX 3xxx) and later GPUs.
    FP16 is used as fallback on older CUDA hardware.
    CPU inference always uses FP32.
    """
    if not torch.cuda.is_available():
        return torch.float32
    if prefer_bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def estimate_vram_gb(
    num_params_b: float,
    bits: int = 4,
    overhead_factor: float = 1.15,
) -> float:
    """
    Rough VRAM estimate for a quantized model.

    Parameters
    ----------
    num_params_b : float
        Model size in billions of parameters.
    bits : int
        Quantization bit-width (4 for NF4).
    overhead_factor : float
        Safety multiplier for KV-cache and activation overhead.
    """
    bytes_per_param = bits / 8
    raw_gb = (num_params_b * 1e9 * bytes_per_param) / (1024 ** 3)
    return raw_gb * overhead_factor
