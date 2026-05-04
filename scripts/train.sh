#!/usr/bin/env bash
# =============================================================================
# scripts/train.sh
#
# Launch MedLingo training on a single-GPU machine.
# Loads environment from .env and dispatches to the training pipeline.
#
# Usage:
#   ./scripts/train.sh                         # Full 4-stage pipeline
#   ./scripts/train.sh --stages 1 2            # Alignment + specialists only
#   ./scripts/train.sh --stages 2 --domain radiology
#   ./scripts/train.sh --stages 3 4 --resume   # Resume from stage 3
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/.env"

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
if [ -f "$ENV_FILE" ]; then
    echo "[train.sh] Loading environment from $ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
else
    echo "[train.sh] WARNING: .env not found at $PROJECT_ROOT — using system environment."
fi

# Validate required environment variables
: "${MEDLINGO_WEIGHTS_DIR:?Set MEDLINGO_WEIGHTS_DIR in .env}"

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    echo "[train.sh] GPU detected: $GPU_NAME ($GPU_MEM) — $GPU_COUNT device(s)"
else
    echo "[train.sh] WARNING: nvidia-smi not found. Training will use CPU."
fi

# ---------------------------------------------------------------------------
# Weights & Biases (optional)
# ---------------------------------------------------------------------------
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "[train.sh] W&B logging enabled."
    export WANDB_PROJECT="${WANDB_PROJECT:-medlingo-training}"
else
    echo "[train.sh] W&B not configured (WANDB_API_KEY not set). Logging disabled."
    export WANDB_DISABLED=true
fi

# ---------------------------------------------------------------------------
# Launch training
# ---------------------------------------------------------------------------
cd "$PROJECT_ROOT"
echo "[train.sh] Starting MedLingo training pipeline..."
echo "[train.sh] Weights directory: $MEDLINGO_WEIGHTS_DIR"
echo ""

python -m medlingo.training.train_pipeline \
    --config configs/training_config.yaml \
    --device auto \
    "$@"

echo ""
echo "[train.sh] Training complete."
