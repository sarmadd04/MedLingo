#!/usr/bin/env bash
# =============================================================================
# scripts/infer.sh
#
# Launch MedLingo inference (interactive or single-query mode).
#
# Usage:
#   ./scripts/infer.sh --interactive
#   ./scripts/infer.sh --image scan.jpg --query "Any anomalies?"
#   ./scripts/infer.sh --batch queries.json --output results.json
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/.env"

if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

: "${MEDLINGO_WEIGHTS_DIR:?Set MEDLINGO_WEIGHTS_DIR in .env}"

cd "$PROJECT_ROOT"
python -m medlingo.inference.cli \
    --config configs/inference_config.yaml \
    --device auto \
    "$@"
