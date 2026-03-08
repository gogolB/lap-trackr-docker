#!/usr/bin/env bash
#
# run_offline.sh — Run the offline grader on a session directory.
#
# Usage:
#   ./run_offline.sh <session_dir> [options]
#   ./run_offline.sh data/sessions/2025-01-15_10-30-00
#   ./run_offline.sh data/sessions/2025-01-15_10-30-00 --segmentation-backend sam3
#   ./run_offline.sh data/sessions/2025-01-15_10-30-00 --sample-interval 3 --device mps
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
DATA_DIR="$SCRIPT_DIR/data"
GRADER_DIR="$SCRIPT_DIR/services/grader"

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

if [[ $# -lt 1 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    echo "Usage: $0 <session_dir> [grade_offline.py options]"
    echo ""
    echo "Arguments:"
    echo "  session_dir    Path to a session directory containing MP4, NPZ, and tip_init.json"
    echo ""
    echo "Options (passed through to grade_offline.py):"
    echo "  --sample-interval N            Keep every Nth frame (default: 1 = all)"
    echo "  --device DEVICE                Force device: cuda, mps, or cpu"
    echo "  --segmentation-backend BACKEND Use sam2 (default) or sam3 for Pass 1"
    echo "  --sam2-model PATH              Override SAM2 checkpoint path"
    echo "  --sam3-model PATH              Override SAM3 checkpoint path"
    echo "  --cotracker-model PATH         Override CoTracker checkpoint path"
    echo ""
    echo "Examples:"
    echo "  $0 data/sessions/2025-01-15_10-30-00"
    echo "  $0 data/sessions/2025-01-15_10-30-00 --segmentation-backend sam3"
    echo "  $0 data/sessions/2025-01-15_10-30-00 --sample-interval 3 --device cpu"
    exit 0
fi

SESSION_DIR="$1"
shift  # remaining args passed through to grade_offline.py

# Resolve relative paths
if [[ ! "$SESSION_DIR" = /* ]]; then
    SESSION_DIR="$SCRIPT_DIR/$SESSION_DIR"
fi

if [[ ! -d "$SESSION_DIR" ]]; then
    echo "ERROR: Session directory not found: $SESSION_DIR" >&2
    exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: Virtual environment not found. Run ./setup_offline.sh first." >&2
    exit 1
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# ---------------------------------------------------------------------------
# Auto-detect model paths (if not overridden via CLI args)
# ---------------------------------------------------------------------------

ALL_ARGS="$*"
EXTRA_ARGS=()

# Detect if user requested SAM3
USING_SAM3=false
if echo "$ALL_ARGS" | grep -q -- "--segmentation-backend sam3"; then
    USING_SAM3=true
fi

# Auto-detect SAM2 model (only if not using SAM3 and not explicitly set)
if [[ "$USING_SAM3" == "false" ]] && ! echo "$ALL_ARGS" | grep -q -- "--sam2-model"; then
    SAM2_PATH="$DATA_DIR/models/sam2/sam2.1_hiera_large.pt"
    if [[ -f "$SAM2_PATH" ]]; then
        EXTRA_ARGS+=(--sam2-model "$SAM2_PATH")
    else
        echo "WARNING: SAM2 model not found at $SAM2_PATH"
        echo "  Pass 1 will fail. Download the model, or try: --segmentation-backend sam3"
    fi
fi

# Auto-detect CoTracker model (if not explicitly set)
if ! echo "$ALL_ARGS" | grep -q -- "--cotracker-model"; then
    CT_PATH="$DATA_DIR/models/cotracker/scaled_offline.pth"
    if [[ -f "$CT_PATH" ]]; then
        EXTRA_ARGS+=(--cotracker-model "$CT_PATH")
    else
        echo "WARNING: CoTracker model not found at $CT_PATH"
        echo "  Pass 2 (point tracking) will be skipped."
    fi
fi

# ---------------------------------------------------------------------------
# Run the grader
# ---------------------------------------------------------------------------

cd "$GRADER_DIR"

exec python -m app.grade_offline "$SESSION_DIR" "${EXTRA_ARGS[@]}" "$@"
