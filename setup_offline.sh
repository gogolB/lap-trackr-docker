#!/usr/bin/env bash
#
# setup_offline.sh — Set up a local Python environment for offline grading.
#
# Creates a venv, installs PyTorch (CUDA > MPS > CPU), SAM2, CoTracker,
# and all grader dependencies. Also creates the data/ directory structure.
#
# Usage:
#   ./setup_offline.sh              # auto-detect GPU
#   ./setup_offline.sh --cpu        # force CPU-only install
#   ./setup_offline.sh --cuda       # force CUDA install
#   ./setup_offline.sh --mps        # force MPS (Apple Silicon) install
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
DATA_DIR="$SCRIPT_DIR/data"
GRADER_DIR="$SCRIPT_DIR/services/grader"
PYTHON="${PYTHON:-python3}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

info()  { echo "==> $*"; }
warn()  { echo "WARNING: $*" >&2; }
error() { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------

FORCE_DEVICE=""
DOWNLOAD_MODELS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)   FORCE_DEVICE="cpu";  shift ;;
        --cuda)  FORCE_DEVICE="cuda"; shift ;;
        --mps)   FORCE_DEVICE="mps";  shift ;;
        --download-models) DOWNLOAD_MODELS=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--cpu|--cuda|--mps] [--download-models]"
            echo ""
            echo "Options:"
            echo "  --cpu              Force CPU-only PyTorch install"
            echo "  --cuda             Force CUDA PyTorch install"
            echo "  --mps              Force MPS (Apple Silicon) PyTorch install"
            echo "  --download-models  Download ML models from HuggingFace"
            echo ""
            echo "If no device flag is given, auto-detects: CUDA > MPS > CPU."
            exit 0
            ;;
        *) error "Unknown option: $1" ;;
    esac
done

# ---------------------------------------------------------------------------
# Detect device
# ---------------------------------------------------------------------------

detect_device() {
    if [[ -n "$FORCE_DEVICE" ]]; then
        echo "$FORCE_DEVICE"
        return
    fi

    # Check for NVIDIA GPU
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        echo "cuda"
        return
    fi

    # Check for Apple Silicon (macOS with arm64)
    if [[ "$(uname -s)" == "Darwin" ]] && [[ "$(uname -m)" == "arm64" ]]; then
        echo "mps"
        return
    fi

    echo "cpu"
}

DEVICE=$(detect_device)
info "Target device: $DEVICE"

# ---------------------------------------------------------------------------
# Create data directory structure
# ---------------------------------------------------------------------------

info "Creating data directory structure"
mkdir -p "$DATA_DIR/sessions"
mkdir -p "$DATA_DIR/models/sam2"
mkdir -p "$DATA_DIR/models/cotracker"
mkdir -p "$DATA_DIR/models/yolov11-pose"

info "Data directory layout:"
echo "  $DATA_DIR/"
echo "  ├── sessions/          <- copy session dirs here via scp"
echo "  └── models/"
echo "      ├── sam2/          <- sam2.1_hiera_large.pt"
echo "      ├── cotracker/     <- scaled_offline.pth"
echo "      └── yolov11-pose/  <- yolo11l_pose_1088.pt"

# ---------------------------------------------------------------------------
# Create virtual environment
# ---------------------------------------------------------------------------

if [[ -d "$VENV_DIR" ]]; then
    info "Virtual environment already exists at $VENV_DIR"
else
    info "Creating virtual environment at $VENV_DIR"
    $PYTHON -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"
info "Using Python: $(python --version) at $(which python)"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# Install PyTorch
# ---------------------------------------------------------------------------

info "Installing PyTorch for device=$DEVICE"
case "$DEVICE" in
    cuda)
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
        ;;
    mps|cpu)
        # Standard PyTorch install works for both MPS (macOS) and CPU
        pip install torch torchvision
        ;;
esac

# ---------------------------------------------------------------------------
# Install grader dependencies
# ---------------------------------------------------------------------------

info "Installing grader base dependencies"
pip install \
    "numpy>=1.26,<2.0" \
    "scipy>=1.11,<1.15" \
    "opencv-python-headless>=4.10,<5.0" \
    "sqlalchemy>=2.0,<3.0" \
    "psycopg2-binary>=2.9,<3.0" \
    "rich>=13.0"

# ---------------------------------------------------------------------------
# Install SAM2, SAM3, and CoTracker
# ---------------------------------------------------------------------------

info "Installing SAM2 (facebook/sam2)"
pip install "sam-2 @ git+https://github.com/facebookresearch/sam2.git"

info "Installing SAM3 (facebook/sam3)"
pip install "sam3 @ git+https://github.com/facebookresearch/sam3.git"

info "Installing CoTracker (facebook/co-tracker)"
pip install "git+https://github.com/facebookresearch/co-tracker.git"

# ---------------------------------------------------------------------------
# Download models (optional)
# ---------------------------------------------------------------------------

if [[ "$DOWNLOAD_MODELS" == "true" ]]; then
    info "Downloading ML models..."

    # SAM2 checkpoint
    SAM2_PATH="$DATA_DIR/models/sam2/sam2.1_hiera_large.pt"
    if [[ -f "$SAM2_PATH" ]]; then
        info "SAM2 model already exists, skipping"
    else
        info "Downloading SAM2 checkpoint (~900 MB)..."
        curl -L -o "$SAM2_PATH" \
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    fi

    # CoTracker checkpoint
    CT_PATH="$DATA_DIR/models/cotracker/scaled_offline.pth"
    if [[ -f "$CT_PATH" ]]; then
        info "CoTracker model already exists, skipping"
    else
        info "Downloading CoTracker v3 offline checkpoint..."
        mkdir -p "$DATA_DIR/models/cotracker"
        curl -L -o "$CT_PATH" \
            "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth"
    fi

    info "Model download complete"
else
    info ""
    info "Models NOT downloaded (use --download-models to fetch from HuggingFace)."
    info "Or scp them from the Jetson:"
    info "  scp jetson:/data/models/sam2/sam2.1_hiera_large.pt $DATA_DIR/models/sam2/"
    info "  scp jetson:/data/models/cotracker/cotracker-v3-offline/scaled_offline.pth $DATA_DIR/models/cotracker/"
    info "  scp jetson:/data/models/yolov11-pose/yolo11l_pose_1088.pt $DATA_DIR/models/yolov11-pose/"
fi

# ---------------------------------------------------------------------------
# Verify installation
# ---------------------------------------------------------------------------

info "Verifying installation..."

python -c "
import torch
print(f'  PyTorch {torch.__version__}')
if torch.cuda.is_available():
    print(f'  CUDA: {torch.cuda.get_device_name(0)}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print(f'  MPS: available')
else:
    print(f'  CPU only')

import numpy; print(f'  NumPy {numpy.__version__}')
import scipy; print(f'  SciPy {scipy.__version__}')
import cv2;   print(f'  OpenCV {cv2.__version__}')

try:
    import sam2; print('  SAM2: installed')
except ImportError:
    print('  SAM2: NOT installed')

try:
    import cotracker; print('  CoTracker: installed')
except ImportError:
    print('  CoTracker: NOT installed')

try:
    import sam3; print('  SAM3: installed')
except ImportError:
    print('  SAM3: NOT installed')

try:
    import rich; print(f'  Rich {rich.__version__}')
except ImportError:
    print('  Rich: NOT installed')
"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

echo ""
echo "=============================================="
echo "  Offline grading environment ready!"
echo "=============================================="
echo ""
echo "Next steps:"
echo ""
echo "  1. Copy session data:"
echo "     scp -r jetson:/data/users/<user_id>/<session_timestamp> $DATA_DIR/sessions/"
echo ""
echo "  2. Copy models (if not downloaded):"
echo "     scp jetson:/data/models/sam2/sam2.1_hiera_large.pt $DATA_DIR/models/sam2/"
echo "     scp jetson:/data/models/cotracker/cotracker-v3-offline/scaled_offline.pth $DATA_DIR/models/cotracker/"
echo ""
echo "  3. Run the grader:"
echo "     ./run_offline.sh data/sessions/<session_dir>"
echo ""
echo "  4. Or try SAM3 (auto-downloads checkpoint from HuggingFace):"
echo "     ./run_offline.sh data/sessions/<session_dir> --segmentation-backend sam3"
echo ""
