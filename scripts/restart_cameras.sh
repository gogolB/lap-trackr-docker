#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Restarting Jetson camera stack..."
sudo "${SCRIPT_DIR}/wait_for_jetson_cameras.sh"
echo "Done!"
