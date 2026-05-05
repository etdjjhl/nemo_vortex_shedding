#!/usr/bin/env bash
# Run inference on the test set and generate animated GIFs.
# Output: outputs/<date>/<time>/animations/animation_{u,v,p}.gif
# Usage: ./scripts/infer.sh
#        ./scripts/infer.sh num_test_samples=5
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON=/home/chenguanfeng/work/env_conda/nemo/bin/python

cd "$PROJECT_ROOT"

echo "==> Running inference and generating animations"
echo ""

"$PYTHON" inference.py "$@"
