#!/usr/bin/env bash
# Quick smoke test: 2 epochs, 20 samples — completes in a few minutes.
# Use this to verify the entire pipeline (data loading → training → checkpoint) works
# before launching a full run.
# Usage: ./scripts/train_quick.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON=/home/chenguanfeng/work/env_conda/nemo/bin/python

cd "$PROJECT_ROOT"

echo "==> Quick smoke test (2 epochs, 20 samples)"
echo "==> Project root : $PROJECT_ROOT"
echo ""

"$PYTHON" train.py +experiment=quick_test "$@"
