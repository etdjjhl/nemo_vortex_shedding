#!/usr/bin/env bash
# Run quantitative analysis on the test set.
# Computes RMSE / relative error for u, v, p and saves plots + summary table.
# Output: outputs/<date>/<time>/analysis/
# Usage: ./scripts/analyze.sh
#        ./scripts/analyze.sh num_test_samples=20
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON=/home/chenguanfeng/work/env_conda/nemo/bin/python

cd "$PROJECT_ROOT"

echo "==> Running quantitative analysis"
echo ""

"$PYTHON" analyze.py "$@"
