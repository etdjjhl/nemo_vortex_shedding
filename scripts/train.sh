#!/usr/bin/env bash
# Standard training run.
# Usage: ./scripts/train.sh
#        ./scripts/train.sh epochs=50          # override any config field
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON=/home/chenguanfeng/work/env_conda/nemo/bin/python

cd "$PROJECT_ROOT"

echo "==> Project root : $PROJECT_ROOT"
echo "==> Python       : $PYTHON"
echo "==> Config       : conf/config.yaml (full_train)"
echo ""

"$PYTHON" train.py "$@"
