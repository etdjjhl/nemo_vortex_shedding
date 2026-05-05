#!/usr/bin/env bash
# Download the DeepMind cylinder_flow dataset.
# The dataset is ~1 GB. Requires git and an internet connection.
# Output: raw_dataset/cylinder_flow/
# Usage: ./scripts/download_data.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "==> Downloading cylinder_flow dataset from DeepMind"
echo "    This may take a while depending on your connection speed."
echo ""

command -v git >/dev/null 2>&1 || { echo "Error: git is required but not found."; exit 1; }

mkdir -p raw_dataset
cd raw_dataset

if [ ! -d deepmind-research ]; then
    echo "==> Cloning DeepMind research repository (shallow clone)..."
    git clone --depth=1 https://github.com/deepmind/deepmind-research.git
else
    echo "==> deepmind-research already present, skipping clone."
fi

echo "==> Running dataset download script..."
sh deepmind-research/meshgraphnets/download_dataset.sh cylinder_flow cylinder_flow

echo ""
echo "==> Done!"
echo "    Dataset location: $PROJECT_ROOT/raw_dataset/cylinder_flow/"
echo "    Update conf/config.yaml → data_dir if you placed it elsewhere."
