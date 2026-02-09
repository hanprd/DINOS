#!/usr/bin/env bash
set -euo pipefail

DATASET="changheonhan/dinos-diverse-industrial-operation-sounds"

FILES=(
    "DINOS.7z"
)

for f in "${FILES[@]}"; do
    echo "Downloading ${f}..."
    kaggle datasets download -d "$DATASET" -f "$f" --unzip
done

echo "All downloads complete."
