#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC="$ROOT/src/recognition"
TOP_RECOG="$ROOT/recognition"

echo "Moving recognition assets into $SRC"
mkdir -p "$SRC/models"
mkdir -p "$SRC/dictionaries"

if [ -d "$TOP_RECOG/models" ]; then
  mv "$TOP_RECOG/models"/* "$SRC/models/" || true
fi

if [ -d "$TOP_RECOG/dictionaries" ]; then
  mv "$TOP_RECOG/dictionaries"/* "$SRC/dictionaries/" || true
fi

echo "Assets moved. You can remove the now-empty recognition/ directory if needed:"
echo "  rm -rf $TOP_RECOG"
