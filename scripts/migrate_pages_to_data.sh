#!/usr/bin/env bash
set -euo pipefail

# Simple migration script: moves existing pages/ directory to data/
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

if [ ! -d "$ROOT_DIR/pages" ]; then
  echo "No pages/ directory found — nothing to migrate."
  exit 0
fi

if [ -d "$ROOT_DIR/data" ]; then
  echo "data/ already exists. Will copy pages/* into data/ while keeping existing files."
  cp -r "$ROOT_DIR/pages/"* "$ROOT_DIR/data/"
else
  echo "Renaming pages/ -> data/"
  mv "$ROOT_DIR/pages" "$ROOT_DIR/data"
fi

echo "Migration complete. To keep backward compatibility, you can create a symlink:"
echo "  ln -s data pages"
