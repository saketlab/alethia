#!/usr/bin/env bash
# Build the alethia documentation (MkDocs Material).
#
# Usage:
#   scripts/build_docs.sh              build the static site into site/
#   scripts/build_docs.sh --execute    re-run the example notebooks first, then build
#   scripts/build_docs.sh --strict     treat MkDocs warnings as errors (matches CI)
#   scripts/build_docs.sh --serve      serve with live reload at http://localhost:8000
#
# Flags combine, e.g.  scripts/build_docs.sh --execute --strict

set -euo pipefail

cd "$(dirname "$0")/.."

EXECUTE=0
STRICT=0
SERVE=0
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    --strict)  STRICT=1 ;;
    --serve)   SERVE=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [ "$EXECUTE" -eq 1 ]; then
  echo "Executing notebooks..."
  jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=900 docs/notebooks/*.ipynb
fi

if [ "$SERVE" -eq 1 ]; then
  echo "Serving at http://localhost:8000 (Ctrl-C to stop)"
  exec mkdocs serve
fi

MKDOCS_OPTS=()
if [ "$STRICT" -eq 1 ]; then
  MKDOCS_OPTS+=(--strict)
fi

echo "Building site..."
mkdocs build "${MKDOCS_OPTS[@]}"
echo "Built site/index.html"
