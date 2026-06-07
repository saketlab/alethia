#!/usr/bin/env bash
# Build the alethia documentation.
#
# Usage:
#   scripts/build_docs.sh              build HTML from committed notebook outputs
#   scripts/build_docs.sh --execute    re-run the notebooks first, then build
#   scripts/build_docs.sh --strict     treat Sphinx warnings as errors (matches CI)
#   scripts/build_docs.sh --serve      build, then serve at http://localhost:8000
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

SPHINX_OPTS=()
if [ "$STRICT" -eq 1 ]; then
  SPHINX_OPTS+=(-W --keep-going)
fi

echo "Building HTML docs..."
sphinx-build -b html "${SPHINX_OPTS[@]}" docs docs/_build/html
echo "Built docs/_build/html/index.html"

if [ "$SERVE" -eq 1 ]; then
  echo "Serving at http://localhost:8000 (Ctrl-C to stop)"
  python -m http.server 8000 --directory docs/_build/html
fi
