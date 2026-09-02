#!/usr/bin/env bash
# Assemble the teaser bundle under publish/.staging/.
#
# Layout mirrors the repo (data/ + notebooks/) so the notebooks' hard-coded
# DATA_DIR = '../data' keeps working with no edits:
#
#   .staging/
#     data/{metadata.json, process_parameters.csv, h5/sample.zip}
#     rddac_documentation.pdf
#     README.md
#
# Notebooks are published separately as Kaggle kernels (publish/kaggle/kernels),
# so they are intentionally not bundled into the dataset.
#
# Source dir holding metadata.json + process_parameters.csv + h5/sample.zip.
# Defaults to ./data; override with RDDAC_TEASER_SRC.
# Files are dereferenced on copy. Only .staging/ is git-ignored; the scripts are committed.
set -euo pipefail

PUB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # publish/
ROOT="$(cd "$PUB/.." && pwd)"                          # repo root
STAGE="$PUB/.staging"
SRC="${RDDAC_TEASER_SRC:-$ROOT/data}"

rm -rf "$STAGE"
mkdir -p "$STAGE/data/h5"

# 1. Data bundle — require the three core files to resolve under $SRC.
missing=()
for rel in metadata.json process_parameters.csv "h5/sample.zip"; do
  [ -e "$SRC/$rel" ] || missing+=("$rel")
done
if [ "${#missing[@]}" -ne 0 ]; then
  echo "Missing under SRC='$SRC': ${missing[*]}" >&2
  echo "Point RDDAC_TEASER_SRC at a dir with metadata.json, process_parameters.csv, h5/sample.zip" >&2
  echo "(e.g. RDDAC_TEASER_SRC=<data source dir> $0)" >&2
  exit 1
fi
echo "Staging data from $SRC ..."
cp -L "$SRC/metadata.json"          "$STAGE/data/metadata.json"
cp -L "$SRC/process_parameters.csv" "$STAGE/data/process_parameters.csv"
cp -L "$SRC/h5/sample.zip"          "$STAGE/data/h5/sample.zip"

# Documentation PDF — from the source dir, else from the doc build in .helper.
if [ -e "$SRC/rddac_documentation.pdf" ]; then
  cp -L "$SRC/rddac_documentation.pdf" "$STAGE/rddac_documentation.pdf"
  echo "Included rddac_documentation.pdf (from SRC)"
elif [ -e "$ROOT/.helper/v1/documentation/rddac_documentation.pdf" ]; then
  cp -L "$ROOT/.helper/v1/documentation/rddac_documentation.pdf" "$STAGE/rddac_documentation.pdf"
  echo "Included rddac_documentation.pdf (from .helper)"
else
  echo "note: rddac_documentation.pdf not found — skipping" >&2
fi

# 2. Bundle-level readme (HF's upload overwrites this with the dataset card).
cp "$PUB/teaser/README.md" "$STAGE/README.md"

# 3. Platform descriptions are generated from this README (single source).
python3 "$PUB/build_descriptions.py"

echo "Staged teaser at $STAGE"
du -sh "$STAGE"
