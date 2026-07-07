#!/usr/bin/env bash
# RDDAC end-to-end acceptance: fresh install -> real download -> tutorials -> tests.
#
# What it does, in order, all in a throwaway directory:
#   1. builds a fresh venv and installs rddac[torch] from this repo (wheel build,
#      dependencies incl. ddacs resolved from PyPI — exactly what a user gets)
#   2. downloads the small bundle of the requested dataset version from DaRUS
#      with the real CLI (API token from .env when present, needed for drafts)
#   3. executes all six tutorial notebooks against the downloaded data
#   4. runs the full test suite with the small-bundle fixture pointed at the
#      downloaded data (so the tests exercise the download too)
#
# Usage:
#   ./scripts/e2e.sh              # dataset version = package default
#   ./scripts/e2e.sh :draft       # against the draft (needs DARUS_API_TOKEN in .env)
#   KEEP=1 ./scripts/e2e.sh       # keep the work dir for inspection
set -uo pipefail

VERSION="${1:-}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d /tmp/rddac_e2e.XXXXXX)"
LOG="$WORK/e2e.log"

say()  { echo -e "\n=== $* ===" | tee -a "$LOG"; }
fail() { echo "E2E FAILED at: $*  (log: $LOG, work dir kept: $WORK)"; exit 1; }

[ -f "$REPO/.env" ] && { set -a; source "$REPO/.env"; set +a; }

say "1/4 fresh install into $WORK"
uv venv --python 3.11 "$WORK/.venv" >> "$LOG" 2>&1 || fail "venv creation"
uv pip install --python "$WORK/.venv/bin/python" \
    "$REPO[torch]" nbformat nbclient ipykernel pytest >> "$LOG" 2>&1 || fail "install"
PY="$WORK/.venv/bin/python"
"$PY" -c "import rddac, ddacs; print('rddac', rddac.__version__, '| ddacs', ddacs.__version__)" \
    | tee -a "$LOG" || fail "import"

say "2/4 download (version: ${VERSION:-package default})"
DL=("$WORK/.venv/bin/rddac")
[ -n "${DARUS_API_TOKEN:-}" ] && DL+=(--token "$DARUS_API_TOKEN")
DL+=(download)
[ -n "$VERSION" ] && DL+=("$VERSION")
DL+=(--small -y --quiet --out "$WORK/data")
"${DL[@]}" || fail "download"
for f in metadata.json process_parameters.csv h5/sample.zip; do
    [ -f "$WORK/data/$f" ] || fail "downloaded file missing: $f"
done
du -sh "$WORK/data" | tee -a "$LOG"

say "3/4 tutorials against the download"
cp "$REPO"/notebooks/0*.ipynb "$WORK/notebooks_run" 2>/dev/null || mkdir -p "$WORK/notebooks_run"
cp "$REPO"/notebooks/0*.ipynb "$WORK/notebooks_run/"
cat > "$WORK/run_nb.py" << 'PYEOF'
import sys, time
import nbformat
from nbclient import NotebookClient
name = sys.argv[1]
nb = nbformat.read(name, as_version=4)
t0 = time.time()
NotebookClient(nb, timeout=300, kernel_name="python3",
               resources={"metadata": {"path": "."}}).execute()
nbformat.write(nb, name)
print(f"{name}: OK in {time.time()-t0:.0f}s")
PYEOF
cd "$WORK/notebooks_run"
export PATH="$WORK/.venv/bin:$PATH"
for nb in 0*.ipynb; do
    "$PY" "$WORK/run_nb.py" "$nb" | tee -a "$LOG" || fail "notebook $nb"
done
cd "$REPO"

say "4/4 test suite against the download"
RDDAC_SMALL_DATA_DIR="$WORK/data" "$PY" -m pytest "$REPO/tests" -q 2>&1 | tail -2 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "pytest"

say "E2E PASSED"
if [ -z "${KEEP:-}" ]; then rm -rf "$WORK"; else echo "kept: $WORK"; fi
