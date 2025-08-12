#!/usr/bin/env bash
# Run the geometry baseline pipeline over all defined anchors, f values and seeds.

set -euo pipefail

# Activate placeholder mode if desired.  To test the code without real data,
# export FPHS_GEOM_USE_PLACEHOLDER=1 before running this script.

# Determine project root (directory of this script's parent).
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( dirname "$SCRIPT_DIR" )"

CONFIG="${REPO_ROOT}/configs/anchors.yaml"
DATA_DIR="${REPO_ROOT}/data/inputs"
OUT_DIR="${REPO_ROOT}/runs"

echo "Running geometry baseline analysis..."
python -m src.geometry.run_geometry --config "$CONFIG" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"

# Aggregate results and generate report.
python "${SCRIPT_DIR}/aggregate.py" --runs-dir "$OUT_DIR" --out-dir "${REPO_ROOT}/results"
python "${SCRIPT_DIR}/generate_report.py" --summary "${REPO_ROOT}/results/summary.csv" --out-report "${REPO_ROOT}/reports/REPORT.md"

echo "All done.  Results in ${REPO_ROOT}/results and report at reports/REPORT.md."