#!/usr/bin/env bash
# Part B: run the exported suite natively under Inspect AI (and promptfoo if Node.js is installed)
# for the reference agent and the 15 design faults, and compare every verdict with the harness.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
$PY export_suite.py
FW="inspect"
command -v npx >/dev/null && FW="inspect promptfoo"
$PY frameworks/measure_agreement.py --frameworks $FW
echo "Agreement: frameworks/agreement/summary.json"
