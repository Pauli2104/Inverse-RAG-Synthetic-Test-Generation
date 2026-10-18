#!/usr/bin/env bash
# Part A: controlled study, statistics and figures (Python only, no internet). About 30 minutes on 2 CPU cores.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
WORKERS=${WORKERS:-2}
step() { echo; echo "== $1"; }
step "A1 main study: coverage, budget sweep, ablations, noise, Shapley, amplification"; $PY experiments.py
step "A2 critic comparison";                         $PY e4b.py
step "A3 size-matched diversity";                    $PY diversity.py
step "A4 independent faults and contemporary-style generators ($WORKERS workers)"; $PY revision_experiments.py "$WORKERS"
step "A5 critics on less structured documentation";  $PY messy_oracle.py
step "A6 uniform versus stratified selection";       $PY selection_experiment.py
step "A7 hypothesis tests, post hoc tests, sensitivity analyses"; $PY stats_analysis.py > /dev/null; $PY make_stats_tables.py > /dev/null
step "A8 figures";                                   $PY figures.py; $PY figures_revision.py
step "A9 framework export";                          $PY export_suite.py
if command -v pdflatex >/dev/null; then
  step "A10 Fig. 1 (TikZ)"; (cd figures && pdflatex -interaction=nonstopmode Fig1_source.tex > /dev/null && mv Fig1_source.pdf Fig1.pdf && rm -f Fig1_source.aux Fig1_source.log)
fi
echo; echo "Done: results/ and figures/ regenerated."
