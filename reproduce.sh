#!/usr/bin/env bash
# Reproduction driver.  Usage: bash reproduce.sh [tests|studies|quick|evaluate|assets|all]
# Hardware used for the reported times: 2 CPU cores, 7 GB memory, no GPU.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
step=${1:-all}

tests()    { $PY -m pytest -q tests; }
studies()  { ( cd experiments
               $PY exp_fidelity.py
               $PY exp_linear_stochastic.py
               $PY exp_joint.py
               $PY exp_common_domain.py
               $PY exp_incomplete.py
               $PY exp_latency.py ); }
evaluate() { # Re-runs the acceptance test with the shipped checkpoints.  The run reported in the article is kept in
             # results/reference/ and is compared with the new outcome, never overwritten; the earlier runs (before the
             # review corrections) are kept under results/ as listed in docs/PROTOCOL.md.
             ( cd experiments
               $PY evaluate.py structured
               $PY evaluate.py nl
               $PY evaluate.py reading
               $PY evaluate.py summarise
               $PY criterion_effect.py
               $PY compare_with_reference.py ); }
quick()    { # about 10 minutes: coordinator on the 600 structured cases, deployed reader on the 229 requests; compared with the reference run
             ( cd experiments && $PY evaluate.py structured --quick && $PY evaluate.py nl --quick && $PY compare_with_reference.py --quick ); }
assets()   { ( cd experiments && $PY make_figures.py && $PY make_paper_assets.py )
             ( cd paper/figures && for f in fig_lineage fig_framework fig_pipeline fig_human; do pdflatex -interaction=nonstopmode -halt-on-error $f.tex >/dev/null; done ); }
case "$step" in
  tests) tests ;; studies) studies ;; evaluate) evaluate ;; quick) quick ;; assets) assets ;;
  all) tests; studies; evaluate; assets ;;
  *) echo "unknown step: $step"; exit 2 ;;
esac
