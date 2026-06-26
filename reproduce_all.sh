#!/usr/bin/env bash
# Reproduce every number and figure in "When does triangulation work?" from scratch.
# Each step is seeded; re-running gives identical results. Requires Python 3.11 +
# numpy/scipy/pandas/statsmodels (and matplotlib for figures). ~a few minutes total.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> [1/8] Core operating characteristics + sweeps  (sim/results.json)"
( cd sim && python runner.py )

echo "==> [2/8] Failure envelope: weak control, causal+confounding  (results_adversarial.json)"
( cd sim && python adversarial.py )

echo "==> [3/8] Threshold sweep, collider blind spot, naive-vs-strength  (results_extreme.json)"
( cd sim && python extreme.py )

echo "==> [4/8] Out-of-design: NHANES plasmode + randomized ensemble  (results_adversarial_dgp.json)"
( cd sim && python adversarial_dgp.py )   # needs ../../nhanes-paper covariate matrix; see header

echo "==> [5/8] Head-to-head methods + the negctrl+landmark composite  (results_comparison.json)"
( cd sim && python compare_methods.py )

echo "==> [6/8] Design-effect (survey-variance) robustness of the operating characteristics"
( cd sim && python robust_se_check.py )

echo "==> [6b] Out-of-family stress test: non-proportional-hazards log-logistic AFT  (results_nonph_weibull.json)"
( cd sim && python nonph_weibull.py )

echo "==> [7/8] Figures  (figures/sim_fig*.pdf)"
( cd sim && python sim_figures.py )

echo "==> [8/8] Tool test-suite + numerical-consistency audit"
( cd tool && python test_triangulate.py )
python audit_numbers.py

echo ""
echo "NHANES worked examples (need the CDC public-use files, not bundled):"
echo "  nhanes3/build_nhanes3.py  -> analytic_nhanes3.csv   (parse raw NHANES III .dat)"
echo "  nhanes3/export_clean.py   -> analytic_nhanes3_clean.csv (public-deposit columns)"
echo "  nhanes3/run_nhanes3.py    -> nhanes3_replication_results.txt"
echo ""
echo "DONE — every headline number and figure regenerated and audited."
