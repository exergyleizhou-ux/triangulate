# Causal triangulation for biomarker–mortality associations — method, validation, tool

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20904051.svg)](https://doi.org/10.5281/zenodo.20904051)


This repository turns "triangulation" from a slogan into a **measured, reproducible instrument**
for deciding whether an observational biomarker–mortality association deserves a causal reading.
It is the methodological follow-up to the NHANES cadmium–mortality study, built to escape the
"another single-dataset association paper" trap by contributing a *method* (and an open tool),
not another association.

## What's here
| Folder | Contents |
|---|---|
| `sim/` | The simulation that **validates** the protocol: cohorts generated under known truth (causal / confounded / reverse-caused / null), the battery applied, operating characteristics (sensitivity, specificity, per-check complementarity, sweeps, a design-effect robustness check, and a non-proportional-hazards out-of-family stress test). `results*.json` hold the computed numbers. |
| `tool/` | `triangulate` — open-source (MIT) implementation of the protocol + a test suite (7/7) that verifies it on simulated cohorts of known truth. |
| `nhanes3/` | Parser + analysis **scripts** for the free temporal replication in NHANES III (1988–1994). The raw CDC public-use files (lab/adult/linked-mortality `.dat`) are not vendored here — `build_nhanes3.py` downloads/parses them; the analysis (`run_nhanes3.py`) then reproduces the urinary-cadmium HR 1.12 result. |
| `reproduce_all.sh`, `audit_numbers.py` | One command regenerates every figure and number; the audit confirms the manuscript matches the code. |

> The two manuscripts (Paper #1, *"When does triangulation work?"*, and the Paper #2 cross-cohort
> Stage-1 protocol) are released with the journal/OSF on publication; this repository is the
> reproducible **code** behind them.

## The idea in one paragraph
Individual checks have non-overlapping blind spots: a negative-control exposure detects generic
confounding but not reverse causation; a left-truncated landmark detects reverse causation but
not time-stable confounding; an E-value bounds unmeasured confounding but neither. Triangulation
*claims* these blind spots don't overlap. We pre-specify the battery as one decision rule, then
**measure** whether it works by simulating worlds whose truth we control. It does: the protocol
credits the genuinely causal exposure (high sensitivity) and rejects confounded, reverse-caused
and null exposures (high specificity), where a naïve "one significant model" analyst is fooled
every time — and each threat is caught by the check built for it.

## Reproduce
```bash
pip install pandas numpy scipy statsmodels matplotlib   # lifelines optional
./reproduce_all.sh        # one command: all operating characteristics, sweeps, the failure
                          # envelope, the method comparison, the non-PH stress test, figures,
                          # the tool test-suite (7/7), and the numerical-consistency audit
# or step by step:
cd sim && python runner.py && python sim_figures.py
cd ../tool && python test_triangulate.py
```

## Honesty notes
- Every number in the manuscript is computed by `sim/runner.py` (no fabricated values).
- The cross-cohort paper is a **pre-registration**, not results: the cohorts that carry both
  toxic-metal biomarkers and linked mortality are all controlled-access (fee / same-country
  collaborator), so we fix the hypotheses and analysis before any data are seen.
- The NHANES III replication is complete and reported in the paper: the legacy files are
  downloaded, the parser is validated (blood-lead median 3.20 µg/dL matches the literature), and
  the analysis runs end-to-end (urinary cadmium credited robust; reduced battery cannot
  discriminate lead, by design).
