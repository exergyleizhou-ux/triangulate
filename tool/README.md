# `triangulate` — a causal-triangulation protocol for biomarker–mortality associations

Observational biomarker–mortality associations are routinely confounded, reverse-caused, or
method-fragile, and the field has no systematic way to decide which deserve a causal reading.
`triangulate` runs a **pre-specified discrimination battery** on a single cohort and returns a
structured verdict — *robust-causal-candidate* vs *fragile-or-spurious* — with the per-check
reasoning. Each check targets a distinct threat to validity, so an artefact that defeats one is
not expected to defeat the others.

It is the protocol applied in the NHANES cadmium–mortality study and **validated by simulation**
(`../sim`): under known truth it credits genuinely causal exposures with ~0.98 sensitivity and
rejects confounded / reverse-caused / null exposures with high specificity.

## The battery
| Check | Threat targeted | "Passes" when |
|---|---|---|
| Primary survey/Cox | — | harmful & significant |
| Negative-control exposure | generic confounding | control's excess HR ≪ exposure's |
| Left-truncated landmark | reverse causation / early-death bias | retains ≥50% of the excess |
| Cross-sectional contrast | reverse causation (sign reversal) | not sign-reversed |
| E-value | unmeasured confounding | CI-bound E-value ≥ floor |
| Mixture index (optional) | single-metal co-exposure confounding | joint signal localises to the exposure |

## Usage
```python
import pandas as pd
from triangulate import Spec, run

df = pd.read_csv("cohort.csv")           # one row per participant
spec = Spec(
    time="t", event="death", exposure="zCadmium",
    negative_control="zMercury",          # pre-specified falsification exposure
    cross_section="prevalent_diabetes",   # optional reverse-causation contrast
    covariates=["age","sex","smoke","bmi","eGFR"],
    weight="svy_weight", cluster="psu",   # optional design-aware estimation
    landmarks=(2.0, 5.0),
)
result = run(df, spec)
print(result["verdict"])      # 'robust-causal-candidate' or 'fragile-or-spurious'
print(result["rationale"])    # human-readable reasons
print(result["checks"])       # every HR/OR/E-value
```
Standardise continuous exposures to 1-SD of the log beforehand for comparable, cross-cohort HRs.

## Install / deps
Pure Python: `pandas`, `numpy`, `lifelines`, `statsmodels`. MIT-licensed.

## Tests
`python test_triangulate.py` — verifies, on simulated cohorts of known truth, that the protocol
credits the causal one and rejects the three spurious ones (7/7).
