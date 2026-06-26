"""Design-effect robustness of the operating characteristics.

The headline operating characteristics use the simulation's model-based (iid, unweighted) Cox SE,
whereas the deployed tool runs a design-based, survey-weighted Cox whose robust SE is generally
WIDER (design effect DEFF = Var_design / Var_iid > 1; ~1.3-2.0 is typical for NHANES). Because the
decision rule gates significance (lo>1) on the SE, a wider SE can only lower sensitivity and can
never raise the false-positive rate. We therefore re-measure the operating characteristics with the
primary and landmark CIs (and the CI-bound E-value) widened by sqrt(DEFF), reconstructing each SE
from the reported CI. The negative-control and landmark-retention magnitude clauses use point
estimates and are unchanged. This quantifies the gap between the validated (iid-SE) and deployed
(design-based-SE) inference procedures.
"""
import numpy as np
from sim_engine import simulate, run_battery, evalue
WORLDS_SPURIOUS = ["confounded", "reverse", "null"]

def classify_inflated(b, deff):
    f = np.sqrt(deff)
    xs = max(b["hr"] - 1.0, 0.01)
    # reconstruct iid SE from the reported CI, then widen by sqrt(DEFF)
    se_x = (np.log(b["hr"]) - np.log(b["lo"])) / 1.96
    lo_x = np.exp(np.log(b["hr"]) - 1.96 * f * se_x)
    se_l = (np.log(b["lhr"]) - np.log(b["llo"])) / 1.96
    lo_l = np.exp(np.log(b["lhr"]) - 1.96 * f * se_l)
    ev_ci = evalue(b["hr"], lo_x, b["hi"])[1]             # E-value at the widened lower bound (harmful HR)
    primary_sig   = lo_x > 1.0
    persists_lm   = (lo_l > 1.0) and ((b["lhr"] - 1.0) >= 0.5 * xs)
    negctrl_clean = (b["zhr"] - 1.0) <= 0.4 * xs          # point estimate -> unchanged
    evalue_ok     = ev_ci >= 1.25
    return primary_sig and persists_lm and negctrl_clean and evalue_ok

def oc(deff, R=400, n=5000, seed=2):
    rng = np.random.default_rng(seed); res = {}
    for sc in ["causal"] + WORLDS_SPURIOUS:
        c = 0
        for _ in range(R):
            b = run_battery(simulate(sc, n, rng))
            c += bool(classify_inflated(b, deff))
        res[sc] = c / R
    sens = res["causal"]; spec = 1 - np.mean([res[s] for s in WORLDS_SPURIOUS])
    return sens, spec, res

print(f"{'DEFF':>5s} {'sqrt':>5s} {'sens':>6s} {'spec':>6s}   {'conf':>5s} {'rev':>5s} {'null':>5s}")
for deff in [1.0, 1.3, 1.5, 2.0]:
    sens, spec, res = oc(deff)
    print(f"{deff:5.1f} {np.sqrt(deff):5.2f} {sens:6.2f} {spec:6.2f}   "
          f"{res['confounded']:5.2f} {res['reverse']:5.2f} {res['null']:5.2f}")
