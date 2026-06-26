"""Head-to-head: the triangulation PROTOCOL vs named existing single-strategy methods, run as
competing CLASSIFIERS on the same known-truth worlds, plus combined-threat worlds. This measures
the protocol's marginal contribution OVER current best practice, not just over a strawman.

Competing classifiers (each credits 'causal' under its own rule, given the same battery outputs):
  naive          : a single significant harmful primary model (b.lo > 1).
  evalue_only    : VanderWeele E-value gate alone (CI-bound E-value >= 1.5).
  negctrl_calib  : Schuemie/Lipsitch-style negative-control bias calibration -- subtract the
                   negative control's log-HR (an estimate of residual confounding bias) from the
                   primary, credit if the bias-corrected CI still excludes the null.
  speccurve      : specification-curve / multiverse -- refit the primary under many covariate
                   subsets; credit if the harmful effect is significant in >= 60% of specs.
  protocol       : the full triangulation battery (negative control + landmark + E-value gate).
"""
import json, numpy as np, pandas as pd
from sim_engine import simulate, run_battery, classify, cox_hr, COV, LANDMARK
WORLDS_SPURIOUS = ["confounded", "reverse", "null"]

def naive(b, df):        return b["lo"] > 1.0
def evalue_only(b, df):  return b["evalue_ci"] >= 1.5
def negctrl_calib(b, df):
    # bias-corrected log-HR = primary - negative-control (the control estimates residual bias);
    # approximate SE by the primary's (conservative). Credit if corrected CI excludes null.
    import numpy as _np
    bx = _np.log(b["hr"]); bz = _np.log(b["zhr"]); se = (_np.log(b["hi"]) - _np.log(b["lo"])) / (2*1.96)
    corr = bx - bz
    return (corr - 1.96*se) > 0.0
def speccurve(b, df, K=15, thresh=0.6, seed=0):
    rng = np.random.default_rng(seed); sig = 0; ok = 0
    for _ in range(K):
        k = rng.integers(1, len(COV)+1); cols = list(rng.choice(COV, size=k, replace=False))
        import numpy as _np
        from sim_engine import _fast_cox
        Xmat = df[["X"]+cols].to_numpy(float)
        try:
            bb, se = _fast_cox(Xmat, df["t"].to_numpy(float), df["death"].to_numpy(float))
            lo = _np.exp(bb[0]-1.96*se[0]); ok += 1; sig += (lo > 1.0)
        except Exception: pass
    return ok > 0 and (sig/ok) >= thresh
def protocol(b, df):     return classify(b)["verdict"] == "causal"
def negctrl_landmark(b, df):
    # FAIR COMPOSITE: conjoin the two complementary discriminating checks -- a negative-control
    # clause AND a landmark-persistence clause (with primary significance), i.e. the full protocol
    # WITHOUT its E-value gate. This measures whether the discrimination comes from conjoining the
    # two complementary checks (not from any single check, and not from the E-value clause).
    c = classify(b)
    return c["primary_sig"] and c["negctrl_clean"] and c["persists_lm"]

METHODS = {"naive": naive, "evalue_only": evalue_only, "negctrl_calib": negctrl_calib,
           "speccurve": speccurve, "negctrl_landmark": negctrl_landmark, "protocol": protocol}

# combined-threat worlds (override the canonical params to co-occur)
COMBINED = {
    "confounded+reverse":  ("confounded", {"rc": 0.9, "dDist": 0.7}),
    "confounded+causal":   ("causal",     {"confU": 0.6, "bU": 0.6, "zU": 0.8}),   # true effect + real confounding
}

def op_chars(method_fn, R=400, n=5000, seed=2):
    rng = np.random.default_rng(seed); res = {}
    for sc in ["causal"] + WORLDS_SPURIOUS:
        cc = 0
        for _ in range(R):
            df = simulate(sc, n, rng); b = run_battery(df); cc += bool(method_fn(b, df))
        res[sc] = cc / R
    sens = res["causal"]; spec = 1 - np.mean([res[s] for s in WORLDS_SPURIOUS])
    return sens, spec, res

OUT = {"by_method": {}}
print("=== competing classifiers on the canonical worlds ===")
print(f"{'method':14s} {'sens':>6s} {'spec':>6s}   {'conf':>5s} {'rev':>5s} {'null':>5s}")
for name, fn in METHODS.items():
    sens, spec, res = op_chars(fn)
    OUT["by_method"][name] = dict(sensitivity=sens, specificity=spec, per_world=res)
    print(f"{name:14s} {sens:6.2f} {spec:6.2f}   {res['confounded']:5.2f} {res['reverse']:5.2f} {res['null']:5.2f}")

print("\n=== combined-threat worlds (false-causal rate; lower=better) ===")
OUT["combined"] = {}
rng = np.random.default_rng(9)
for cname, (base_sc, ov) in COMBINED.items():
    OUT["combined"][cname] = {}
    row = []
    for name, fn in METHODS.items():
        R=300; cc=0
        rng2 = np.random.default_rng(13)
        for _ in range(R):
            df = simulate(base_sc, 5000, rng2, ov); b = run_battery(df); cc += bool(fn(b, df))
        val = cc/R
        OUT["combined"][cname][name] = val
        row.append(f"{name}={val:.2f}")
    tag = "(true effect -> want HIGH)" if base_sc == "causal" else "(spurious -> want LOW)"
    print(f"  {cname} {tag}: " + "  ".join(row))

json.dump(OUT, open("results_comparison.json", "w"), indent=2)
print("\nsaved results_comparison.json")
