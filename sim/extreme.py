"""'To the extreme' robustness analyses demanded of any serious methods paper:
  (A) THRESHOLD SENSITIVITY -- are the operating characteristics an artefact of the three
      hand-chosen gate thresholds (landmark-retain 0.5, negctrl 0.4, E-value 1.25)? Sweep them.
  (B) COLLIDER / SELECTION-BIAS adversary -- a 5th threat NONE of the checks target. Honest test
      of a known blind spot: does the battery wrongly credit a selection-induced association?
  (C) NAIVE-vs-ARTEFACT-STRENGTH -- is the naive analyst's poor 0.32 specificity cherry-picked?
      Show protocol vs naive specificity as a function of confounding strength.
"""
import json, numpy as np, pandas as pd
from sim_engine import simulate, run_battery, cox_hr, logit_or, evalue, COV, LANDMARK
OUT = {}

# ---------- (A) threshold sensitivity ----------
def classify_param(b, lm_retain, negctrl_frac, eval_floor):
    xs = max(b["hr"] - 1.0, 0.01)
    primary = b["lo"] > 1.0
    persists = (b["llo"] > 1.0) and ((b["lhr"] - 1.0) >= lm_retain * xs)
    negclean = (b["zhr"] - 1.0) <= negctrl_frac * xs
    evok = b["evalue_ci"] >= eval_floor
    return primary and persists and negclean and evok

def op_chars(lm_retain, negctrl_frac, eval_floor, R=300, n=5000, seed=3):
    rng = np.random.default_rng(seed)
    res = {}
    for sc in ["causal", "confounded", "reverse", "null"]:
        cc = sum(classify_param(run_battery(simulate(sc, n, rng)), lm_retain, negctrl_frac, eval_floor) for _ in range(R)) / R
        res[sc] = cc
    sens = res["causal"]; spec = 1 - np.mean([res["confounded"], res["reverse"], res["null"]])
    return sens, spec

print("(A) threshold sensitivity")
grid = []
for lm in [0.4, 0.5, 0.6]:
    for nc in [0.3, 0.4, 0.5]:
        for ev in [1.15, 1.25, 1.40]:
            s, sp = op_chars(lm, nc, ev)
            grid.append((lm, nc, ev, s, sp))
sens_all = [g[3] for g in grid]; spec_all = [g[4] for g in grid]
OUT["threshold_sensitivity"] = dict(n_configs=len(grid),
    sens_min=min(sens_all), sens_max=max(sens_all), spec_min=min(spec_all), spec_max=max(spec_all))
print(f"   {len(grid)} threshold configs: sensitivity {min(sens_all):.2f}-{max(sens_all):.2f}, "
      f"specificity {min(spec_all):.2f}-{max(spec_all):.2f}")

# ---------- (B) collider / selection-bias adversary ----------
def simulate_collider(n, rng, s=1.0, bF=0.8):
    """No causal X effect. A frailty F raises the hazard; selection into the cohort is a COLLIDER
    of X and F (S more likely when X high AND F low), which induces a spurious harmful X->death in
    the selected sample. The negative control Z is independent of the selection -> stays clean,
    so the battery has no check that targets this."""
    m = n * 4                                                  # oversample, then select
    age = rng.normal(0,1,m); sex = rng.binomial(1,.5,m).astype(float); smoke = rng.binomial(1,.4,m).astype(float)
    X = rng.normal(0,1,m); F = rng.normal(0,1,m)               # X has NO effect; F = frailty
    Z = rng.normal(0,1,m)                                      # negative control, independent
    lp = 0.5*age + 0.3*smoke + bF*F                            # hazard: covs + frailty, NO X
    te = rng.exponential(1.0/(0.03*np.exp(lp))); t = np.minimum(te, 12.0); death = (te<=12.0).astype(int)
    psel = 1/(1+np.exp(-(s*X - s*F)))                          # collider: select high X, low F
    sel = rng.random(m) < psel
    idx = np.where(sel)[0][:n]
    lpD = -1.0 + 0.4*age + 0.3*smoke
    D = rng.binomial(1, 1/(1+np.exp(-lpD)))
    d = pd.DataFrame(dict(t=t,death=death,X=(X-X.mean())/X.std(),Z=(Z-Z.mean())/Z.std(),D=D,age=age,sex=sex,smoke=smoke))
    return d.iloc[idx].reset_index(drop=True)

print("(B) collider / selection-bias adversary (a threat no check targets)")
rng = np.random.default_rng(5); cc=0; prim=0; lmcaught=0; R=300
from sim_engine import classify
for _ in range(R):
    b = run_battery(simulate_collider(5000, rng)); v = classify(b)
    cc += (v["verdict"]=="causal"); prim += b["lo"]>1.0; lmcaught += (not v["persists_lm"])
OUT["collider"] = dict(false_causal=cc/R, primary_sig=prim/R, landmark_attenuates=lmcaught/R)
print(f"   spurious primary significant in {prim/R:.2f}; protocol mis-credits causal in {cc/R:.2f}; "
      f"landmark happens to attenuate in {lmcaught/R:.2f}")

# ---------- (C) naive vs protocol specificity across confounding strength ----------
def naive_causal(b): return b["lo"] > 1.0
print("(C) protocol vs naive specificity across confounding strength")
OUT["naive_vs_strength"] = {}
rng = np.random.default_rng(7)
for cs in [0.3, 0.5, 0.7, 0.9]:
    R=300; pc=0; nc=0
    for _ in range(R):
        b = run_battery(simulate("confounded", 5000, rng, {"confU": cs, "bU": cs, "zU": min(0.95, cs+0.2)}))
        pc += (classify(b)["verdict"]=="causal"); nc += naive_causal(b)
    OUT["naive_vs_strength"][cs] = dict(protocol_spec=1-pc/R, naive_spec=1-nc/R)
    print(f"   confounding={cs:.1f}: protocol specificity={1-pc/R:.2f}  naive specificity={1-nc/R:.2f}")

json.dump(OUT, open("results_extreme.json","w"), indent=2)
print("\nsaved results_extreme.json")
