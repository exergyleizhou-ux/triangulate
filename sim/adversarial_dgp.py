"""Break the 'you designed the worlds to match your checks' circularity with DGPs I did NOT
hand-build:
  (1) PLASMODE: real NHANES covariate matrix (real joint distribution / real confounding
      structure) + a SYNTHETIC exposure and a KNOWN imposed effect. The nuisance structure is
      empirical, not normal draws I tuned.
  (2) RANDOMIZED ENSEMBLE: per replicate, draw the DGP structure at random (truth, confounding
      strength, negative-control share -- sometimes WEAK, reverse strength, PH violation), so no
      world is hand-picked to be exactly what a check targets. Report operating characteristics
      marginalized over the random structure (honestly below the isolated-threat headline).
"""
import os, json, numpy as np, pandas as pd
from sim_engine import run_battery, classify, evalue
NH = os.path.expanduser("~/Desktop/bos/nhanes-paper/analysis/analytic_metals.csv")
OUT = {}

# ---------- (1) PLASMODE on real NHANES covariates ----------
if not os.path.exists(NH):
    import sys
    sys.exit(f"[skip] plasmode needs the companion NHANES covariate matrix at {NH}; "
             "obtain it from the companion study and re-run.")
real = pd.read_csv(NH)
def z(s): s = pd.to_numeric(s, errors="coerce"); return (s - s.mean()) / s.std()
RC = pd.DataFrame({c: z(real[c]) for c in ["AGE", "BMI", "PIR"]}).dropna().reset_index(drop=True)
RC["sex"] = pd.to_numeric(real["SEX"], errors="coerce").reset_index(drop=True).reindex(RC.index).fillna(1).map(lambda v:1.0 if v==1 else 0.0)
RC["smoke"] = (pd.to_numeric(real["SMOKE"], errors="coerce").reset_index(drop=True).reindex(RC.index).fillna(0) > 0).astype(float)
print(f"plasmode: real NHANES covariate rows = {len(RC)}")

def plasmode(bX, rng, cU=0.5, zU=0.7):
    s = RC.sample(5000, replace=True, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    age = s.AGE.values; sex = s.sex.values; smoke = s.smoke.values
    U = ((s.BMI.values + s.PIR.values))                          # HIDDEN confounder = REAL BMI+PIR (not adjusted)
    U = (U - U.mean())/U.std()
    X = cU*U + 0.3*age + rng.normal(0,1,len(s)); X = (X-X.mean())/X.std()   # confounded by real U (strength cU)
    Z = zU*U + rng.normal(0,1,len(s)); Z = (Z-Z.mean())/Z.std()            # neg control shares U
    lp = 0.4*age + 0.3*smoke + 0.7*U + bX*X
    te = rng.exponential(1/(0.03*np.exp(lp))); t = np.minimum(te,12.0); death=(te<=12).astype(int)
    D = rng.binomial(1, 1/(1+np.exp(-(-1+0.4*age+0.3*smoke+0.4*U))))
    return pd.DataFrame(dict(t=t,death=death,X=X,Z=Z,D=D,age=age,sex=sex,smoke=smoke))

print("(1) PLASMODE (real-covariate confounding; imposed known effect)")
rng = np.random.default_rng(4)
cases = [("true causal, MILD real-confounding (HR 1.45)", np.log(1.45), 0.15, 0.30, "plasmode_causal_mild"),
         ("true causal, STRONG real-confounding (HR 1.45)", np.log(1.45), 0.50, 0.70, "plasmode_causal_strong"),
         ("null + real-confounding", 0.0, 0.50, 0.70, "plasmode_confounded_null")]
for label, bX, cU, zU, key in cases:
    R=400; cc=0
    for _ in range(R): cc += (classify(run_battery(plasmode(bX, rng, cU, zU)))["verdict"]=="causal")
    OUT[key] = cc/R
    metric = "sensitivity" if bX>0 else "false-causal"
    print(f"   {label}: {metric} = {cc/R:.3f}")

# ---------- (2) RANDOMIZED ENSEMBLE ----------
from sim_engine import simulate
print("(2) RANDOMIZED ENSEMBLE (DGP structure drawn at random per replicate)")
rng = np.random.default_rng(6); R=1500; sens_hit=sens_n=spec_hit=spec_n=0
for _ in range(R):
    truth = rng.choice(["causal","confounded","reverse","null"], p=[.4,.25,.2,.15])
    confU = rng.uniform(0.0, 0.9); zU = rng.uniform(0.25, 0.95)   # control share sometimes WEAK
    ov = {"confU": confU, "bU": confU*rng.uniform(0.7,1.0), "zU": zU}
    if truth=="causal": ov["bX"]=np.log(rng.uniform(1.25,1.5))
    if truth=="reverse": ov.update(rc=rng.uniform(0.6,1.3), dDist=rng.uniform(0.5,1.0), bX=0.0)
    if truth=="null": ov.update(confU=0.0,bU=0.0,zU=0.0,bX=0.0)
    df = simulate(truth if truth!="causal" else "causal", 5000, rng, ov)
    v = classify(run_battery(df))["verdict"]=="causal"
    if truth=="causal": sens_n+=1; sens_hit+=v
    else: spec_n+=1; spec_hit+=(not v)
OUT["randomized_ensemble"] = dict(sensitivity=sens_hit/sens_n, specificity=spec_hit/spec_n,
                                  n_causal=sens_n, n_spurious=spec_n)
print(f"   over random structure: sensitivity={sens_hit/sens_n:.3f}  specificity={spec_hit/spec_n:.3f}")
print(f"   (specificity < the isolated-threat 1.00 because weak negative controls are mixed in)")

json.dump(OUT, open("results_adversarial_dgp.json","w"), indent=2)
print("\nsaved results_adversarial_dgp.json")
