"""Out-of-family stress test: a NON-proportional-hazards log-logistic AFT generator.

The headline operating characteristics, the plasmode and the randomized ensemble all draw survival
from an exponential proportional-hazards (PH) model -- the family the protocol was tuned on. A Cox
battery is semiparametric, so a Weibull baseline with constant shape (still PH) is a trivial test,
and letting the Weibull SHAPE depend on the exposure manufactures a spurious exposure--survival
association in every world (including the null), which is a DGP artefact, not a protocol failure.

We therefore draw survival from a log-logistic ACCELERATED-FAILURE-TIME (AFT) model: the baseline is
log-logistic (a hazard that rises then falls -- strongly non-PH), and the exposure/covariates act by
multiplicatively accelerating time, T = T0 * exp(-eta*lp). This is genuinely outside the
exponential-PH family and violates the proportional-hazards assumption the primary/landmark clauses
lean on, yet -- crucially -- the exposure has NO effect on survival when its coefficient is zero, so
the confounded/reverse/null worlds stay genuinely spurious. We then ask whether the protocol's
sensitivity/specificity survive.
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sim_engine import CFG, COV, TAU, run_battery, classify
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
WORLDS_SPURIOUS = ["confounded", "reverse", "null"]


def aft_sim(scenario, n, rng, scale=11.0, shape=2.0, eta=0.55):
    """Mirror sim_engine.simulate() EXCEPT survival is a non-PH log-logistic AFT model."""
    c = dict(CFG[scenario])
    age = rng.normal(0, 1, n); sex = rng.binomial(1, 0.5, n).astype(float); smoke = rng.binomial(1, 0.4, n).astype(float)
    U = rng.normal(0, 1, n)
    Xtrue = 0.3 * age + 0.2 * smoke + c["confU"] * U + rng.normal(0, 1, n)
    Xtrue = (Xtrue - Xtrue.mean()) / Xtrue.std()
    Z = c["zU"] * U + rng.normal(0, 1, n); Z = (Z - Z.mean()) / Z.std()
    lp = 0.5 * age + 0.3 * smoke + c["bU"] * U + c["bX"] * Xtrue      # same predictor as the exp sim
    V = rng.uniform(1e-9, 1 - 1e-9, n)
    T0 = scale * (V / (1 - V)) ** (1.0 / shape)                       # log-logistic baseline (non-PH)
    t_event = T0 * np.exp(-eta * lp)                                  # AFT: harmful lp -> shorter time
    t = np.minimum(t_event, TAU); death = (t_event <= TAU).astype(int)
    lpD = -1.0 + 0.4 * age + 0.3 * smoke + 0.4 * U + c["dX"] * Xtrue
    D = rng.binomial(1, 1 / (1 + np.exp(-lpD)))
    imp = np.exp(-t_event / 1.5); imp = (imp - imp.mean()) / imp.std()
    Xmeas = Xtrue + c["rc"] * imp - c["dDist"] * D
    Xmeas = (Xmeas - Xmeas.mean()) / Xmeas.std()
    return pd.DataFrame(dict(t=t, death=death, X=Xmeas, Z=Z, D=D, age=age, sex=sex, smoke=smoke))


def ph_pvalue(df):
    cph = CoxPHFitter().fit(df[["t", "death", "X"] + COV], "t", "death")
    return float(proportional_hazard_test(cph, df[["t", "death", "X"] + COV], time_transform="rank").summary.loc["X", "p"])


def op_chars(R=400, n=5000, seed=7, **kw):
    rng = np.random.default_rng(seed); res = {}; hrs = {}; phps = []
    for sc in ["causal"] + WORLDS_SPURIOUS:
        cc = 0; hr_acc = []
        for _ in range(R):
            df = aft_sim(sc, n, rng, **kw); b = run_battery(df)
            cc += (classify(b)["verdict"] == "causal"); hr_acc.append(b["hr"])
            if sc == "causal" and len(phps) < 40:
                try: phps.append(ph_pvalue(df))
                except Exception: pass
        res[sc] = cc / R; hrs[sc] = float(np.mean(hr_acc))
    sens = res["causal"]; spec = 1 - np.mean([res[s] for s in WORLDS_SPURIOUS])
    return sens, spec, res, hrs, (float(np.median(phps)) if phps else float("nan"))


if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(1)
    for sc in ["causal", "null"]:
        d = aft_sim(sc, 8000, rng)
        cph = CoxPHFitter().fit(d[["t", "death", "X"] + COV], "t", "death")
        print(f"sanity {sc:10s}: death={d.death.mean():.2f}  med-FU={d.t.median():.1f}  primary HR={np.exp(cph.params_['X']):.2f}  PH p={ph_pvalue(d):.4f}")
    if "--tune" in sys.argv:
        sys.exit(0)
    sens, spec, res, hrs, phmed = op_chars()
    print(f"\nNON-PH log-logistic AFT operating characteristics (R=400/world, n=5000):")
    print(f"  sensitivity = {sens:.3f}   specificity = {spec:.3f}   (PH-test p median = {phmed:.4f}, PH violated)")
    print(f"  per-world credit rate: " + "  ".join(f"{k}={v:.3f}" for k, v in res.items()))
    print(f"  mean primary HR:       " + "  ".join(f"{k}={v:.2f}" for k, v in hrs.items()))
    import json
    json.dump(dict(sensitivity=sens, specificity=spec, per_world=res, mean_hr=hrs, ph_p_median=phmed),
              open("results_nonph_weibull.json", "w"), indent=2)
    print("\nsaved results_nonph_weibull.json")
