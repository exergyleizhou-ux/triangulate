"""Operating-characteristics study for the causal-triangulation protocol.

Reports, over many replicates under known truth:
  - sensitivity  = P(verdict=causal | truly causal)
  - specificity  = P(verdict=spurious | truly spurious: confounded / reverse / null)
  - per-check detection (which check catches which threat)
  - head-to-head vs a NAIVE analyst who credits any single significant primary model
  - sweeps over sample size, true effect size, and confounding strength
Saves everything to results.json. No fabricated numbers: every value is computed here.
"""
import json, time, numpy as np
from sim_engine import simulate, run_battery, classify, CFG, LOG

def one_rep(scenario, n, rng, override=None):
    b = run_battery(simulate(scenario, n, rng, override))
    v = classify(b)
    naive = b["lo"] > 1.0                         # naive analyst: one significant harmful primary model
    return v, naive

def rates(scenario, n, R, seed, override=None):
    rng = np.random.default_rng(seed)
    keys = ["verdict_causal", "naive_causal", "primary_sig", "persists_lm", "negctrl_clean", "xsec_consist", "evalue_ok"]
    acc = {k: 0 for k in keys}
    for _ in range(R):
        v, naive = one_rep(scenario, n, rng, override)
        acc["verdict_causal"] += (v["verdict"] == "causal")
        acc["naive_causal"] += naive
        for k in ["primary_sig", "persists_lm", "negctrl_clean", "xsec_consist", "evalue_ok"]:
            acc[k] += v[k]
    return {k: acc[k] / R for k in keys}

OUT = {}
t0 = time.time()

# 1) MAIN operating characteristics
print("[1/4] main operating characteristics ...", flush=True)
R, N = 1000, 5000
OUT["main"] = {"R": R, "n": N, "scenarios": {}}
for sc in ["causal", "confounded", "reverse", "null"]:
    r = rates(sc, N, R, seed=100)
    OUT["main"]["scenarios"][sc] = r
    print(f"   {sc:11s} protocol-credits-causal={r['verdict_causal']:.3f}  naive-credits-causal={r['naive_causal']:.3f}", flush=True)
m = OUT["main"]["scenarios"]
OUT["main"]["sensitivity"] = m["causal"]["verdict_causal"]
OUT["main"]["specificity"] = 1 - np.mean([m["confounded"]["verdict_causal"], m["reverse"]["verdict_causal"], m["null"]["verdict_causal"]])
OUT["main"]["naive_specificity"] = 1 - np.mean([m["confounded"]["naive_causal"], m["reverse"]["naive_causal"], m["null"]["naive_causal"]])
print(f"   => protocol sensitivity={OUT['main']['sensitivity']:.3f}  specificity={OUT['main']['specificity']:.3f}  (naive specificity={OUT['main']['naive_specificity']:.3f})", flush=True)

# 2) SAMPLE-SIZE sweep
print("[2/4] sample-size sweep ...", flush=True)
OUT["sample_size"] = {}
for n in [1500, 3000, 6000, 12000]:
    OUT["sample_size"][n] = {sc: rates(sc, n, 500, seed=200)["verdict_causal"] for sc in CFG}
    print(f"   n={n:5d}  causal={OUT['sample_size'][n]['causal']:.3f}  conf={OUT['sample_size'][n]['confounded']:.3f}  rev={OUT['sample_size'][n]['reverse']:.3f}  null={OUT['sample_size'][n]['null']:.3f}", flush=True)

# 3) EFFECT-SIZE sweep (causal): can the protocol detect smaller true effects?
print("[3/4] effect-size sweep (causal) ...", flush=True)
OUT["effect_size"] = {}
for hr in [1.10, 1.20, 1.30, 1.45]:
    r = rates("causal", 5000, 600, seed=300, override={"bX": LOG(hr)})
    OUT["effect_size"][hr] = r["verdict_causal"]
    print(f"   true HR={hr:.2f}  protocol-credits-causal(sensitivity)={r['verdict_causal']:.3f}", flush=True)

# 4) CONFOUNDING-strength sweep (confounded): how strong must confounding be to be caught?
print("[4/4] confounding-strength sweep (confounded) ...", flush=True)
OUT["confounding"] = {}
for s in [0.3, 0.5, 0.7, 0.9]:
    r = rates("confounded", 5000, 500, seed=400, override={"bU": s, "confU": s, "zU": min(0.95, s + 0.2)})
    OUT["confounding"][s] = {"false_causal": r["verdict_causal"], "negctrl_flags": 1 - r["negctrl_clean"]}
    print(f"   conf-strength={s:.1f}  protocol-false-causal={r['verdict_causal']:.3f}  negctrl-flags={1-r['negctrl_clean']:.3f}", flush=True)

OUT["runtime_min"] = round((time.time() - t0) / 60, 1)
json.dump(OUT, open("results.json", "w"), indent=2)
print(f"\nDONE in {OUT['runtime_min']} min -> results.json", flush=True)
