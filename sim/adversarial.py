"""Adversarial / off-design sweeps demanded by verification: the headline operating
characteristics are CONDITIONAL. Two cases the main study never tested:
  (A) weak negative control: confounded world, but the control shares the confounder LESS than
      the exposure (zU < confU) -- the realistic case. Specificity should collapse.
  (B) causal-WITH-confounding: a genuinely causal exposure that ALSO carries shared confounding.
      Negative-control logic cannot separate 'real effect + confounding' from 'pure confounding',
      so sensitivity should collapse -- a fundamental limitation, stated honestly.
Also (C) a mixed confounding+reverse adversary the battery was not designed around.
"""
import json, numpy as np
from sim_engine import simulate, run_battery, classify, LOG

def rate(scenario, override, R=500, n=5000, seed=11):
    rng = np.random.default_rng(seed); cc = 0; negflag = 0
    for _ in range(R):
        v = classify(run_battery(simulate(scenario, n, rng, override)))
        cc += (v["verdict"] == "causal"); negflag += (not v["negctrl_clean"])
    return cc / R, negflag / R

OUT = {}

# (A) weak-control confounding: confounded world (confU=0.9,bU=0.85), vary control sharing zU
print("(A) weak negative control -> specificity collapses")
OUT["weak_control"] = {}
for zU in [0.2, 0.4, 0.6, 0.9]:
    fc, nf = rate("confounded", {"zU": zU})
    OUT["weak_control"][zU] = {"false_causal": fc, "negctrl_flags": nf}
    print(f"   zU={zU:.1f} (vs confU=0.9): false-causal={fc:.3f}  negctrl-flags={nf:.3f}")

# (B) causal WITH confounding: causal world, inject confounding (confU=bU), control tracks it
print("(B) causal exposure + shared confounding -> sensitivity collapses (true signal discarded)")
OUT["causal_with_confounding"] = {}
for c in [0.15, 0.3, 0.5, 0.7, 0.9]:
    sens, nf = rate("causal", {"confU": c, "bU": c, "zU": min(0.95, c + 0.2)})
    OUT["causal_with_confounding"][c] = {"sensitivity": sens, "negctrl_flags": nf}
    print(f"   confounding={c:.2f}: sensitivity={sens:.3f}  negctrl-flags-the-true-effect={nf:.3f}")

# (C) mixed adversary: no causal effect, BOTH confounding AND reverse causation present
print("(C) mixed confounding+reverse adversary (off-design)")
mix_sens, _ = rate("confounded", {"rc": 0.8, "dDist": 0.7})  # confounded + add reverse distortions
OUT["mixed_adversary_false_causal"] = mix_sens
print(f"   false-causal under mixed confounding+reverse = {mix_sens:.3f}")

json.dump(OUT, open("results_adversarial.json", "w"), indent=2)
print("\nsaved results_adversarial.json")
