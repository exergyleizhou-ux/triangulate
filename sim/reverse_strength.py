import json, numpy as np
from sim_engine import simulate, run_battery, classify
OUT = {}; rng = np.random.default_rng(11); R = 400; n = 5000
print("rc   (dDist)  primaryHR  protocol_spec  naive_spec  landmark_catches")
for rc in [0.3, 0.5, 0.7, 0.9, 1.1]:
    dd = 0.9 * rc / 1.1
    pc = nc = lm = 0; hrs = []
    for _ in range(R):
        df = simulate("reverse", n, rng, {"rc": rc, "dDist": dd})
        b = run_battery(df); v = classify(b)
        pc += (v["verdict"] == "causal"); nc += (b["lo"] > 1.0)
        lm += (not v["persists_lm"]); hrs.append(b["hr"])
    OUT[f"{rc:.1f}"] = dict(primary_hr=float(np.median(hrs)), protocol_spec=round(1-pc/R,3),
                            naive_spec=round(1-nc/R,3), landmark_catches=round(lm/R,3))
    print(f"{rc:.1f}  {dd:.2f}     {np.median(hrs):.2f}      {1-pc/R:.3f}        {1-nc/R:.3f}     {lm/R:.3f}")
json.dump(OUT, open("results_reverse_strength.json","w"), indent=2)
print("\nsaved results_reverse_strength.json  (R=400/cell, binomial MCSE <=0.025)")
