"""Run the triangulation protocol on NHANES III (1988-1994) -- a free temporal replication of
the cadmium(urinary body-burden)+lead -> mortality association at the higher 1988-94 exposure.
No Hg negative control and no blood Cd in NHANES III, so this is the Tier-B (Cd-arm) replication:
primary Cox + left-truncated landmark + E-value + cross-sectional (prevalent diabetes) contrast.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tool"))
from triangulate import Spec, run
HERE = os.path.dirname(os.path.abspath(__file__))

d = pd.read_csv(os.path.join(HERE, "analytic_nhanes3.csv"))

def z(s): s = pd.to_numeric(s, errors="coerce"); return (s - s.mean()) / s.std()
def zlog(s): s = np.log(pd.to_numeric(s, errors="coerce").clip(lower=1e-3)); return (s - s.mean()) / s.std()

d["zCd"] = zlog(d.UCD)        # urinary cadmium (body burden)
d["zPb"] = zlog(d.PB)         # blood lead
d["age_z"] = z(d.AGE); d["pir_z"] = z(d.PIR.replace({888888: np.nan, 999999: np.nan}))
d["male"] = (d.SEX == 1).astype(float)
for r in [2, 3, 4]: d[f"race_{r}"] = (d.RACE == r).astype(float)
for s in [1, 2]: d[f"smoke_{s}"] = (d.SMOKE == s).astype(float)
COV = ["age_z", "male", "race_2", "race_3", "race_4", "smoke_1", "smoke_2", "pir_z"]

print(f"N={len(d)}  all-cause deaths={int(d.death.sum())}  CVD deaths={int(d.cvd.sum())}  median FU={d.t.median():.1f}y\n")

def analyse(expo, label):
    sp = Spec(time="t", event="death", exposure=expo, covariates=COV,
              cross_section="diabetes", landmarks=(2.0, 5.0))
    r = run(d.dropna(subset=[expo] + COV + ["t", "death"]).copy(), sp)
    c = r["checks"]
    print(f"== {label} (per 1-SD log) ==")
    print(f"   all-cause Cox HR   = {c['primary']['hr']:.2f} ({c['primary']['lo']:.2f}-{c['primary']['hi']:.2f})  p={c['primary']['p']:.1e}")
    print(f"   2y landmark HR     = {c['landmark'][2.0]['hr']:.2f} ({c['landmark'][2.0]['lo']:.2f}-{c['landmark'][2.0]['hi']:.2f})")
    print(f"   5y landmark HR     = {c['landmark'][5.0]['hr']:.2f} ({c['landmark'][5.0]['lo']:.2f}-{c['landmark'][5.0]['hi']:.2f})")
    print(f"   E-value (CI bound) = {c['evalue']['point']:.2f} ({c['evalue']['ci_bound']:.2f})")
    print(f"   cross-sec OR (diabetes) = {c['cross_section']['orr']:.2f}")
    print(f"   verdict: {r['verdict']}  [{'; '.join(r['rationale'])}]\n")
    return c

cd = analyse("zCd", "Urinary cadmium")
pb = analyse("zPb", "Blood lead")

# CVD-specific primary (same tool, swap event)
for expo, lab in [("zCd", "Urinary cadmium"), ("zPb", "Blood lead")]:
    sp = Spec(time="t", event="cvd", exposure=expo, covariates=COV, landmarks=())
    r = run(d.dropna(subset=[expo] + COV + ["t", "cvd"]).copy(), sp)
    c = r["checks"]["primary"]
    print(f"CVD-specific {lab}: HR={c['hr']:.2f} ({c['lo']:.2f}-{c['hi']:.2f})")
