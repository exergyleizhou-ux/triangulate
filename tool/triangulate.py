"""triangulate — an open-source implementation of the causal-triangulation protocol for
discriminating robust from spurious biomarker--mortality associations in a single cohort.

Given a participant-level table with a survival outcome, an exposure biomarker, a
pre-specified negative-control exposure, optional cross-sectional condition, and covariates,
it runs the discrimination battery (primary survey/Cox, negative control, left-truncated
landmark, cross-sectional contrast, E-value, and an optional mixture index) and returns a
structured verdict with the per-check reasoning.

This is the same protocol applied in the NHANES cadmium--mortality study and validated by
simulation (see ../sim). Pure-Python, depends only on pandas / numpy / lifelines /
statsmodels. MIT-licensed.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass, field, asdict
from lifelines import CoxPHFitter
import statsmodels.api as sm


def evalue(hr: float, lo: float, hi: float) -> tuple[float, float]:
    """VanderWeele & Ding E-value for a hazard ratio and the CI bound nearest the null.
    The CI-bound E-value is 1.0 whenever the interval crosses the null (lo<1<hi)."""
    g = lambda x: (x if x >= 1 else 1 / x) + np.sqrt((x if x >= 1 else 1 / x) * ((x if x >= 1 else 1 / x) - 1))
    if hr > 1:
        b = lo if lo > 1 else 1.0            # harmful: lower bound is nearest the null
    elif hr < 1:
        b = hi if hi < 1 else 1.0            # protective: upper bound is nearest the null
    else:
        b = 1.0
    return g(hr), (g(b) if b != 1.0 else 1.0)


@dataclass
class Spec:
    time: str                       # follow-up time column
    event: str                      # 0/1 death indicator
    exposure: str                   # primary exposure biomarker (continuous; pre-standardise to 1-SD if desired)
    covariates: list[str]
    negative_control: str | None = None   # pre-specified negative-control exposure
    cross_section: str | None = None       # 0/1 prevalent condition for the reverse-causation contrast
    weight: str | None = None              # survey weight (enables design-aware estimation)
    cluster: str | None = None             # PSU/cluster for cluster-robust variance
    landmarks: tuple[float, ...] = (2.0, 5.0)
    mixture: list[str] = field(default_factory=list)  # optional: quartile-scored metals for a mixture index
    # decision-rule thresholds (the operationalised triangulation logic, validated in ../sim)
    negctrl_frac: float = 0.4       # control "clean" if its excess HR < this fraction of the exposure's
    landmark_retain: float = 0.5    # landmark "persists" if it retains >= this fraction of the excess
    evalue_floor: float = 1.25      # CI-bound E-value must clear this


def _cox(df, expo, sp: Spec, entry=None):
    cols = [c for c in sp.covariates if df[c].nunique() > 1]
    use = df[[expo, sp.time, sp.event] + cols + ([sp.weight] if sp.weight else []) + ([sp.cluster] if sp.cluster else []) + ([entry] if entry else [])].dropna()
    kw = dict(show_progress=False)
    if sp.weight: kw["weights_col"] = sp.weight
    if sp.cluster: kw["cluster_col"] = sp.cluster; kw["robust"] = True
    cph = CoxPHFitter(penalizer=0.01).fit(use, sp.time, sp.event, entry_col=entry, **kw)
    s = cph.summary.loc[expo]
    return dict(hr=float(s["exp(coef)"]), lo=float(s["exp(coef) lower 95%"]), hi=float(s["exp(coef) upper 95%"]), p=float(s["p"]))


def _logit(df, expo, sp: Spec):
    cols = [c for c in sp.covariates if df[c].nunique() > 1]
    use = df[[sp.cross_section, expo] + cols].dropna()
    X = sm.add_constant(use[[expo] + cols])
    m = sm.Logit(use[sp.cross_section], X).fit(disp=0)
    return dict(orr=float(np.exp(m.params[expo])), p=float(m.pvalues[expo]))


def run(df: pd.DataFrame, sp: Spec) -> dict:
    out = {"checks": {}}
    prim = _cox(df, sp.exposure, sp); out["checks"]["primary"] = prim
    xs = max(prim["hr"] - 1.0, 0.01)
    ev, evci = evalue(prim["hr"], prim["lo"], prim["hi"]); out["checks"]["evalue"] = {"point": ev, "ci_bound": evci}

    primary_sig = prim["lo"] > 1.0

    negctrl_clean = None
    if sp.negative_control:
        nc = _cox(df, sp.negative_control, sp); out["checks"]["negative_control"] = nc
        negctrl_clean = (nc["hr"] - 1.0) <= sp.negctrl_frac * xs

    persists = None
    out["checks"]["landmark"] = {}
    for L in sp.landmarks:
        sub = df[df[sp.time] > L].copy(); sub["_entry"] = float(L)
        lm = _cox(sub, sp.exposure, sp, entry="_entry"); out["checks"]["landmark"][L] = lm
    if sp.landmarks:
        L0 = sp.landmarks[0]; lm0 = out["checks"]["landmark"][L0]
        persists = (lm0["lo"] > 1.0) and ((lm0["hr"] - 1.0) >= sp.landmark_retain * xs)

    xsec_consistent = None
    if sp.cross_section:
        cs = _logit(df, sp.exposure, sp); out["checks"]["cross_section"] = cs
        xsec_consistent = cs["orr"] >= 0.85   # not sign-reversed relative to a harmful prospective effect

    mix = None
    if sp.mixture:
        psi = sum(np.log(_cox(df, q, sp)["hr"]) for q in sp.mixture)
        out["checks"]["mixture_index_hr"] = float(np.exp(psi)); mix = np.exp(psi) > 1.0

    evalue_ok = evci >= sp.evalue_floor
    flags = dict(primary_sig=primary_sig, negctrl_clean=negctrl_clean, persists_landmark=persists,
                 xsec_consistent=xsec_consistent, evalue_ok=evalue_ok)
    out["flags"] = flags
    # GATE = primary + negative control + landmark persistence + E-value. The cross-sectional
    # contrast is a corroborating DIAGNOSTIC (reverse-causation signature), not a hard veto: a
    # genuinely causal exposure can show a reversed cross-section, so it is reported but does not
    # by itself condemn an exposure whose prospective signal persists under the landmark.
    gate = [flags["negctrl_clean"], flags["persists_landmark"], flags["evalue_ok"]]
    decisive = [v for v in gate if v is not None]
    out["verdict"] = "robust-causal-candidate" if primary_sig and all(decisive) else "fragile-or-spurious"
    out["rationale"] = _explain(flags, out["checks"])
    return out


def _explain(flags, checks):
    msgs = []
    if not flags["primary_sig"]:
        msgs.append("primary association is not significantly harmful")
    if flags.get("negctrl_clean") is False:
        msgs.append("negative control lights up — generic confounding not excluded")
    if flags.get("persists_landmark") is False:
        msgs.append("association attenuates under the landmark — reverse causation / early-death bias")
    if flags.get("xsec_consistent") is False:
        msgs.append("cross-sectional sign reverses — reverse causation signature")
    if flags.get("evalue_ok") is False:
        msgs.append("E-value too low — fragile to unmeasured confounding")
    return msgs or ["passes every pre-specified check"]


if __name__ == "__main__":
    # self-test on a synthetic causal cohort from the simulation engine
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
    from sim_engine import simulate
    rng = np.random.default_rng(7)
    for sc in ["causal", "confounded", "reverse", "null"]:
        df = simulate(sc, 6000, rng)
        sp = Spec(time="t", event="death", exposure="X", negative_control="Z",
                  cross_section="D", covariates=["age", "sex", "smoke"], landmarks=(2.0,))
        r = run(df, sp)
        print(f"[{sc:11s}] verdict={r['verdict']:26s} | {'; '.join(r['rationale'])}")
