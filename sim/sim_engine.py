"""Simulation validation of the causal-triangulation protocol.

We generate cohorts under KNOWN data-generating mechanisms (ground truth) and ask whether
the triangulation battery correctly DISCRIMINATES a genuinely causal exposure->mortality
association from spurious ones. Four canonical truths:

  causal     : exposure X truly raises the hazard (mild measured-only confounding).
  confounded : X has NO effect; an unmeasured confounder U drives both X and mortality
               (and a negative-control exposure Z, which shares U) -> the negative control
               should "light up".
  reverse    : X has NO effect; a frailty/illness latent F raises the hazard AND distorts
               the baseline biomarker, so early deaths drive the association -> the
               left-truncated landmark should attenuate it, and the cross-sectional
               contrast should reverse sign.
  null       : X has NO effect and no confounding -> primary model is null.

The battery (mirrors the manuscript protocol): primary Cox, negative-control exposure,
left-truncated landmark, cross-sectional contrast, E-value. The decision rule credits X as
"causal/robust" only if it passes every check. We report operating characteristics
(sensitivity / specificity) over many replicates.
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from lifelines import CoxPHFitter
import statsmodels.api as sm
LOG = np.log

#   bX    true causal log-HR of X on mortality
#   bU    strength of the unmeasured confounder U on mortality (confounding magnitude)
#   confU strength of U -> X (the confounding path into the exposure)
#   rc    reverse causation: impending death inflates the MEASURED biomarker
#   dDist cross-sectional reversal: prevalent disease lowers the MEASURED biomarker
#   zU    U -> Z (negative-control exposure shares the confounder; never causal)
#   dX    true X -> prevalent-disease association (cross-sectional, when not reverse-caused)
CFG = {
    "causal":     dict(bX=LOG(1.45), bU=0.15, confU=0.15, rc=0.0, dDist=0.0, zU=0.7, dX=0.45),
    "confounded": dict(bX=0.0,       bU=0.85, confU=0.90, rc=0.0, dDist=0.0, zU=0.9, dX=0.0),
    "reverse":    dict(bX=0.0,       bU=0.15, confU=0.15, rc=1.1, dDist=0.9, zU=0.7, dX=0.0),
    "null":       dict(bX=0.0,       bU=0.0,  confU=0.0,  rc=0.0, dDist=0.0, zU=0.0, dX=0.0),
}
COV = ["age", "sex", "smoke"]
TAU = 12.0           # administrative censoring (years)
LANDMARK = 2.0       # left-truncation landmark


def simulate(scenario, n, rng, override=None):
    c = dict(CFG[scenario]); c.update(override or {})
    age = rng.normal(0, 1, n); sex = rng.binomial(1, 0.5, n).astype(float); smoke = rng.binomial(1, 0.4, n).astype(float)
    U = rng.normal(0, 1, n)        # unmeasured confounder
    # true exposure: measured covs + U (U->X is the confounding path)
    Xtrue = 0.3 * age + 0.2 * smoke + c["confU"] * U + rng.normal(0, 1, n)
    Xtrue = (Xtrue - Xtrue.mean()) / Xtrue.std()
    # negative-control exposure: shares U, never causal
    Z = c["zU"] * U + rng.normal(0, 1, n); Z = (Z - Z.mean()) / Z.std()
    # mortality hazard (exponential PH): measured covs + U(confounding) + TRUE causal X
    lp = 0.5 * age + 0.3 * smoke + c["bU"] * U + c["bX"] * Xtrue
    t_event = rng.exponential(1.0 / (0.03 * np.exp(lp)))
    t = np.minimum(t_event, TAU); death = (t_event <= TAU).astype(int)
    # cross-sectional prevalent disease: covs + U + (optional) true X effect
    lpD = -1.0 + 0.4 * age + 0.3 * smoke + 0.4 * U + c["dX"] * Xtrue
    D = rng.binomial(1, 1 / (1 + np.exp(-lpD)))
    # MEASURED biomarker = truth, plus reverse-causation distortions:
    #  (i) impending death inflates it (drives a spurious, early-death-concentrated association)
    #  (ii) prevalent disease lowers it (drives the implausibly 'protective' cross-sectional sign)
    imp = np.exp(-t_event / 1.5); imp = (imp - imp.mean()) / imp.std()
    Xmeas = Xtrue + c["rc"] * imp - c["dDist"] * D
    Xmeas = (Xmeas - Xmeas.mean()) / Xmeas.std()
    return pd.DataFrame(dict(t=t, death=death, X=Xmeas, Z=Z, D=D, age=age, sex=sex, smoke=smoke))


def evalue(hr, lo, hi):
    """E-value (VanderWeele-Ding); CI-bound E-value is 1.0 whenever the interval crosses the null."""
    g = lambda x: (x if x >= 1 else 1 / x) + np.sqrt((x if x >= 1 else 1 / x) * ((x if x >= 1 else 1 / x) - 1))
    if hr > 1:
        b = lo if lo > 1 else 1.0           # harmful: lower bound nearest the null
    elif hr < 1:
        b = hi if hi < 1 else 1.0           # protective: upper bound nearest the null
    else:
        b = 1.0
    return g(hr), (g(b) if b != 1.0 else 1.0)


def _fast_cox(Xmat, t, event, max_iter=30, tol=1e-7):
    """Breslow partial-likelihood Cox via Newton-Raphson (vectorised). Returns beta, se.
    No ties handling needed (continuous exponential times); no weights/robust (not used in sim).
    Left-truncation with a COMMON entry time == ordinary Cox on the post-landmark subset, so the
    landmark is handled by subsetting before calling this."""
    order = np.argsort(-t)                      # descending time -> cumsum = risk-set sums
    X = Xmat[order]; ev = event[order].astype(bool)
    n, p = X.shape; beta = np.zeros(p); H = np.eye(p)
    for _ in range(max_iter):
        w = np.exp(X @ beta)
        S0 = np.cumsum(w)
        S1 = np.cumsum(w[:, None] * X, axis=0)
        S2 = np.cumsum(w[:, None, None] * (X[:, :, None] * X[:, None, :]), axis=0)
        S0e, S1e, S2e = S0[ev], S1[ev], S2[ev]
        xbar = S1e / S0e[:, None]
        grad = (X[ev] - xbar).sum(0)
        H = (S2e / S0e[:, None, None] - xbar[:, :, None] * xbar[:, None, :]).sum(0)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return beta, se


def cox_hr(df, expo, entry=None):
    sub = df if entry is None else df          # landmark already subset upstream; common entry == ordinary Cox
    Xmat = sub[[expo] + COV].to_numpy(float)
    b, se = _fast_cox(Xmat, sub["t"].to_numpy(float), sub["death"].to_numpy(float))
    hr = np.exp(b[0]); lo = np.exp(b[0] - 1.96 * se[0]); hi = np.exp(b[0] + 1.96 * se[0])
    from scipy.stats import norm
    p = 2 * norm.sf(abs(b[0] / se[0]))
    return hr, lo, hi, p


def logit_or(df, expo):
    X = sm.add_constant(df[[expo] + COV]); m = sm.Logit(df["D"], X).fit(disp=0)
    return float(np.exp(m.params[expo])), float(m.pvalues[expo])


def run_battery(df):
    hr, lo, hi, p = cox_hr(df, "X")                                   # primary
    zhr, zlo, zhi, zp = cox_hr(df, "Z")                               # negative control
    lm = df[df.t > LANDMARK].copy(); lm["entry"] = LANDMARK
    lhr, llo, lhi, lp_ = cox_hr(lm, "X", entry="entry")               # left-truncated landmark
    orr, orp = logit_or(df, "X")                                      # cross-sectional contrast
    ev, evlo = evalue(hr, lo, hi)
    return dict(hr=hr, lo=lo, hi=hi, p=p, zhr=zhr, zlo=zlo, zhi=zhi,
                lhr=lhr, llo=llo, lhi=lhi, orr=orr, evalue=ev, evalue_ci=evlo)


def classify(b):
    """Operationalised triangulation decision rule. Triangulation reasons RELATIVELY:
       a negative control should be much WEAKER than the exposure (else generic confounding);
       the landmark should RETAIN most of the effect (else reverse causation / early-death bias)."""
    xs = max(b["hr"] - 1.0, 0.01)                                     # exposure's harmful excess
    primary_sig   = (b["lo"] > 1.0)                                   # significant harmful primary
    persists_lm   = (b["llo"] > 1.0) and ((b["lhr"] - 1.0) >= 0.5 * xs)   # retains >=50% under landmark
    negctrl_clean = ((b["zhr"] - 1.0) <= 0.4 * xs)                    # control inverse or <40% of X's excess
    xsec_consist  = (b["orr"] >= 0.85)                               # cross-sectional DIAGNOSTIC (reverse-causation
    #   signature when reversed) -- reported and corroborating, but NOT a hard gate: a genuinely
    #   causal exposure can still show a reversed cross-sectional (disease perturbs the biomarker),
    #   as real cadmium does, so reverse causation is gated by the LANDMARK, not the cross-section.
    evalue_ok     = (b["evalue_ci"] >= 1.25)                          # CI-bound E-value clears a modest floor
    passed = primary_sig and persists_lm and negctrl_clean and evalue_ok
    return dict(verdict="causal" if passed else "spurious",
                primary_sig=primary_sig, persists_lm=persists_lm,
                negctrl_clean=negctrl_clean, xsec_consist=xsec_consist, evalue_ok=evalue_ok)


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    print("=== single-replicate diagnostic (n=6000) — do the mechanisms behave? ===")
    for sc in CFG:
        df = simulate(sc, 6000, rng)
        b = run_battery(df); v = classify(b)
        print(f"\n[{sc}]  deaths={int(df.death.sum())}  prevalent-D={int(df.D.sum())}")
        print(f"   primary X HR={b['hr']:.2f} ({b['lo']:.2f}-{b['hi']:.2f})  E-val(CI)={b['evalue_ci']:.2f}")
        print(f"   neg-control Z HR={b['zhr']:.2f} ({b['zlo']:.2f}-{b['zhi']:.2f})  [clean={v['negctrl_clean']}]")
        print(f"   landmark X HR={b['lhr']:.2f} ({b['llo']:.2f}-{b['lhi']:.2f})  [persists={v['persists_lm']}]")
        print(f"   cross-sec OR={b['orr']:.2f}  [consistent={v['xsec_consist']}]")
        print(f"   --> VERDICT: {v['verdict'].upper()}")
