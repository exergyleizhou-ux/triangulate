"""Build the NHANES III (1988-1994) analytic dataset for a FREE temporal replication of the
cadmium(body-burden)+lead -> mortality association, then apply the triangulation tool.

NHANES III measures blood LEAD (PBP) and URINARY cadmium (UDP) -- the body-burden marker --
but NOT blood cadmium and NOT mercury. So this replicates the urinary-Cd + Pb arms and the
landmark / E-value / cross-sectional (prevalent diabetes) checks; the Hg negative control
and a blood-Cd like-for-like do NOT travel (stated honestly in the manuscript).

Fixed-width column positions taken from the official NHANES III SAS layouts (1-indexed,
inclusive -> converted to pandas 0-indexed half-open colspecs).
"""
import os, glob, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))

# ---- fixed-width specs (pandas 0-indexed half-open) ----
LAB = {"SEQN": (0, 5), "PBP": (1422, 1426), "UDP": (1944, 1949)}          # blood lead, urinary cadmium
ADULT = {"SEQN": (0, 5), "DMARETHN": (11, 12), "HSSEX": (14, 15), "HSAGEIR": (17, 19),
         "DMPPIR": (35, 41), "HFA8R": (1255, 1257), "HAD1": (1560, 1561),
         "HAR1": (2280, 2281), "HAR3": (2284, 2285)}
MORT = {"SEQN": (0, 5), "ELIGSTAT": (14, 15), "MORTSTAT": (15, 16), "UCOD": (16, 19), "PERMTH_EXM": (45, 48)}


def read_fw(path, spec):
    df = pd.read_fwf(path, colspecs=list(spec.values()), names=list(spec.keys()), dtype=str)
    return df


def num(s):
    return pd.to_numeric(s.astype(str).str.strip().replace({"": np.nan}), errors="coerce")


def main():
    for f in ["lab.dat", "adult.dat", "mort3.dat"]:
        p = os.path.join(HERE, f)
        if not os.path.exists(p) or os.path.getsize(p) < 1_000_000:
            print(f"!! {f} not fully downloaded yet ({os.path.getsize(p) if os.path.exists(p) else 0} bytes). Re-run when complete."); return
    lab = read_fw(os.path.join(HERE, "lab.dat"), LAB)
    adult = read_fw(os.path.join(HERE, "adult.dat"), ADULT)
    mort = read_fw(os.path.join(HERE, "mort3.dat"), MORT)
    for df in (lab, adult, mort):
        df["SEQN"] = num(df["SEQN"])
    print(f"rows: lab={len(lab)} adult={len(adult)} mort={len(mort)}")

    d = adult.merge(lab, on="SEQN", how="inner").merge(mort, on="SEQN", how="inner")
    print(f"merged on SEQN: {len(d)}")

    # exposures (drop NHANES III sentinel/blank codes, e.g. 8888 = blank-but-applicable)
    d["PB"] = num(d["PBP"]).where(lambda s: s < 100)     # blood lead ug/dL; >100 are sentinels
    d["UCD"] = num(d["UDP"]).where(lambda s: s < 50)     # urinary cadmium ng/mL; >50 are sentinels
    # NHANES III blood lead stored in ug/dL; urinary cadmium in ng/mL. Calibrate by inspection:
    print("\n-- raw exposure percentiles (calibration check) --")
    for v in ["PB", "UCD"]:
        q = d[v].dropna().quantile([.05, .5, .95]).round(2).tolist()
        print(f"   {v}: p5/p50/p95 = {q}  (n={d[v].notna().sum()})")
    # known NHANES III: median blood lead ~2-3 ug/dL, urinary Cd ~0.3-0.5 ug/g; flag if implied decimals
    # mortality
    for v in ["ELIGSTAT", "MORTSTAT", "UCOD", "PERMTH_EXM"]:
        d[v] = num(d[v])
    d = d[d.ELIGSTAT == 1].copy()
    d["t"] = d.PERMTH_EXM / 12.0
    d["death"] = (d.MORTSTAT == 1).astype(int)
    d["cvd"] = ((d.MORTSTAT == 1) & (d.UCOD.isin([1, 5]))).astype(int)   # heart disease / cerebrovascular
    # covariates
    d["AGE"] = num(d.HSAGEIR); d["SEX"] = num(d.HSSEX); d["RACE"] = num(d.DMARETHN)
    d["PIR"] = num(d.DMPPIR); d["EDU"] = num(d.HFA8R)
    har1 = num(d.HAR1); har3 = num(d.HAR3)
    d["SMOKE"] = np.where(har3 == 1, 2, np.where(har1 == 1, 1, 0))       # current / former / never
    d["diabetes"] = (num(d.HAD1) == 1).astype(float)                     # ever told diabetes (cross-sectional)

    # analytic restrictions: adults >=20, valid exposure + mortality
    d = d[(d.AGE >= 20) & d.t.notna() & (d.t > 0)].copy()
    n_pb = d[d.PB.notna()]; n_cd = d[d.UCD.notna()]
    print(f"\nadults>=20 with mortality: {len(d)}  | with blood Pb: {len(n_pb)} ({int(n_pb.death.sum())} deaths)"
          f"  | with urinary Cd: {len(n_cd)} ({int(n_cd.death.sum())} deaths)")
    d.to_csv(os.path.join(HERE, "analytic_nhanes3.csv"), index=False)
    print("saved analytic_nhanes3.csv")


if __name__ == "__main__":
    main()
