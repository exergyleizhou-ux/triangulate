"""Export a clean public-deposit version of the NHANES III analytic cohort.

The working file `analytic_nhanes3.csv` keeps both the raw NHANES variable names and the renamed,
sentinel-cleaned analytic columns (for traceability during development). For public deposit we keep
only the analysis-ready columns: this drops the redundant raw duplicates (e.g. UDP≡UCD, PBP≡PB,
HSSEX≡SEX) and the leftover sentinel codes (88888 in raw urinary cadmium, 8888 in raw blood lead),
which live only in the raw columns and were already excluded from the analysis variables UCD/PB.
The HR estimates are unchanged: UCD/PB/t/death/cvd are identical to the columns the models used.
"""
import pandas as pd

KEEP = ["SEQN", "AGE", "SEX", "RACE", "PIR", "EDU", "SMOKE", "diabetes",
        "PB", "UCD", "t", "death", "cvd"]

df = pd.read_csv("analytic_nhanes3.csv")
clean = df[KEEP].copy()
clean.to_csv("analytic_nhanes3_clean.csv", index=False)
assert (clean["UCD"].dropna() < 8888).all() and (clean["PB"].dropna() < 8888).all(), "sentinel leak!"
print(f"wrote analytic_nhanes3_clean.csv: {len(clean)} rows, {len(KEEP)} columns")
print(f"  urinary cadmium UCD: n={clean['UCD'].notna().sum()}, median={clean['UCD'].median():.2f}, max={clean['UCD'].max():.2f}")
print(f"  blood lead PB:       n={clean['PB'].notna().sum()}, median={clean['PB'].median():.2f}, max={clean['PB'].max():.2f}")
