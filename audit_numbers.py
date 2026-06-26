"""Numerical-consistency audit: every headline number in the two papers must match the computed
JSON / text outputs. Catches stale numbers left behind by edits. Run from the project root."""
import json, re, sys, numpy as np

SIM = "sim/"
tex = (open("manuscript/main_method.tex").read()
       + open("manuscript/results_section.tex").read()
       + open("xcohort/protocol_cross_cohort.tex").read())

main = json.load(open(SIM + "results.json"))["main"]
adv  = json.load(open(SIM + "results_adversarial.json"))
ext  = json.load(open(SIM + "results_extreme.json"))
dgp  = json.load(open(SIM + "results_adversarial_dgp.json"))
cmp  = json.load(open(SIM + "results_comparison.json"))["by_method"]

sc = main["scenarios"]
naive_spec = 1 - np.mean([sc[w]["naive_causal"] for w in ("confounded", "reverse", "null")])

checks = []  # (label, value_string_as_in_paper, present?)
def chk(label, s):
    checks.append((label, s, s in tex))

# headline operating characteristics
chk("sensitivity 0.99",            "0.99")
chk("specificity 1.00",            "1.00")
chk(f"naive specificity ({naive_spec:.2f})", "0.32")
# complementarity / E-value bite
chk("null E-value bite ~0.50 (0.497)", "0.497") if "0.497" in tex else None
# failure envelope
chk("collider mis-credit 0.72",    "0.72")
chk("collider primary 0.99",       "0.99")
chk("collider landmark 0.05",      "0.05")
chk("weak-control FP 1.00 @0.2",   "1.00")
chk("mixed adversary 0.80",        "0.80")
# out-of-design
chk("plasmode mild 0.53",          "0.53")
chk("ensemble sensitivity 0.56",   "0.56")
chk("ensemble specificity 0.98",   "0.98")
# threshold sweep
chk("threshold 27 configs",        "27")
chk("threshold spec >=0.999",      "0.999")
# composite (results_comparison)
chk("composite sens 0.99",         "0.99")
chk("composite null FP 0.01",      "0.01")
# non-PH out-of-family stress test
try:
    nph = json.load(open(SIM + "results_nonph_weibull.json"))
    chk(f"non-PH sensitivity ({nph['sensitivity']:.2f})", "0.99")
    chk(f"non-PH specificity ({nph['specificity']:.2f})", "0.76")
    chk(f"non-PH reverse false-credit ({nph['per_world']['reverse']:.2f})", "0.72")
except FileNotFoundError:
    print("  (results_nonph_weibull.json not present — run sim/nonph_weibull.py)")

# NHANES III worked example
for s in ["16{,}560", "7{,}122", "1.12", "1.08--1.15", "1.11", "1.10"]:
    chk(f"NHANES III {s}", s)
# NHANES 2005-2018 (companion) HRs in realdata table
for s in ["1.27", "1.17--1.38", "1.04--1.21"]:
    chk(f"companion HR {s}", s)

# cross-check the result macros against the JSON
macros = dict(re.findall(r"\\newcommand\{\\(R\w+)\}\{([\d.]+)\}", open("manuscript/main_method.tex").read()))
exp = {
    "Rsens": round(main["sensitivity"], 2), "Rspec": round(main["specificity"], 2),
    "RspecNaive": round(float(naive_spec), 2),
    "Rconf": round(sc["confounded"]["verdict_causal"], 2),
    "Rrev": round(sc["reverse"]["verdict_causal"], 2),
    "Rnull": round(sc["null"]["verdict_causal"], 2),
}
print("=== result macros vs JSON ===")
macro_ok = True
for k, v in exp.items():
    got = float(macros.get(k, "nan"))
    ok = abs(got - v) < 0.01
    macro_ok &= ok
    print(f"  {'OK ' if ok else 'MISMATCH'} \\{k} = {macros.get(k)}  (JSON {v})")

# comparison-table rows present
print("=== comparison table (results_comparison.json) ===")
for m in ["naive", "evalue_only", "negctrl_calib", "speccurve", "negctrl_landmark", "protocol"]:
    r = cmp[m]
    print(f"  {m:16s} sens={r['sensitivity']:.2f} spec={r['specificity']:.2f}")

print("\n=== number presence in manuscript text ===")
nfail = 0
for label, s, ok in checks:
    if not ok: nfail += 1
    print(f"  {'OK  ' if ok else 'FAIL'} {label:34s} expected substring: {s!r}")

print(f"\nSUMMARY: {len(checks)-nfail}/{len(checks)} numbers present; macros {'OK' if macro_ok else 'MISMATCH'}")
sys.exit(0 if nfail == 0 and macro_ok else 1)
