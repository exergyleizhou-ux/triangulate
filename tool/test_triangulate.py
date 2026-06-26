"""Tests for the triangulate tool: on simulated cohorts with KNOWN truth, the protocol must
credit the causal one and reject the confounded / reverse-caused / null ones."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
from sim_engine import simulate
from triangulate import Spec, run, evalue

SPEC = Spec(time="t", event="death", exposure="X", negative_control="Z",
            cross_section="D", covariates=["age", "sex", "smoke"], landmarks=(2.0,))


def _verdict(scenario, seed):
    df = simulate(scenario, 6000, np.random.default_rng(seed))
    return run(df, SPEC)["verdict"]


def test_causal_is_credited():
    assert _verdict("causal", 1) == "robust-causal-candidate"


def test_confounded_is_rejected():
    assert _verdict("confounded", 2) == "fragile-or-spurious"


def test_reverse_is_rejected():
    assert _verdict("reverse", 3) == "fragile-or-spurious"


def test_null_is_rejected():
    assert _verdict("null", 4) == "fragile-or-spurious"


def test_evalue_known_value():
    # VanderWeele-Ding: HR 1.27 -> E-value ~1.86; CI bound 1.17 -> ~1.62 (harmful: hi unused)
    pt, ci = evalue(1.27, 1.17, 1.38)
    assert abs(pt - 1.86) < 0.03 and abs(ci - 1.62) < 0.03
    # protective HR whose CI crosses the null must give CI-bound E-value 1.0 (the bug this guards)
    assert evalue(0.85, 0.70, 1.05)[1] == 1.0


def test_confounded_flags_negative_control():
    df = simulate("confounded", 6000, np.random.default_rng(5))
    assert run(df, SPEC)["flags"]["negctrl_clean"] is False


def test_reverse_flags_landmark_or_xsec():
    df = simulate("reverse", 6000, np.random.default_rng(6))
    f = run(df, SPEC)["flags"]
    assert (f["persists_landmark"] is False) or (f["xsec_consistent"] is False)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); passed += 1; print(f"PASS {fn.__name__}")
        except AssertionError:
            print(f"FAIL {fn.__name__}")
    print(f"\n{passed}/{len(fns)} passed")
