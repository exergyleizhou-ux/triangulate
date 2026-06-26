"""Figures for the simulation-validation study (reads results.json from runner.py)."""
import os, json, numpy as np
import matplotlib.pyplot as plt
import figstyle as S
S.apply()
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "..", "figures")
R = json.load(open(os.path.join(HERE, "results.json")))
SC = ["causal", "confounded", "reverse", "null"]
LAB = {"causal": "Causal\n(truth: real)", "confounded": "Confounded", "reverse": "Reverse\ncausation", "null": "Null"}


def fig_main():
    """Protocol vs naive analyst: who gets credited 'causal' under each truth."""
    m = R["main"]["scenarios"]
    prot = [m[s]["verdict_causal"] for s in SC]
    naive = [m[s]["naive_causal"] for s in SC]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    x = np.arange(len(SC)); w = 0.38
    ax.bar(x - w/2, naive, w, color=S.HG, label="Naïve analyst (one significant model)")
    ax.bar(x + w/2, prot, w, color=S.CD, label="Triangulation protocol")
    ax.axhspan(-0.02, 1.02, xmin=0, xmax=0, color="none")
    for i, (p, n) in enumerate(zip(prot, naive)):
        ax.text(i - w/2, n + 0.02, f"{n:.2f}", ha="center", fontsize=6.5, color=S.MUTE)
        ax.text(i + w/2, p + 0.02, f"{p:.2f}", ha="center", fontsize=6.5, color=S.CD, fontweight="bold")
    ax.axvline(0.5, color=S.HAIR, lw=0.8)
    ax.text(0.0, 1.12, "wants HIGH", ha="center", fontsize=6.4, color=S.MIX, style="italic")
    ax.text(2.0, 1.12, "wants LOW (spurious truths)", ha="center", fontsize=6.4, color=S.MUTE, style="italic")
    ax.set_xticks(x); ax.set_xticklabels([LAB[s] for s in SC], fontsize=7.3)
    ax.set_ylabel("Fraction credited as 'causal'"); ax.set_ylim(0, 1.2)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    S.clean(ax); ax.set_title("The protocol keeps the true signal and rejects the spurious ones", fontsize=9.2)
    fig.tight_layout(); S.save(fig, "sim_fig1_operating", FIG)


def fig_checks():
    """Which check catches which threat (detection = 1 - pass-rate for that check)."""
    m = R["main"]["scenarios"]
    checks = ["negctrl_clean", "persists_lm", "xsec_consist"]
    cl = {"negctrl_clean": "Negative control\nlights up", "persists_lm": "Landmark\nattenuates", "xsec_consist": "Cross-section\nreverses"}
    threats = ["confounded", "reverse"]
    flag_rate = {c: [1 - m[t][c] for t in threats] for c in checks}   # 1 - pass = flagged
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    x = np.arange(len(threats)); w = 0.26; cols = [S.CD, S.PB, S.GOLD]
    for j, c in enumerate(checks):
        ax.bar(x + (j-1)*w, flag_rate[c], w, color=cols[j], label=cl[c].replace("\n", " "))
    ax.set_xticks(x); ax.set_xticklabels(["Confounded\ntruth", "Reverse-causation\ntruth"], fontsize=7.6)
    ax.set_ylabel("Fraction of replicates the check flags"); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=6.8, loc="upper center", ncol=3)
    S.clean(ax); ax.set_title("Complementary checks: each threat is caught by the matching check", fontsize=9.0)
    fig.tight_layout(); S.save(fig, "sim_fig2_checks", FIG)


def fig_sweeps():
    """Sample-size, effect-size and confounding sweeps."""
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    # (a) sample size: sensitivity (causal) and specificity (mean of 1-false over spurious)
    ss = R["sample_size"]; ns = sorted(int(k) for k in ss)
    sens = [ss[str(n)]["causal"] for n in ns]
    spec = [1 - np.mean([ss[str(n)][s] for s in ["confounded", "reverse", "null"]]) for n in ns]
    ax = axes[0]
    ax.plot(ns, sens, "-o", color=S.CD, lw=2, ms=4, label="Sensitivity (causal)")
    ax.plot(ns, spec, "-s", color=S.PB, lw=2, ms=4, label="Specificity (spurious)")
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns], fontsize=6.8)
    ax.minorticks_off()          # suppress matplotlib's default log-decade minor ticks (avoid label overprint)
    ax.set_xlabel("Sample size"); ax.set_ylabel("Rate"); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=6.6, loc="lower right"); S.clean(ax); S.panel(ax, "a")
    # (b) effect size sweep
    es = R["effect_size"]; hrs = sorted(float(k) for k in es); se = [es[str(h) if str(h) in es else h] for h in hrs]
    se = [es[k] for k in sorted(es, key=lambda z: float(z))]
    ax = axes[1]
    ax.plot([float(k) for k in sorted(es, key=lambda z: float(z))], se, "-o", color=S.CD, lw=2, ms=4)
    ax.axhline(0.8, color=S.HAIR, ls=(0,(4,3)), lw=0.9); ax.text(1.11, 0.82, "80%", fontsize=6.2, color=S.MUTE)
    ax.set_xlabel("True hazard ratio (per 1-SD)"); ax.set_ylabel("Sensitivity"); ax.set_ylim(0, 1.05)
    S.clean(ax); S.panel(ax, "b")
    # (c) confounding sweep
    cf = R["confounding"]; ks = sorted(float(k) for k in cf)
    fc = [cf[k]["false_causal"] for k in sorted(cf, key=lambda z: float(z))]
    nf = [cf[k]["negctrl_flags"] for k in sorted(cf, key=lambda z: float(z))]
    ax = axes[2]
    ax.plot(ks, nf, "-o", color=S.PB, lw=2, ms=4, label="Negative control flags")
    ax.plot(ks, fc, "-s", color=S.CD, lw=2, ms=4, label="Protocol false-positive")
    ax.set_xlabel("Confounding strength"); ax.set_ylabel("Rate"); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=6.4, loc="center right"); S.clean(ax); S.panel(ax, "c")
    fig.suptitle("Operating characteristics across sample size, effect size and confounding strength",
                 fontsize=9.0, fontweight="semibold", y=1.02)
    fig.tight_layout(); S.save(fig, "sim_fig3_sweeps", FIG)


def fig_adversarial():
    """Failure envelope: (a) weak negative control -> specificity collapses;
       (b) causal-with-confounding -> the true signal is discarded."""
    A = json.load(open(os.path.join(HERE, "results_adversarial.json")))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    wc = A["weak_control"]; zs = sorted(float(k) for k in wc)
    fcs = [wc[k]["false_causal"] for k in sorted(wc, key=lambda z: float(z))]
    ax = axes[0]
    ax.plot(zs, fcs, "-o", color=S.CD, lw=2, ms=4)
    ax.axvline(0.9, color=S.HAIR, ls=(0,(4,3)), lw=0.9); ax.text(0.9, 0.9, " exposure's\n confounder share", fontsize=5.8, color=S.MUTE, va="top")
    ax.set_xlabel("Negative control's confounder share (z$_U$)"); ax.set_ylabel("Protocol false-positive rate")
    ax.set_ylim(-0.03, 1.05); S.clean(ax); S.panel(ax, "a")
    ax.set_title("Weak control $\\Rightarrow$ specificity collapses", fontsize=8.2)
    cw = A["causal_with_confounding"]; cs = sorted(float(k) for k in cw)
    sens = [cw[k]["sensitivity"] for k in sorted(cw, key=lambda z: float(z))]
    ax = axes[1]
    ax.plot(cs, sens, "-o", color=S.PB, lw=2, ms=4)
    ax.set_xlabel("Confounding in the (truly causal) exposure"); ax.set_ylabel("Sensitivity (true signal kept)")
    ax.set_ylim(-0.03, 1.05); S.clean(ax); S.panel(ax, "b")
    ax.set_title("Causal + confounding $\\Rightarrow$ true signal discarded", fontsize=8.2)
    fig.suptitle("Failure envelope: the protocol's specificity and sensitivity are conditional",
                 fontsize=9.0, fontweight="semibold", y=1.02)
    fig.tight_layout(); S.save(fig, "sim_fig4_adversarial", FIG)


def fig_comparison():
    """Head-to-head: protocol vs named single-strategy methods (sensitivity + specificity)."""
    C = json.load(open(os.path.join(HERE, "results_comparison.json")))["by_method"]
    order = ["naive", "evalue_only", "negctrl_calib", "speccurve", "negctrl_landmark", "protocol"]
    labels = ["naive\n(1 sig. model)", "E-value\nalone", "neg-control\ncalibration", "spec.\ncurve", "neg-control\n+ landmark", "protocol\n(battery)"]
    sens = [C[m]["sensitivity"] for m in order]; spec = [C[m]["specificity"] for m in order]
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    ax.bar(x - w/2, sens, w, color=S.PB, label="sensitivity")
    ax.bar(x + w/2, spec, w, color=S.CD, label="specificity")
    for i, (s, p) in enumerate(zip(sens, spec)):
        ax.text(i - w/2, s + .02, f"{s:.2f}", ha="center", fontsize=6.4, color=S.PB)
        ax.text(i + w/2, p + .02, f"{p:.2f}", ha="center", fontsize=6.4, color=S.CD)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.2); ax.set_ylim(0, 1.12)
    ax.set_ylabel("rate"); ax.legend(frameon=False, fontsize=7.4, ncol=2, loc="lower center")
    ax.set_title("Only the combined battery keeps both sensitivity AND specificity", fontsize=8.6)
    S.clean(ax); fig.tight_layout(); S.save(fig, "sim_fig5_comparison", FIG)

def fig_envelope():
    """The honest out-of-design envelope: plasmode (real covariates) + randomized ensemble vs the
       isolated-threat best case."""
    A = json.load(open(os.path.join(HERE, "results_adversarial_dgp.json")))
    M = json.load(open(os.path.join(HERE, "results.json")))["main"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    cats = ["isolated\nthreats", "plasmode\n(mild conf.)", "plasmode\n(strong conf.)", "randomized\nensemble"]
    sens = [M["sensitivity"], A["plasmode_causal_mild"], A["plasmode_causal_strong"], A["randomized_ensemble"]["sensitivity"]]
    spec = [M["specificity"], 1 - A["plasmode_confounded_null"], 1 - A["plasmode_confounded_null"], A["randomized_ensemble"]["specificity"]]
    x = np.arange(len(cats)); w = 0.38
    ax.bar(x - w/2, sens, w, color=S.PB, label="sensitivity")
    ax.bar(x + w/2, spec, w, color=S.CD, label="specificity")
    for i,(s,p) in enumerate(zip(sens,spec)):
        ax.text(i-w/2, s+.02, f"{s:.2f}", ha="center", fontsize=6.4, color=S.PB)
        ax.text(i+w/2, p+.02, f"{p:.2f}", ha="center", fontsize=6.4, color=S.CD)
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=7.0); ax.set_ylim(0,1.12); ax.set_ylabel("rate")
    ax.legend(frameon=False, fontsize=7.4, ncol=2, loc="lower center")
    ax.set_title("Out-of-design: specificity holds, sensitivity is conservative", fontsize=8.6)
    S.clean(ax); fig.tight_layout(); S.save(fig, "sim_fig6_envelope", FIG)


def fig_dag():
    """The four causal structures (DGP) as directed graphs."""
    from matplotlib.patches import FancyArrowPatch, Circle
    pos = {"U": (0.5, 0.90), "X": (0.13, 0.48), "Y": (0.87, 0.48), "Z": (0.13, 0.08), "D": (0.87, 0.08)}
    name = {"U": "U", "X": "X", "Y": "death", "Z": "Z", "D": "D"}
    worlds = {
        "Causal": [("U","X","f"),("U","Y","f"),("U","Z","f"),("X","Y","key"),("X","D","f")],
        "Confounded": [("U","X","s"),("U","Y","s"),("U","Z","f")],
        "Reverse causation": [("U","X","f"),("U","Y","f"),("U","Z","f"),("Y","X","rev"),("D","X","rev")],
        "Null": [("U","Z","f")],
    }
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.0))
    for ax, (title, edges) in zip(axes.ravel(), worlds.items()):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.set_title(title, fontsize=8.6, fontweight="semibold", color=S.INK)
        for nd, (x, y) in pos.items():
            unobs = (nd == "U")
            ax.add_patch(Circle((x, y), 0.075, facecolor="white",
                         edgecolor=S.MUTE if unobs else S.INK, lw=0.9, ls=(0,(2,2)) if unobs else "-", zorder=3))
            ax.text(x, y, name[nd], ha="center", va="center", fontsize=7.0,
                    color=S.MUTE if unobs else S.INK, zorder=4, style="italic" if unobs else "normal")
        for a, b, kind in edges:
            (x0, y0), (x1, y1) = pos[a], pos[b]
            col = {"key": S.CD, "s": S.INK, "rev": S.PB, "f": S.FAINT}[kind]
            lw = {"key": 2.2, "s": 1.6, "rev": 1.6, "f": 0.9}[kind]
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
                         lw=lw, color=col, shrinkA=10, shrinkB=10,
                         connectionstyle="arc3,rad=0.12", zorder=2))
        if title == "Causal":
            ax.text(0.5, 0.46, "true effect", fontsize=5.6, color=S.CD, ha="center", style="italic")
    fig.suptitle("Data-generating structures (U unmeasured; red = the true causal edge)",
                 fontsize=8.8, fontweight="semibold", y=0.99)
    fig.tight_layout(rect=(0,0,1,0.96)); S.save(fig, "sim_fig0_dag", FIG)


if __name__ == "__main__":
    fig_dag(); fig_main(); fig_checks(); fig_sweeps(); fig_adversarial(); fig_comparison(); fig_envelope()
    print("saved sim figures ->", FIG)
