"""Shared refined aesthetic for the metals-mortality figures: a restrained
Nature/Lancet-grade system (sophisticated muted palette, hairline axes, generous
whitespace, editable-text SVG) with a cohesive semantic colour identity:
  cadmium = deep rose (primary signal) · lead = steel blue · mercury = warm grey
  (neutral negative control) · mixture = teal.
Every figure imports this so colour, type and spacing are identical throughout.
"""
import os, matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---- semantic palette ----
INK   = "#22262C"   # primary text / strong lines
MUTE  = "#737B85"   # secondary text / ticks
FAINT = "#AEB4BC"   # tertiary
HAIR  = "#DDE1E6"   # hairlines, reference grid
CD    = "#B23A48"   # cadmium — deep rose
PB    = "#2F6E8E"   # lead — steel blue
HG    = "#9BA1A8"   # mercury — warm grey (neutral)
MIX   = "#3C8A77"   # mixture — teal
GOLD  = "#C99A4E"   # warm accent
PLUM  = "#7E5A86"   # secondary accent (mediation)
HARM  = "#B23A48"   # harmful-zone tint base
PROT  = "#2F6E8E"   # protective-zone tint base
# sequential quartile ramp (low→high risk: cool→warm→rose)
SEQ4  = ["#A9C4D4", "#6E9CB4", "#C98A5C", "#B23A48"]
# alternative single-hue (purple) sequential ramp, for an ordered quartile figure that
# must read as visually distinct from SEQ4 (used for the competing-risks CIF, Fig 9)
SEQ4P = ["#CDBCDD", "#A688C6", "#7E57A8", "#553A7E"]
C = dict(ink=INK, mute=MUTE, faint=FAINT, hair=HAIR, cd=CD, pb=PB, hg=HG, mix=MIX,
         gold=GOLD, plum=PLUM, harm=HARM, prot=PROT, seq4=SEQ4,
         blue=PB, red=CD, green=MIX, grey=HG)


def apply():
    fams = {f.name for f in font_manager.fontManager.ttflist}
    sans = [f for f in ["Helvetica Neue", "Helvetica", "Arial", "TeX Gyre Heros", "DejaVu Sans"] if f in fams] or ["DejaVu Sans"]
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": sans,
        "font.size": 8.3, "axes.titlesize": 9.8, "axes.titleweight": "semibold",
        "axes.labelsize": 8.4, "axes.labelcolor": INK, "text.color": INK,
        "xtick.labelsize": 7.6, "ytick.labelsize": 7.6, "legend.fontsize": 7.6,
        "axes.edgecolor": INK, "axes.linewidth": 0.6,
        "xtick.color": MUTE, "ytick.color": MUTE,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "axes.titlepad": 7, "axes.labelpad": 3.5,
        "lines.linewidth": 1.5, "lines.solid_capstyle": "round",
        "mathtext.fontset": "custom", "mathtext.rm": sans[0], "mathtext.it": sans[0],
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": 600, "figure.dpi": 150, "figure.facecolor": "white",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    })


def clean(ax, left=True, bottom=True):
    """Hairline two-spine axis."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(left); ax.spines["bottom"].set_visible(bottom)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK); ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=MUTE, labelcolor=INK)


def panel(ax, letter, x=-0.02, y=1.06, fs=12):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=fs, fontweight="bold",
            va="bottom", ha="right", color=INK)


def refline(ax, x=1.0, vertical=True):
    (ax.axvline if vertical else ax.axhline)(x, color=FAINT, lw=0.9, ls=(0, (4, 3)), zorder=1)


def save(fig, name, figdir):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(os.path.join(figdir, f"{name}.{ext}"))
    plt.close(fig)
    print("  saved", name)
