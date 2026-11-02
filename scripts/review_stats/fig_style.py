"""Shared chart chrome for the review-round figures (dataviz reference palette, light mode)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"      # categorical slots 1-3
NEUTRAL = "#898781"                                       # baseline / reference series


def setup():
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                         "axes.labelcolor": INK, "figure.facecolor": SURFACE})


def style(ax, grid_axis="both"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9, length=3, width=0.8)
    ax.grid(color=GRID, linewidth=0.6, axis=grid_axis)
    ax.set_axisbelow(True)


def panel(ax, letter, x=-0.14):
    ax.set_title(letter, loc="left", fontsize=13, fontweight="bold", x=x)


def p_text(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.2f}" if p >= 0.01 else f"p = {p:.3f}"
