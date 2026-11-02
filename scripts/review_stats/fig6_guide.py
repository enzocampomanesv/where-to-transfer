"""
fig6_guide.py  (review round 1; rebuilds Fig 6, the practitioner's guide) -- DRAFT
==================================================================================

The guide redrawn around what the leak-free analyses support, with every step
labelled by when its information is available (R1-m7): before annotation,
after a seed sample, or only retrospectively. Replaces the published guide,
whose source choice used the full-label Spearman ceiling, whose diagnosis
matrix used coverage as a predictor, and whose sampling step recommended
informed sampling.

Numbers in the boxes come from source_selection.py (regret, aggregates), the
rerun (finetuning gains), and alr_inference.py / iso_sensitivity.py (PD share
of zero-shot error). The third decision is drawn under framing C and is a
draft until the co-authors agree on the framing.

  python fig6_guide.py --out ../../figures/fig6_guide.png
"""

from __future__ import annotations

import argparse
import os
import textwrap

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import fig_style as fs

BAND = {"before": "#eef4fc", "seed": "#fdf1ec", "retro": "#f1f0ec"}   # light tints of blue / orange / neutral


def box(ax, x, y, w, h, title, body, edge=fs.MUTED, face=fs.SURFACE, width=None, title_size=10, body_size=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor=face, edgecolor=edge, linewidth=1.2, zorder=2))
    ty = y + h - 2.2
    if title:
        ax.text(x + 1.8, ty, title, fontsize=title_size, fontweight="bold", color=fs.INK, va="top", zorder=3)
        ty -= 4.2
    wrap = width or int(w * 1.55)
    lines = []
    for para in body:
        lines += textwrap.wrap(para, wrap) + [""]
    ax.text(x + 1.8, ty, "\n".join(lines[:-1]), fontsize=body_size, color=fs.INK2, va="top",
            linespacing=1.35, zorder=3)


def band(ax, y, h, key, label):
    ax.add_patch(FancyBboxPatch((0.5, y), 99, h, boxstyle="round,pad=0,rounding_size=1.5",
                                facecolor=BAND[key], edgecolor="none", zorder=0))
    ax.text(2, y + h - 1.4, label, fontsize=9.5, color=fs.INK, fontweight="bold", va="top", zorder=1)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=fs.INK2, lw=1.2, zorder=4))


def main(args):
    fs.setup()
    fig, ax = fs.plt.subplots(figsize=(11, 9), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(9.5, 99.5)
    ax.axis("off")

    band(ax, 84, 15, "before", "BEFORE ANNOTATION  ·  no target labels")
    box(ax, 3, 85.5, 94, 8, None,
        ["Shortlist candidate single-city source models. Coverage (feature overlap with the target) can be "
         "computed now: it describes how much of the target is unfamiliar to each source, but it does not "
         "identify the best source and is not used to choose one."], width=128)

    band(ax, 28.5, 54, "seed", "AFTER A SEED SAMPLE  ·  500 random target comparisons, reused for finetuning")
    box(ax, 3, 68.5, 94, 9, None,
        ["Score every candidate city zero-shot on the seed comparisons. Its accuracy (1 − error rate) is the "
         "one quantity the decisions below need. Most errors are Preference Disagreement: familiar pairs that "
         "target residents judge differently from the source city."], width=128)

    top, h = 30, 36.5
    box(ax, 3, top, 30, h, "Which source?",
        ["Take the single city most accurate on the seed.",
         "Expected loss against the best of all 22 candidate models: ≈ 0.02 Spearman "
         "(≈ 0.06 with a 100-comparison seed).",
         "Prefer single cities: aggregates often score higher zero-shot but gain less from finetuning."],
        edge=fs.BLUE, width=36)
    box(ax, 35, top, 30, h, "Finetune, and how much?",
        ["If the seed accuracy is sufficient for your purpose, use the zero-shot model; otherwise finetune. "
         "The higher the seed accuracy, the less finetuning adds (ρ = −0.57).",
         "Our reference: at ≈ 70% (Accra ↔ Tema; zero-shot Spearman 0.63–0.68) finetuning added "
         "only +0.01 to +0.05.",
         "Near chance (≈ 50%): finetune on as many comparisons as feasible (gains rise up to 5,000), "
         "not on ~100 (worse in 15 of 42 pairs)."],
        edge=fs.BLUE, width=36)
    box(ax, 67, top, 30, h, "Which comparisons?",
        ["Sample comparisons at random.",
         "SAT-informed selection does not beat random sampling: Preference Disagreement is defined by the "
         "target's labels and is spread evenly through the pool, so it cannot be targeted before annotation.",
         "Annotation effort is better spent on more comparisons than on choosing them."],
        edge=fs.ORANGE, width=36)
    ax.text(96, top + 1.2, "DRAFT: depends on framing", fontsize=7.5, color=fs.ORANGE, ha="right", va="bottom",
            fontweight="bold", zorder=5)

    band(ax, 10.5, 16.5, "retro", "RETROSPECTIVE  ·  full target labels")
    box(ax, 3, 11.5, 94, 10.5, None,
        ["The Source Applicability Typology decomposes the zero-shot error after the fact: Preference "
         "Disagreement accounts for about 95% of misclassified comparisons and dominates the variation in "
         "zero-shot performance across city pairs; Structural and Relational Ignorance are small. This explains "
         "why transfer fails but is not needed for the three decisions above."], width=128)

    arrow(ax, (88, 85.5), (88, 77.5))
    for x in (18, 50, 82):
        arrow(ax, (x, 68.5), (x, top + h))
    arrow(ax, (33, top + h / 2), (35, top + h / 2))
    arrow(ax, (65, top + h / 2), (67, top + h / 2))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor=fs.SURFACE)
    print(f"saved {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../../figures/fig6_guide.png")
    main(ap.parse_args())
