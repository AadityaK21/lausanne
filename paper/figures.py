"""
Figures for the manuscript, in the visual language of the reference paper.

The reference article (Shrestha, Krishna & von Krogh, 2021, Journal of Business
Research) uses a recognisable diagram idiom: white boxes with thin dark rules,
arrows in a mid Office blue, stage bands tinted amber / periwinkle / sage
behind grouped steps, sans-serif labels, and italic captions set below.
Matching that idiom rather than inventing one is deliberate; a paper reads as
one body of work when its figures share a grammar.

Every figure is drawn to the width of a single journal column. That is a
typesetting constraint rather than an aesthetic one: a figure wider than a
column forces a page break, which strands whatever text was mid-column and
leaves most of a page blank. Designing to the column removes the problem at
source instead of managing it afterwards.

Figures are generated from code so that changing a number changes the figure,
rather than leaving a stale image quietly disagreeing with the text.

Run:
    python paper/figures.py
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = pathlib.Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# A journal column is about 3.4 in. Everything is drawn to that.
COL_IN = 3.38

ARROW = "#4472C4"
BOX_EDGE = "#3F3F3F"
BOX_FILL = "#FFFFFF"
BAND_DATA = "#FDF1D9"
BAND_LEARN = "#DEE3F1"
BAND_EVAL = "#E2EFD9"
BAND_RISK = "#F8E3DE"
SAFE = "#2F6F5E"
RISK = "#A8382C"
INK = "#1A1A1A"
MUTED = "#5A5A5A"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------


def box(ax, x, y, w, h, text, *, fs=6.4, fill=BOX_FILL, edge=BOX_EDGE,
        bold=False, italic=False, lw=0.8):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="square,pad=0", linewidth=lw,
        edgecolor=edge, facecolor=fill, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, zorder=4, linespacing=1.35,
            fontweight="bold" if bold else "normal",
            style="italic" if italic else "normal")


def arrow(ax, x1, y1, x2, y2, *, color=ARROW, lw=1.2, ms=7):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=ms, linewidth=lw, color=color,
                                 zorder=5))


def canvas(h_in, xlim, ylim):
    fig, ax = plt.subplots(figsize=(COL_IN, h_in))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


# --------------------------------------------------------------------------
# Fig. 1 - staged pipeline
# --------------------------------------------------------------------------


def fig_pipeline():
    fig, ax = canvas(6.9, (0, 10), (0, 20.4))

    bx, bw, bh, pitch = 1.35, 8.2, 1.9, 2.55
    top = 17.1

    steps = [
        ("Speech recognition", "faster-whisper, word timings", "~2.5 GB VRAM"),
        ("Speaker diarization", "pyannote, exclusive spans", "~1.5 GB VRAM"),
        ("Reconciliation", "words to attributed utterances", "CPU only"),
        ("Note selection", "rule | model-selected | abstractive",
         "0 or ~5.5 GB"),
        ("Validation", "identifier check, re-materialisation", "CPU only"),
        ("Rendering", "Markdown, Word, record", "CPU only"),
        ("Provenance attestation", "hashes, endpoints, egress", "CPU only"),
    ]
    ys = [top - i * pitch for i in range(len(steps))]

    bands = [
        (ys[2] - 0.45, (ys[0] + bh) - (ys[2] - 0.45) + 0.45, BAND_DATA,
         "Stage 1: RECORD"),
        (ys[4] - 0.45, (ys[3] + bh) - (ys[4] - 0.45) + 0.45, BAND_LEARN,
         "Stage 2: NOTES"),
        (ys[6] - 0.45, (ys[5] + bh) - (ys[6] - 0.45) + 0.45, BAND_EVAL,
         "Stage 3: OUTPUT"),
    ]
    for y0, height, tint, label in bands:
        ax.add_patch(Rectangle((0.5, y0), 9.2, height, facecolor=tint,
                               edgecolor="none", zorder=1))
        ax.text(0.86, y0 + height / 2, label, ha="center", va="center",
                fontsize=5.9, rotation=90, style="italic", fontweight="bold",
                color=INK, zorder=2)

    for (title, sub, cost), y in zip(steps, ys):
        ax.add_patch(FancyBboxPatch((bx, y), bw, bh, boxstyle="square,pad=0",
                                    linewidth=0.8, edgecolor=BOX_EDGE,
                                    facecolor=BOX_FILL, zorder=3))
        ax.text(bx + bw / 2, y + bh * 0.72, title, ha="center", va="center",
                fontsize=6.5, color=INK, zorder=4)
        ax.text(bx + bw / 2, y + bh * 0.44, sub, ha="center", va="center",
                fontsize=5.4, color=MUTED, zorder=4)
        ax.text(bx + bw / 2, y + bh * 0.17, cost, ha="center", va="center",
                fontsize=5.2, color=ARROW, style="italic", zorder=4)

    for i in range(len(steps) - 1):
        arrow(ax, bx + bw / 2, ys[i], bx + bw / 2, ys[i + 1] + bh)

    ax.text(5.0, 19.5,
            "Each stage runs as an isolated process,\n"
            "communicating only through JSON on disk.",
            ha="center", va="center", fontsize=6.0, color=INK, linespacing=1.4)

    ax.add_patch(Rectangle((0.5, 0.25), 9.2, 1.8, facecolor="#F2F2F2",
                           edgecolor=BOX_EDGE, linewidth=0.7, zorder=2))
    ax.text(5.1, 1.15,
            "Combined peak demand ~12 GB against ~7.3 GB\n"
            "usable: the stages cannot be co-resident.",
            ha="center", va="center", fontsize=5.9, color=INK,
            linespacing=1.4, zorder=3)

    fig.savefig(OUT / "fig1_pipeline.png")
    plt.close(fig)
    return "fig1_pipeline.png"


# --------------------------------------------------------------------------
# Fig. 2 - where output text originates
# --------------------------------------------------------------------------


def fig_modes():
    """
    Three rows rather than three columns. The comparison still reads across,
    and the figure fits a single column, which is what keeps the page whole.
    """
    fig, ax = canvas(4.6, (0, 10), (0, 13.6))

    rh = 3.9
    rows = [
        (9.4, "(a)  Rule-based extraction", BAND_EVAL, SAFE,
         "Frequency scoring\nand cue patterns", "Notes\nfrom transcript",
         "Fabrication structurally impossible", False),
        (5.0, "(b)  Model-selected extraction", BAND_EVAL, SAFE,
         "Language model\nselects lines", "Notes\nfrom transcript",
         "Fabrication structurally impossible", True),
        (0.6, "(c)  Abstractive generation", BAND_RISK, RISK,
         "Language model\nwrites prose", "Notes\nfrom the model",
         "Fabrication possible; citations required", False),
    ]

    for y, title, tint, accent, mid, out, verdict, is_b in rows:
        ax.add_patch(Rectangle((0.25, y), 9.5, rh, facecolor=tint,
                               edgecolor="none", zorder=1))
        ax.text(0.6, y + rh - 0.4, title, ha="left", va="center", fontsize=6.6,
                fontweight="bold", color=INK, zorder=4)

        by = y + 1.5
        box(ax, 0.5, by, 2.35, 1.2, "Transcript", fs=5.8)
        box(ax, 3.35, by, 3.0, 1.2, mid, fs=5.3)
        box(ax, 6.85, by, 2.75, 1.2, out, fs=5.3, edge=accent,
            lw=1.2)

        arrow(ax, 2.85, by + 0.6, 3.35, by + 0.6)
        arrow(ax, 6.35, by + 0.6, 6.85, by + 0.6)

        if is_b:
            ax.text(5.0, by + 1.45, "returns identifiers only", ha="center",
                    va="center", fontsize=5.4, color=ARROW, style="italic",
                    zorder=4)
            # The identifiers are resolved back against the transcript; that
            # return edge is what makes the emitted wording verbatim.
            ax.plot([8.2, 8.2], [by, by - 0.55], color=ARROW, lw=1.0, zorder=5)
            ax.plot([8.2, 1.65], [by - 0.55, by - 0.55], color=ARROW, lw=1.0,
                    zorder=5)
            arrow(ax, 1.65, by - 0.55, 1.65, by, lw=1.0)
            ax.text(5.0, by - 0.88, "lookup re-materialises the text",
                    ha="center", va="center", fontsize=5.3, color=ARROW,
                    style="italic", zorder=4)
            ax.text(5.0, y + 0.26, verdict, ha="center", va="center",
                    fontsize=6.0, color=accent, fontweight="bold", zorder=4)
        else:
            ax.text(5.0, y + 0.5, verdict, ha="center", va="center",
                    fontsize=6.0, color=accent, fontweight="bold", zorder=4)

    fig.savefig(OUT / "fig2_modes.png")
    plt.close(fig)
    return "fig2_modes.png"


# --------------------------------------------------------------------------
# Fig. 3 - the measurement that motivates the paper
# --------------------------------------------------------------------------


def fig_wer(agg_wer=6.86, legal_wer=5.90, agg_sub=1.01, legal_sub=3.07,
            n_legal=424, n_total=11620):
    """
    Two panels, because the whole finding is that they disagree.

    Measured conventionally, legal tokens look marginally SAFER than average
    on this recording. That is a transcription-convention artefact: a verbatim
    court record keeps false starts the recogniser drops, deletions swamp the
    count, and they land on repeated function words rather than case names.
    Restricted to substitutions the relationship inverts by a factor of three.
    Showing only the second panel would be the more flattering figure and the
    less honest one.
    """
    fig, axes = plt.subplots(1, 2, figsize=(COL_IN, 2.5), sharey=False)

    panels = [
        (axes[0], "All errors\n(S+D+I)", agg_wer, legal_wer, "#9DB4DC"),
        (axes[1], "Substitutions\nonly", agg_sub, legal_sub, "#9DB4DC"),
    ]
    for ax, title, a_val, l_val, base in panels:
        bars = ax.bar(["all", "legal"], [a_val, l_val], width=0.55,
                      color=[base, "#C4614F"], edgecolor=BOX_EDGE,
                      linewidth=0.7, zorder=3)
        top = max(a_val, l_val)
        for rect, val in zip(bars, [a_val, l_val]):
            ax.text(rect.get_x() + rect.get_width() / 2, val + top * 0.04,
                    f"{val:.2f}%", ha="center", va="bottom", fontsize=6.8,
                    fontweight="bold", color=INK)
        ratio = l_val / a_val if a_val else 0
        ax.set_title(title, fontsize=6.9, color=INK, pad=3)
        ax.annotate(f"{ratio:.1f}×", xy=(0.5, top * 0.55), ha="center",
                    fontsize=9.5, fontweight="bold",
                    color=RISK if ratio > 1.2 else "#7A7A7A")
        ax.set_ylim(0, top * 1.30)
        ax.tick_params(labelsize=6.6)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#999999")
            ax.spines[sp].set_linewidth(0.7)

    axes[0].set_ylabel("Error rate (%)", fontsize=7.2)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig3_wer.png")
    plt.close(fig)
    return "fig3_wer.png"


# --------------------------------------------------------------------------
# Fig. 4 - the confidentiality decision
# --------------------------------------------------------------------------


def fig_guard():
    fig, ax = canvas(4.05, (0, 10), (0, 12.0))

    bw, bh, cx = 4.3, 1.3, 6.5
    bx = cx - bw / 2

    box(ax, bx, 10.2, bw, 1.0, "Run requested", fs=6.4, bold=True)

    steps = [
        (8.2, "Material classified\nprivileged?", BOX_EDGE, 0.8),
        (5.9, "Endpoint on\nthis host?", BOX_EDGE, 0.8),
        (3.6, "Model executes\non this host?", ARROW, 1.3),
    ]
    for y, label, edge, lw in steps:
        box(ax, bx, y, bw, bh, label, fs=6.0, edge=edge, lw=lw)

    box(ax, bx, 1.5, bw, 1.0, "Proceed", fs=6.4, bold=True, fill=BAND_EVAL,
        edge=SAFE, lw=1.2)

    arrow(ax, cx, 10.2, cx, 9.5)
    for y_from, y_to in ((8.2, 7.2), (5.9, 4.9), (3.6, 2.5)):
        arrow(ax, cx, y_from, cx, y_to)
        ax.text(cx + 0.22, (y_from + y_to) / 2, "yes", fontsize=5.5,
                color=MUTED, ha="left", va="center")

    box(ax, 0.25, 8.35, 3.5, 1.0, "Public material:\nno restriction", fs=5.8,
        fill="#F2F2F2")
    arrow(ax, bx, 8.85, 3.75, 8.85)
    ax.text((bx + 3.75) / 2, 9.1, "no", fontsize=5.5, color=MUTED, ha="center")

    for y, label in ((5.9, "Refused:\nremote endpoint"),
                     (3.6, "Refused: cloud model\ndespite a local URL")):
        box(ax, 0.25, y, 3.5, bh, label, fs=5.7, fill=BAND_RISK, edge=RISK,
            lw=1.1)
        arrow(ax, bx, y + bh / 2, 3.75, y + bh / 2, color=RISK)
        ax.text((bx + 3.75) / 2, y + bh / 2 + 0.26, "no", fontsize=5.5,
                color=MUTED, ha="center")

    ax.text(2.0, 1.7, "The lower branch is the\none that looks safe.",
            ha="center", va="center", fontsize=5.7, color=RISK, style="italic",
            linespacing=1.4)

    fig.savefig(OUT / "fig4_guard.png")
    plt.close(fig)
    return "fig4_guard.png"


def main():
    for name in (fig_pipeline(), fig_modes(), fig_wer(), fig_guard()):
        p = OUT / name
        print(f"  {name:22} {p.stat().st_size / 1024:7.1f} KB")


if __name__ == "__main__":
    main()
