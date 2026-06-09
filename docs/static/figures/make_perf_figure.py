#!/usr/bin/env python3
"""Generate the 3D-DLP manipulation-performance bar figure for the ICML poster.

Controlled-comparison framing: 3D-DLP and the 2D-DLP baselines share the SAME
EC-Diffuser policy backbone (only the representation is swapped, 2D -> 3D);
EquiDiff / PerAct are different policy architectures shown as references.

Minimal, large-text version. Outputs PDF (vector) + 300-DPI PNG.
Numbers are mean success rate (%) across tasks, from the paper's results tables
(tab:2d_vs_3d for MimicGen, tab:rlbench_10task_peract for RLBench).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

OURS, D2D, REF = "ours", "d2d", "ref"
COLORS = {OURS: "#2563eb", D2D: "#aab2c0", REF: "#6b7a99"}

# (label, value, category, x-position)  -- gap before the reference bar
mimicgen = [
    ("3D-DLP\n(Ours)", 48.1, OURS, 0),
    ("2D-DLP\nmulti", 34.1, D2D, 1),
    ("2D-DLP\nsingle", 30.8, D2D, 2),
    ("EquiDiff", 47.3, REF, 3.5),
]
rlbench = [
    ("3D-DLP\n(Ours)", 74.5, OURS, 0),
    ("2D-DLP\nmulti", 67.2, D2D, 1),
    ("2D-DLP\nsingle", 66.7, D2D, 2),
    ("PerAct", 68.8, REF, 3.5),
]
panels = [("MimicGen", mimicgen), ("RLBench", rlbench)]

fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
fig.subplots_adjust(left=0.085, right=0.985, top=0.82, bottom=0.245, wspace=0.16)

for ax, (name, data) in zip(axes, panels):
    xs = [d[3] for d in data]
    vals = [d[1] for d in data]
    cats = [d[2] for d in data]
    labels = [d[0] for d in data]
    bars = ax.bar(xs, vals, width=0.68, color=[COLORS[c] for c in cats], zorder=3)
    for b, c in zip(bars, cats):
        if c == OURS:
            b.set_edgecolor("#1d4ed8")
            b.set_linewidth(1.6)

    for xi, v, c in zip(xs, vals, cats):
        ax.text(xi, v + 1.8, f"{v:.1f}", ha="center", va="bottom",
                fontsize=18, fontweight="bold",
                color="#1d4ed8" if c == OURS else "#3a4252", zorder=4)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=13.5)
    ax.set_xlim(-0.6, 4.1)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.tick_params(axis="y", labelsize=13, length=0, colors="#555")
    ax.tick_params(axis="x", length=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#e6e8ee", linewidth=1.0)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#cfd3db")
    if ax is axes[0]:
        ax.set_ylabel("Success rate (%)", fontsize=15, color="#33415a")
    ax.set_title(name, fontsize=20, fontweight="bold", color="#222", pad=10)

    # group brackets under the x-axis (axis-x in data coords, y in axes fraction)
    def bracket(x0, x1, text, color):
        y = -0.205
        ax.plot([x0, x1], [y, y], transform=ax.get_xaxis_transform(),
                color=color, lw=1.6, clip_on=False)
        ax.text((x0 + x1) / 2, y - 0.045, text, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=11, color=color, fontweight="bold",
                clip_on=False)
    bracket(-0.35, 2.35, "Same policy (EC-Diffuser)", "#2563eb")
    bracket(3.15, 3.85, "ref.", "#8a93a3")

fig.suptitle(r"Same policy (EC-Diffuser), representation swapped 2D $\rightarrow$ 3D",
             fontsize=18, fontweight="bold", color="#1f2937", y=0.955)

legend_handles = [
    Patch(facecolor=COLORS[OURS], label="3D-DLP (ours)"),
    Patch(facecolor=COLORS[D2D], label="2D-DLP — same backbone"),
    Patch(facecolor=COLORS[REF], label="Different backbone (ref.)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False,
           fontsize=13, bbox_to_anchor=(0.5, 0.005), handlelength=1.2,
           columnspacing=2.2)

outdir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(outdir, "perf_poster.pdf"), bbox_inches="tight", pad_inches=0.12)
fig.savefig(os.path.join(outdir, "perf_poster.png"), dpi=300, bbox_inches="tight", pad_inches=0.12)
print("wrote perf_poster.pdf / perf_poster.png")
