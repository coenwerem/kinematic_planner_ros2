#!/usr/bin/env python3
"""Render the MoveIt planner comparison figure from a recorded results file.

Reads the JSON written by `benchmark_moveit_planners.py --out`, so the figure
and the README table always come from one measured run.

Usage:
    python3 tools/benchmark_moveit_planners.py --out results/moveit_comparison.json
    python3 tools/plot_moveit_comparison.py results/moveit_comparison.json
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

_CMU_BOLD = Path("/usr/share/fonts/truetype/cmu/cmunsx.ttf")
_CMU_REG = Path("/usr/share/fonts/truetype/cmu/cmunss.ttf")
if _CMU_BOLD.exists():
    fm.fontManager.addfont(str(_CMU_BOLD))
if _CMU_REG.exists():
    fm.fontManager.addfont(str(_CMU_REG))

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["CMU Sans Serif", "DejaVu Sans", "Arial"],
    "font.size":          13,
    "axes.labelsize":     18,
    "axes.titlesize":     20,
    "axes.titleweight":   "bold",
    "axes.labelweight":   "normal",
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "axes.linewidth":     1,
    "legend.fontsize":    12,
    "xtick.labelsize":    14,
    "ytick.labelsize":    14,
    "figure.dpi":         600,
    "grid.alpha":         0.12,
    "grid.linewidth":     0.3,
    "text.usetex":        False,
    "axes.unicode_minus": False,
})

C_OK_DARK = "#2e7d32"    # solved every trial
C_VNB_DARK = "#0d47a1"   # best path quality
C_BASE_DARK = "#bf360c"  # reference lines
C_NEUTRAL = "#9e9e9e"    # no solution


def short(label):
    if "kinematic_planner" in label:
        return "RRT*\n(kinematic_planner)"
    return label.split("/")[-1]


def render(data, out_png, out_pdf):
    rows = list(data["results"])
    local = data.get("local_planner")
    if local:
        rows.append(local)
    n_moveit = len(data["results"])
    names = [short(r["planner"]) for r in rows]
    x = range(len(rows))

    solved = [r["successes"] / r["trials"] for r in rows]
    lengths = [r["path_length_rad"] for r in rows]
    times = [r["plan_time_s"] for r in rows]

    finished = [r["path_length_rad"] for r in rows if r["path_length_rad"] is not None]
    best = min(finished) if finished else None

    def bar_color(r):
        if r["successes"] == 0:
            return C_NEUTRAL
        if r["path_length_rad"] is not None and best is not None and abs(r["path_length_rad"] - best) < 1e-9:
            return C_VNB_DARK
        return C_OK_DARK

    colors = [bar_color(r) for r in rows]
    # the in-repository planner runs a separate collision stack, so it is drawn
    # hatched to mark that it is measured beside the MoveIt pipelines, not
    # under an identical harness
    hatches = ["" if i < n_moveit else "//" for i in range(len(rows))]

    fig = plt.figure(figsize=(15.0, 4.6))
    gs = gridspec.GridSpec(
        1, 3, figure=fig,
        left=0.06, right=0.99,
        bottom=0.24, top=0.84,
        wspace=0.26,
    )
    ax0, ax1, ax2 = (fig.add_subplot(gs[0, i]) for i in range(3))

    # --- solved fraction ------------------------------------------------
    ax0.bar(x, solved, color=colors, edgecolor="black", linewidth=0.7, width=0.62,
            hatch=hatches)
    ax0.set_ylim(0, 1.18)
    ax0.set_ylabel("solved fraction")
    ax0.set_title("Success rate")
    for i, r in enumerate(rows):
        ax0.text(i, r["successes"] / r["trials"] + 0.05,
                 f"{r['successes']}/{r['trials']}",
                 ha="center", va="bottom", fontsize=12)

    # --- path length ----------------------------------------------------
    plotted = [v if v is not None else 0.0 for v in lengths]
    ax1.bar(x, plotted, color=colors, edgecolor="black", linewidth=0.7, width=0.62,
            hatch=hatches)
    ceiling = max(finished) if finished else 1.0
    ax1.set_ylim(0, ceiling * 1.30)
    ax1.set_ylabel("path length (rad)")
    ax1.set_title("Path length")
    for i, v in enumerate(lengths):
        if v is None:
            ax1.text(i, ceiling * 0.04, "no\nsolution", ha="center", va="bottom",
                     fontsize=11, color=C_NEUTRAL, linespacing=1.15)
        else:
            ax1.text(i, v + ceiling * 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=12)

    # --- planning time --------------------------------------------------
    tvals = [t for t in times if t is not None]
    ax2.bar(x, [t if t is not None else 0.0 for t in times],
            color=colors, edgecolor="black", linewidth=0.7, width=0.62,
            hatch=hatches)
    ax2.set_yscale("log")
    if tvals:
        ax2.set_ylim(min(tvals) / 3.0, max(tvals) * 6.0)
    ax2.set_ylabel("plan time (s, log)")
    ax2.set_title("Planning time")
    for i, t in enumerate(times):
        if t is None:
            ax2.text(i, min(tvals) / 2.2 if tvals else 1.0, "no\nsolution",
                     ha="center", va="bottom", fontsize=11, color=C_NEUTRAL, linespacing=1.15)
        else:
            ax2.text(i, t * 1.35, f"{t*1000:.0f} ms", ha="center", va="bottom", fontsize=12)

    blocked = data["straight_line_in_collision"]
    samples = data["straight_line_samples"]
    for ax in (ax0, ax1, ax2):
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=22, ha="right", fontsize=12)
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=C_VNB_DARK, edgecolor="black", linewidth=0.7),
        plt.Rectangle((0, 0), 1, 1, facecolor=C_OK_DARK, edgecolor="black", linewidth=0.7),
        plt.Rectangle((0, 0), 1, 1, facecolor=C_NEUTRAL, edgecolor="black", linewidth=0.7),
    ]
    labels = ["shortest path", "solved", "no valid solution"]
    if local:
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white",
                                     edgecolor="black", linewidth=0.7, hatch="//"))
        labels.append("kinematic_planner (Python)")
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle("Planner comparison on the dense xArm7 scene",
                 fontsize=20, fontweight="bold", y=0.985)

    fig.savefig(str(out_png), dpi=600, bbox_inches="tight",
                facecolor="white", pad_inches=0.02)
    fig.savefig(str(out_pdf), bbox_inches="tight",
                facecolor="white", pad_inches=0.02)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="JSON from benchmark_moveit_planners.py --out")
    ap.add_argument("--png", default="media/moveit_comparison.png")
    ap.add_argument("--pdf", default="media/moveit_comparison.pdf")
    args = ap.parse_args()
    with open(args.results) as fh:
        data = json.load(fh)
    render(data, Path(args.png), Path(args.pdf))


if __name__ == "__main__":
    main()
