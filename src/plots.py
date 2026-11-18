"""Static figures for the README / model card.

    python -m src.plots history                   # results/history_length.png (from the processed data)
    python -m src.plots results                   # results/results_ndcg.png  (from metrics.json [+ ablations.json])

Palette: validated categorical slots 1-2 (blue #2a78d6, orange #eb6834) on surface #fcfcfb;
text uses text tokens, never the series colour; hairline solid gridlines.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.config import DOMAINS, RESULTS_DIR, processed_dir

SURFACE, GRID = "#fcfcfb", "#e6e5e0"
TEXT, TEXT2 = "#0b0b0b", "#52514e"
SERIES = {"movielens": "#2a78d6", "amazon": "#eb6834"}
ACCENT, MUTED = "#2a78d6", "#b9b8b1"
LABEL = {"movielens": "MovieLens-32M", "amazon": "Amazon Movies & TV"}


def style_axes(ax) -> None:
    """Shared chart styling (also used by notebooks/data_analysis.ipynb)."""
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=9, length=0)
    ax.grid(color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)


def history_lengths(out: Path) -> None:
    from src.data import Sequences

    seq = Sequences.load(processed_dir())
    lens = np.diff(seq.offsets)
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150, facecolor=SURFACE)
    style_axes(ax)
    for d, name in enumerate(DOMAINS):
        x = np.sort(lens[seq.domain == d])
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, color=SERIES[name], linewidth=2, solid_capstyle="round",
                label=f"{LABEL[name]}  ({len(x):,} users, median {int(np.median(x))})")
        ax.annotate(LABEL[name], (x[int(0.9 * len(x))], 0.9), xytext=(8, -2), textcoords="offset points",
                    color=TEXT, fontsize=9, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("Watch-history length per user (events, log scale)", color=TEXT2, fontsize=9)
    ax.set_ylabel("Share of users", color=TEXT2, fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("History length differs by domain: long MovieLens histories, short Amazon ones",
                 loc="left", color=TEXT, fontsize=11)
    leg = ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=TEXT2)
    leg.set_title(None)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    print(f"[plots] wrote {out}")


def results(out: Path, metrics: Path, ablations: Path | None) -> None:
    m = json.loads(metrics.read_text())["results"]
    abl = json.loads(ablations.read_text()) if ablations and ablations.exists() else {}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), dpi=150, facecolor=SURFACE)
    for ax, name in zip(axes, DOMAINS):
        rows = [("Hybrid (published)", m[name]["hybrid"]["ndcg@10"], True)]
        for key, label in (("id_only", "ID-only SASRec"), (f"{name}_only", "Hybrid, single domain")):
            if key in abl and name in abl[key]:
                rows.append((label, abl[key][name]["ndcg@10"], False))
        if "content_only" in m[name]:
            rows.append(("Content only (cold-start mode)", m[name]["content_only"]["ndcg@10"], False))
        rows.append(("Popularity", m[name]["popularity"]["ndcg@10"], False))
        style_axes(ax)
        ax.grid(axis="y", visible=False)
        y = np.arange(len(rows))[::-1]
        vals = [r[1] for r in rows]
        ax.barh(y, vals, height=0.55, color=[ACCENT if r[2] else MUTED for r in rows], edgecolor=SURFACE,
                linewidth=2)
        for yi, v in zip(y, vals):
            ax.text(v + max(vals) * 0.02, yi, f"{v:.4f}", va="center", color=TEXT, fontsize=8.5)
        ax.set_yticks(y, [r[0] for r in rows], color=TEXT, fontsize=8.5)
        ax.set_xlim(0, max(vals) * 1.25)
        ax.set_title(LABEL[name], loc="left", color=TEXT, fontsize=10)
        ax.set_xlabel("NDCG@10, full-catalog ranking (test)", color=TEXT2, fontsize=8.5)
    fig.suptitle("Test NDCG@10 by model variant", x=0.01, ha="left", color=TEXT, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    print(f"[plots] wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["history", "results"])
    p.add_argument("--metrics", type=Path, default=RESULTS_DIR / "metrics.json")
    p.add_argument("--ablations", type=Path, default=RESULTS_DIR / "ablations.json")
    args = p.parse_args()
    matplotlib.use("Agg")  # headless CLI; importing this module must not change a notebook's backend
    if args.what == "history":
        history_lengths(RESULTS_DIR / "history_length.png")
    else:
        results(RESULTS_DIR / "results_ndcg.png", args.metrics, args.ablations)


if __name__ == "__main__":
    main()
