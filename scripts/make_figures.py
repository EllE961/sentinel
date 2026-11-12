"""Render the README figure from results/sweep.json. Same visual system as forge-llm,
forge-align and council on purpose.

    python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("docs/assets")
RES = Path("results")

BG, PANEL = "#0d0c0b", "#131211"
INK, INK2, INK3 = "#ede7de", "#aaa299", "#7a736c"
GRID = "#24221f"
EMBER, TEMPER = "#d95926", "#3987e5"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "semibold",
    "axes.titlesize": 12.5, "axes.titlelocation": "left", "axes.titlepad": 12, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "xtick.color": INK3, "ytick.color": INK3, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
    "xtick.major.pad": 6, "ytick.major.pad": 6, "axes.axisbelow": True,
    "font.family": ["Segoe UI", "DejaVu Sans"], "lines.linewidth": 2.2, "lines.solid_capstyle": "round",
    "legend.frameon": False, "legend.fontsize": 9.5, "legend.labelcolor": INK2,
    "figure.dpi": 110, "savefig.dpi": 200,
})


def drift_vs_quality(rows: list[dict], out: Path) -> None:
    labels = [r["window"] for r in rows]
    x = list(range(len(rows)))
    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)
    ax2.tick_params(colors=INK3)

    l1, = ax1.plot(x, [r["genre_psi"] for r in rows], color=EMBER, marker="o",
                    label="input drift (genre PSI vs. training window)")
    l2, = ax2.plot(x, [r["ndcg@10"] for r in rows], color=TEMPER, marker="o",
                    label="output quality (frozen model's NDCG@10)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_xlabel("time window (quantile of all MovieLens-1M timestamps)")
    ax1.set_ylabel("genre PSI", color=EMBER)
    ax2.set_ylabel("NDCG@10", color=TEMPER)
    ax1.set_title("Input drift and output quality, tracked separately -- they don't move together")
    ax1.annotate("PSI spikes 8-25x here", xy=(x[-1], rows[-1]["genre_psi"]), xytext=(-95, 14),
                 textcoords="offset points", color=INK2, fontsize=9,
                 arrowprops={"arrowstyle": "-", "color": INK3, "lw": 1})
    ax1.annotate("quality doesn't follow", xy=(x[-1], rows[-1]["ndcg@10"]), xytext=(-110, -30),
                 textcoords="offset points", color=INK2, fontsize=9,
                 arrowprops={"arrowstyle": "-", "color": INK3, "lw": 1})
    ax1.legend(handles=[l1, l2], loc="upper center", bbox_to_anchor=(0.42, 1.0))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sweep_path = RES / "sweep.json"
    if sweep_path.exists():
        rows = json.loads(sweep_path.read_text())
        drift_vs_quality(rows, OUT / "drift_vs_quality.png")
    print("[make_figures] wrote figures to docs/assets/")
