"""Generate family, superfamily and fold sensitivity-versus-query curves."""

import csv

import numpy as np

from _common import DATA, output_options, plt, save_figure

METHODS = [
    ("plmblast", "pLM-BLAST", "#000000"),
    ("plmalign", "PLMAlign", "#1f77b4"),
    ("eba", "EBA", "#2ca02c"),
    ("clalign-prott5", "CLAlign", "#ff7f0e"),
    ("tmalign", "TM-align", "#8c564b"),
    ("gtalign", "GTAlign", "#9467bd"),
    ("tea", "TEA", "#d62728"),
]


def load_sensitivities(method):
    with (DATA / "scope40201" / f"{method}.rocx").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 766 or len({r["NAME"] for r in rows}) != 766:
        raise ValueError(f"Incomplete or duplicate queries for {method}")
    return {level: np.array([float(row[level]) for row in rows])
            for level in ("FAM", "SFAM", "FOLD")}


def curve_coordinates(values):
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid sensitivity values")
    indices = np.arange(0, len(values), 20)
    return (indices + 1) / len(values), np.minimum(np.sort(values)[::-1][indices], 1)


def main():
    args = output_options(__doc__)
    plt.rcdefaults()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    results = {method: load_sensitivities(method) for method, _, _ in METHODS}
    for level in ("FAM", "SFAM", "FOLD"):
        fig, ax = plt.subplots(figsize=(5, 5))
        for method, label, color in METHODS:
            x, y = curve_coordinates(results[method][level])
            marker_indices = np.linspace(0, len(x) - 1, 15, dtype=int)
            ax.plot(x, y, color=color, label=label, linewidth=1.5,
                    marker="o", markersize=3, markevery=marker_indices.tolist())
        ax.set_xlabel("Fraction of queries", fontsize=18)
        ax.set_ylabel("Sensitivity up to the 1st FP", fontsize=18)
        ax.tick_params(labelsize=18)
        ax.spines[["top", "right"]].set_visible(False)
        if level == "FOLD":
            ax.legend(loc="upper right", bbox_to_anchor=(0.95, 0.85), frameon=False)
        save_figure(fig, args.output_dir, level, args.format)


if __name__ == "__main__":
    main()
