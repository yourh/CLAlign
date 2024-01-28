"""Generate all four MALIDATA TM-score scatterplots from frozen scores."""

import csv

import numpy as np

from _common import DATA, manuscript_style, output_options, plt, save_figure, sns

NAMES = {
    "plmblast": "pLM-BLAST", "plmalign": "PLMAlign", "eba": "EBA",
    "deepblast": "DeepBLAST", "tea-nw": "TEA-NW", "clalign-prott5": "CLAlign",
    "fast": "FAST", "tmalign": "TM-align", "dali": "Dali",
    "gtalign": "GTAlign", "clalign-esmif": "CLAlign-ESMIF",
}
GROUPS = {
    "sequence": ["plmblast", "plmalign", "eba", "deepblast", "tea-nw", "clalign-prott5"],
    "structure": ["fast", "tmalign", "dali", "gtalign", "clalign-esmif"],
}


def load_scores(dataset):
    with (DATA / f"{dataset}_tm_scores.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {"malidup": 241, "malisam": 130}[dataset]
    if len(rows) != expected or len({r["pair_id"] for r in rows}) != expected:
        raise ValueError(f"Incomplete or duplicate pairs in {dataset}")
    scores = {name: np.array([float(r[name]) for r in rows])
              for name in ["manual", *NAMES]}
    if any(not np.isfinite(v).all() or ((v < 0) | (v > 1)).any()
           for v in scores.values()):
        raise ValueError(f"Invalid TM-scores in {dataset}")
    return scores


def main():
    args = output_options(__doc__)
    manuscript_style()
    for dataset in ("malidup", "malisam"):
        scores = load_scores(dataset)
        for group, methods in GROUPS.items():
            fig, ax = plt.subplots(figsize=(7, 7))
            for method, color in zip(methods, sns.color_palette("viridis_r", len(methods))):
                values = scores[method]
                # Failed Dali cases are not drawn, but remain in the mean denominator.
                valid = values > 0
                ax.scatter(scores["manual"][valid], values[valid], s=80,
                           color=color, edgecolors="white", linewidths=0.64,
                           label=f"{NAMES[method]}:{values.mean():.3f}")
            ax.set_xlabel("Manual TM-score")
            ax.set_ylabel("TM-score")
            ax.legend(loc="lower right", frameon=True)
            save_figure(fig, args.output_dir, f"{dataset}-{group}", args.format)


if __name__ == "__main__":
    main()
