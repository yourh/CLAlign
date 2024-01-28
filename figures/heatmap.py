"""Generate four case-study heatmaps from the bundled, precomputed matrices."""

import json

import numpy as np

from _common import DATA, output_options, plt, save_figure, sns

METHODS = ("plmblast", "plmalign", "eba", "clalign")


def main():
    args = output_options(__doc__)
    case = json.loads((DATA / "heatmap_case.json").read_text())
    shape = (len(case["sequence1"]), len(case["sequence2"]))
    plt.rcdefaults()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    with np.load(DATA / "heatmap_matrices.npz", allow_pickle=False) as matrices:
        for method in METHODS:
            matrix = matrices[method]
            if matrix.shape != shape or not np.isfinite(matrix).all():
                raise ValueError(f"Invalid {method} matrix: expected {shape}")
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            sns.heatmap(matrix, cmap="viridis", ax=ax)
            save_figure(fig, args.output_dir, f"{method}-heatmap", args.format)


if __name__ == "__main__":
    main()
