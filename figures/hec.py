"""Generate four ACC/SOV bar charts using the reported notebook values."""

from _common import bar_figure, output_options, save_figure

METHODS = ["pLM-BLAST", "PLMAlign", "EBA", "DeepBLAST", "TEA-NW", "CLAlign"]
COLORS = ["#B5AED5", "#B2E6FD", "#B8D2CC", "#E8B2A7", "#D9B7C6", "#FEEBB9"]
# These are recorded plotting constants, not recomputed per-pair averages.
RESULTS = {
    ("malidup", "ACC"): ([0.305, 0.341, 0.344, 0.348, 0.363, 0.387], 0.40),
    ("malidup", "SOV"): ([0.511, 0.567, 0.569, 0.580, 0.597, 0.635], 0.66),
    ("malisam", "ACC"): ([0.240, 0.301, 0.265, 0.305, 0.319, 0.367], 0.40),
    ("malisam", "SOV"): ([0.361, 0.450, 0.381, 0.467, 0.472, 0.560], 0.60),
}


def main():
    args = output_options(__doc__)
    for (dataset, metric), (values, upper) in RESULTS.items():
        fig = bar_figure(METHODS, values, COLORS, metric, upper)
        save_figure(fig, args.output_dir, f"{dataset}-{metric}", args.format)


if __name__ == "__main__":
    main()
