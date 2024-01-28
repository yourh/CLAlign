"""Generate SCOPe40-test MAP, P@1 and P@10 bar charts from fixed results."""

from _common import bar_figure, output_options, save_figure

METHODS = ["pLM-BLAST", "PLMAlign", "EBA", "TEA", "TM-align", "GTAlign", "CLAlign"]
COLORS = ["#C9D7A6", "#B5AED5", "#B2E6FD", "#B8D2CC", "#E8B2A7", "#FEEBB9", "#E6B8CF"]
RESULTS = {
    "MAP": [0.746, 0.757, 0.750, 0.689, 0.876, 0.838, 0.844],
    "P@1": [0.961, 0.958, 0.963, 0.909, 0.981, 0.983, 0.959],
    "P@10": [0.794, 0.798, 0.799, 0.734, 0.841, 0.835, 0.825],
}


def main():
    args = output_options(__doc__)
    for metric, values in RESULTS.items():
        fig = bar_figure(METHODS, values, COLORS, metric, 1.0)
        save_figure(fig, args.output_dir, f"scope40201-{metric}", args.format)


if __name__ == "__main__":
    main()
