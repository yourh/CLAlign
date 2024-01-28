"""Generate the gap-penalty ablation plot from the recorded F1-scores."""

import numpy as np

from _common import output_options, plt, save_figure, sns

GAP_PENALTY = np.linspace(0, 1, 6)
RESULTS = {
    "Validation": [0.342, 0.335, 0.326, 0.316, 0.306, 0.302],
    "MALIDUP": [0.646, 0.636, 0.616, 0.585, 0.563, 0.542],
    "MALISAM": [0.267, 0.250, 0.237, 0.226, 0.216, 0.211],
}


def main():
    args = output_options(__doc__)
    sns.set_theme(style="white", context="talk", palette="muted")
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6, 6))
    for (label, values), marker, style in zip(RESULTS.items(), ["o", "^", "s"], ["-", "-.", "--"]):
        ax.plot(GAP_PENALTY, values, marker=marker, linestyle=style, linewidth=2, label=label)
    ax.set_title("Effect of Gap Penalty on F1-score", fontsize=18, y=1.05)
    ax.set_xlabel("Gap Penalty", fontsize=14)
    ax.set_ylabel("F1-score", fontsize=14)
    ax.set_xticks(GAP_PENALTY)
    ax.set_ylim(0.18, 0.7)
    ax.legend(fontsize=10, frameon=False, loc="upper right")
    sns.despine(ax=ax)
    fig.tight_layout()
    save_figure(fig, args.output_dir, "f1_vs_gap_penalty", args.format)


if __name__ == "__main__":
    main()
