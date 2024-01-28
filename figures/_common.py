"""Small shared helpers for the standalone manuscript figure scripts."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results"


def output_options(description, argv=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--format", choices=("pdf", "png"), default="pdf",
                        help="Output format (default: pdf)")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def manuscript_style():
    sns.set_theme(style="white", rc={
        "axes.linewidth": 1.2,
        "axes.edgecolor": "black",
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig, output_dir, name, file_format="pdf"):
    if file_format not in ("pdf", "png"):
        raise ValueError(f"Unsupported figure format: {file_format}")
    destination = output_dir / f"{name}.{file_format}"
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    print(destination)
    plt.close(fig)


def bar_figure(methods, values, colors, metric, upper):
    manuscript_style()
    fig, ax = plt.subplots(figsize=(7, 7))
    bars = ax.bar(methods, values, color=colors, edgecolor="black", linewidth=1.2)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.6, alpha=0.25)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=True, right=True)
    ax.set_ylim(0, upper)
    ax.set_ylabel(metric)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=14)
    fig.tight_layout()
    return fig
