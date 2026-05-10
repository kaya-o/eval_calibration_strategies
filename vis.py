

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_STRATEGY_ORDER = [
    "FULL",
    "S-FIX",
    "S-FULL",
    "ADA",
    "EXPRESS",
    "EXPRESS-M",
    "K-EXPRESS",
]

CORRECT_STRATEGIES = {"S-FIX", "EXPRESS", "K-EXPRESS", "EXPRESS-M"}


PAPER_LABELS = {
    "FULL": "FULL",
    "S-FULL": "S-FULL",
    "S-FIX": "S-FIX",
    "ADA": "ADA",
    "EXPRESS": "EXPRESS",
    "K-EXPRESS": "5-EXPRESS",
    "EXPRESS-M": "EXPRESS-M",
}




def read_results_dump(results_dir):
    results_dir = Path(results_dir)
    aggregate_path = results_dir / "aggregate_results.csv"

    if not aggregate_path.exists():
        raise FileNotFoundError(f"Could not find {aggregate_path}")

    aggregate = pd.read_csv(aggregate_path)
    return aggregate


def prepare_aggregate_for_plot(aggregate, strategy_order=None):
    if strategy_order is None:
        strategy_order = DEFAULT_STRATEGY_ORDER

    present_strategies = [
        strategy for strategy in strategy_order
        if strategy in set(aggregate["strategy"])
    ]

    plot_df = aggregate.set_index("strategy").loc[present_strategies].reset_index()
    plot_df["label"] = plot_df["strategy"].map(PAPER_LABELS).fillna(plot_df["strategy"])
    return plot_df


def plot_simulation_4_1_style(
    results_dir,
    output_filename="simulation_4_1_style.png",
    alpha=0.4,
    strategy_order=None,
    show=True,
):
    """
    Read an experiment results dump and produce a Figure-2-style summary plot.

    The expected input is the directory produced by dump_experiment_results(...),
    containing aggregate_results.csv. The plot follows the paper's compact style:
    strategies on the x-axis, multiple metric series shown with distinct markers,
    the target miscoverage alpha as a dashed horizontal line, and theoretically
    correct strategies marked with a check symbol in the x-axis label.
    """
    results_dir = Path(results_dir)
    aggregate = read_results_dump(results_dir)
    plot_df = prepare_aggregate_for_plot(aggregate, strategy_order=strategy_order)

    x = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(6.2, 5.2))

    ax.scatter(
        x,
        plot_df["miscoverage"],
        marker="x",
        s=58,
        linewidths=1.2,
        zorder=3,
        label="Miscoverage",
    )

    ax.axhline(alpha, linestyle="--", linewidth=1.0, label=fr"target $\alpha={alpha}$")

    ax.set_ylim(0.25, 0.50)
    ax.set_xlim(-0.35, len(plot_df) - 0.65)
    ax.set_yticks(np.arange(0.25, 0.501, 0.05))
    ax.grid(True, alpha=0.22, linewidth=0.7)

    x_labels = []
    for strategy in plot_df["strategy"]:
        label = PAPER_LABELS.get(strategy, strategy)
        if strategy in CORRECT_STRATEGIES:
            label = f"{label}\n✓"
        x_labels.append(label)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.tick_params(axis="y", labelsize=9)

    for idx, row in plot_df.iterrows():
        box_x = x[idx]

        ax.text(
            box_x,
            0.278,
            f"{row['infinite_fraction']:.3f}",
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#f3c78d", edgecolor="black", linewidth=0.6),
        )
        ax.text(
            box_x,
            0.267,
            f"{row['median_interval_length']:.3f}",
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#bcd7e8", edgecolor="black", linewidth=0.6),
        )
        ax.text(
            box_x,
            0.256,
            f"{row['avg_n_calibration']:.3f}",
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#d8d8d8", edgecolor="black", linewidth=0.6),
        )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    fig.tight_layout()

    if not output_filename.lower().endswith((".png", ".pdf", ".svg")):
        output_filename = f"{output_filename}.png"
    output_path = results_dir / output_filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def plot_from_latest_results(output_root="results", **kwargs):
    output_root = Path(output_root)
    result_dirs = [path for path in output_root.iterdir() if path.is_dir()]

    if not result_dirs:
        raise FileNotFoundError(f"No result directories found under {output_root}")

    latest_dir = max(result_dirs, key=lambda path: path.stat().st_mtime)
    return plot_simulation_4_1_style(latest_dir, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_dir",
        nargs="?",
        default=None,
        help="Directory containing aggregate_results.csv. If omitted, the latest directory under ./results is used.",
    )
    parser.add_argument("--output", default="simulation_4_1_style.png")
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.results_dir is None:
        output_path = plot_from_latest_results(
            output_filename=args.output,
            alpha=args.alpha,
            show=not args.no_show,
        )
    else:
        output_path = plot_simulation_4_1_style(
            args.results_dir,
            output_filename=args.output,
            alpha=args.alpha,
            show=not args.no_show,
        )

    print(f"Wrote plot to {output_path}")