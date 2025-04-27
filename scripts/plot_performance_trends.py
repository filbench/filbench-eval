import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scripts.utils import COLORS, PLOT_PARAMS

plt.rcParams.update(PLOT_PARAMS)


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Plot performance trends")
    parser.add_argument("--input_path", type=Path, help="Path to the leaderboard results.")
    parser.add_argument("--output_path", type=Path, default="plots/performance_trends.pdf", help="Path to save the results.")
    parser.add_argument("--figsize", type=int, nargs=2, default=[6, 6], help="Matplotlib figure size.")
    parser.add_argument("--svg", action="store_true", default=False, help="If set, will also save an SVG version.")
    args = parser.parse_args()
    # fmt: on

    df = pd.read_csv(args.input_path)
    df = df[df["Incomplete"]]  # It is inversed because of how toggle works in Gradio
    df = df[
        [
            "Model",
            # "Average",
            "agg_Cultural Knowledge",
            "agg_Classical NLP",
            "agg_Reading Comprehension",
            "agg_Generation",
        ]
    ].rename(columns={col: col[4:] for col in df.columns if col.startswith("agg_")})
    df["Type"] = "Model"

    # Best and last model
    best_model = df.iloc[0].to_dict()
    last_model = df.iloc[-1].to_dict()
    print("Best model:", best_model)
    print("Last model:", last_model)

    # Average perf across all
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    average = df[numeric_cols].mean().to_dict()
    print("Average", average)

    fig, ax = plt.subplots(figsize=args.figsize)
    pd.plotting.parallel_coordinates(
        df,
        "Type",
        cols=[
            "Cultural Knowledge",
            "Classical NLP",
            "Reading Comprehension",
            "Generation",
        ],
        color=COLORS.get("slate"),
        alpha=0.6,
    )

    # Draw best model
    ax.plot(
        [
            best_model.get("Cultural Knowledge"),
            best_model.get("Classical NLP"),
            best_model.get("Reading Comprehension"),
            best_model.get("Generation"),
        ],
        color=COLORS.get("warm_blue"),
        linewidth=2,
    )

    # Draw last model
    ax.plot(
        [
            last_model.get("Cultural Knowledge"),
            last_model.get("Classical NLP"),
            last_model.get("Reading Comprehension"),
            last_model.get("Generation"),
        ],
        color=COLORS.get("warm_crest"),
        linewidth=2,
    )

    # Draw average
    ax.plot(
        [
            average.get("Cultural Knowledge"),
            average.get("Classical NLP"),
            average.get("Reading Comprehension"),
            average.get("Generation"),
        ],
        color="k",
        linewidth=2,
        linestyle="--",
    )

    # Remove clutter
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_legend().remove()
    ax.set_ylabel("Aggregated score\nacross tasks")

    output_path = Path(args.output_path)
    plt.tight_layout()

    fig.savefig(output_path, bbox_inches="tight")
    if args.svg:
        print("Also saving an SVG version")
        fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")


if __name__ == "__main__":
    main()
