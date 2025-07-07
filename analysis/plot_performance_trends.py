import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.utils import PLOT_PARAMS, CATEGORY_COLORS, CATEGORY_2_CODE, COLORS

plt.rcParams.update(PLOT_PARAMS)


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Plot performance trends")
    parser.add_argument("--input_path", type=Path, help="Path to the leaderboard results.")
    parser.add_argument("--output_path", type=Path, default="plots/performance_trends.pdf", help="Path to save the results.")
    parser.add_argument("--figsize", type=int, nargs=2, default=[14, 7], help="Matplotlib figure size.")
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

    # Stdev perf across all
    std = df[numeric_cols].std().to_dict()
    print("Stdev", std)

    fig, ax = plt.subplots(figsize=args.figsize)
    df_melted = df.melt(
        id_vars=["Type"],
        value_vars=[
            "Cultural Knowledge",
            "Classical NLP",
            "Reading Comprehension",
            "Generation",
        ],
        var_name="Category",
        value_name="Score",
    )

    # Create box plot
    categories = df_melted["Category"].unique()
    colors = [
        CATEGORY_COLORS.get(CATEGORY_2_CODE.get(category, ""), "white")
        for category in categories
    ]

    for i, category in enumerate(categories):
        category_data = df_melted[df_melted["Category"] == category]
        ax.boxplot(
            category_data["Score"],
            positions=[i + 1],
            widths=0.4,
            patch_artist=True,
            boxprops=dict(facecolor=colors[i], color="black", alpha=0.8, linewidth=2),
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(color="black", linewidth=2),
            capprops=dict(color="black", linewidth=2),
            flierprops=dict(marker="o", color="black", alpha=0.5),
        )

    ax.set_xticks(range(1, len(categories) + 1))
    ax.set_xticklabels(categories)
    ax.grid(True)

    # Add scatter plot for individual scores
    for idx, category in enumerate(df_melted["Category"].unique()):
        category_data = df_melted[df_melted["Category"] == category]
        ax.scatter(
            x=[idx + 1] * len(category_data),
            y=category_data["Score"],
            alpha=0.6,
            s=80,
            color=COLORS.get("slate"),
            edgecolor="k",
        )

    # Remove clutter
    ax.grid(True)
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    # ax.spines["left"].set_visible(False)
    # ax.get_legend().remove()
    fig.suptitle("")
    ax.set_title("")
    ax.set_ylabel("Aggregated score\nacross tasks")

    output_path = Path(args.output_path)
    plt.tight_layout()

    fig.savefig(output_path, bbox_inches="tight")
    if args.svg:
        print("Also saving an SVG version")
        fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")


if __name__ == "__main__":
    main()
