import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.utils import COLORS, PLOT_PARAMS

plt.rcParams.update(PLOT_PARAMS)


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Plot cost-efficiency chart.")
    parser.add_argument("--input_path", type=Path, help="Path to the leaderboard results.")
    parser.add_argument("--cost_data", type=Path, help="Path to the CSV containing per-token price.")
    parser.add_argument("--output_path", type=Path, default="plots/price_per_model.pdf", help="Path to save the results.")
    parser.add_argument("--figsize", type=int, nargs=2, default=[6, 6], help="Matplotlib figure size.")
    parser.add_argument("--svg", action="store_true", default=False, help="If set, will also save an SVG version.")
    args = parser.parse_args()
    # fmt: on

    scores_df = pd.read_csv(args.input_path)
    scores_df = scores_df[scores_df["Incomplete"]]
    scores_df = scores_df[["Model", "Average", "# Parameters", "Multilingual"]]
    scores_df = scores_df.reset_index(drop=True)
    price_df = pd.read_csv(args.cost_data)
    df = price_df.merge(scores_df, on="Model", how="left")
    df = df.sort_values(by=["Average"], ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=args.figsize)
    colors = {
        "Multilingual": COLORS.get("warm_blue"),
        "SEA-Focused": COLORS.get("warm_crest"),
    }
    for category, color in colors.items():
        mask = df["Multilingual"] == category
        ax.scatter(
            df.loc[mask, "price_per_output_token"],
            df.loc[mask, "Average"],
            s=120,
            alpha=0.9,
            color=color,
            label=category,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Inference Cost (\$/1M output tokens), log")
    ax.set_ylabel("FilBench Score")
    ax.grid(color="gray", alpha=0.2, which="both")

    output_path = Path(args.output_path)

    plt.tight_layout()

    fig.savefig(output_path, bbox_inches="tight")
    if args.svg:
        print("Also saving an SVG version")
        fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")

    print(df)


if __name__ == "__main__":
    main()
