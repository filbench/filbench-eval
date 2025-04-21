import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.utils import COLORS, PLOT_PARAMS

plt.rcParams.update(PLOT_PARAMS)


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Plot impact of LM size.")
    parser.add_argument("--input_path", type=Path, help="Path to the leaderboard results.")
    parser.add_argument("--output_path", type=Path, default="plots/continuous_ft.pdf", help="Path to save the results.")
    parser.add_argument("--figsize", type=int, nargs=2, default=[6, 6], help="Matplotlib figure size.")
    parser.add_argument("--svg", action="store_true", default=False, help="If set, will also save an SVG version.")
    args = parser.parse_args()
    # fmt: on

    models = [
        "aisingapore/Llama-SEA-LION-v3-70B-IT",
        "meta-llama/Llama-3.1-70B-Instruct",
        "aisingapore/gemma2-9b-cpt-sea-lionv3-instruct",
        "google/gemma-2-9b-it",
        "aisingapore/llama3.1-8b-cpt-sea-lionv3-instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
    ]

    df = pd.read_csv(args.input_path)
    df = df[df["Incomplete"]]  # It is inversed because of how toggle works in Gradio
    df = df[df["# Parameters"] != -1]  # Only keep models where # Parameters != -1
    df = df[["Model", "Average"]]
    df = df[df["Model"].isin(models)]
    df = df.reset_index(drop=True)

    # Organize into pairs: base models and their SEA-finetuned versions
    pairs = [
        # fmt: off
        ("meta-llama/Llama-3.1-70B-Instruct", "aisingapore/Llama-SEA-LION-v3-70B-IT"),
        ("google/gemma-2-9b-it", "aisingapore/gemma2-9b-cpt-sea-lionv3-instruct"),
        ("meta-llama/Llama-3.1-8B-Instruct", "aisingapore/llama3.1-8b-cpt-sea-lionv3-instruct"),
        # fmt: on
    ]
    print(df)

    # Create data for plotting
    model_names = ["Llama-3.1-70B", "Gemma-2-9B", "Llama-3.1-8B"]
    base_scores = []
    sea_scores = []

    for base, sea in pairs:
        base_score = df[df["Model"] == base]["Average"].values[0]
        sea_score = df[df["Model"] == sea]["Average"].values[0]
        base_scores.append(base_score)
        sea_scores.append(sea_score)

    # Set up the plot using fig, ax approach
    fig, ax = plt.subplots(figsize=args.figsize)
    x = np.arange(len(model_names))
    width = 0.35

    # Create the bars
    ax.bar(
        x - width / 2,
        base_scores,
        width,
        label="Base Model",
        edgecolor="k",
        color=COLORS.get("blue"),
    )
    ax.bar(
        x + width / 2,
        sea_scores,
        width,
        label="SEA-focused",
        edgecolor="k",
        color=COLORS.get("warm_blue"),
    )

    # Add some text for labels, title and axes
    ax.set_ylabel("FilBench Score")
    ax.set_xticks(x)
    # ax.set_xticklabels(model_names)
    ax.set_xticklabels([])
    ax.set_ylim(50, 61)  # Set y-axis limits to better visualize differences

    # Add a legend and adjust layout
    ax.legend(frameon=False)
    fig.tight_layout()

    plt.tight_layout()

    output_path = Path(args.output_path)
    fig.savefig(output_path, bbox_inches="tight")
    if args.svg:
        print("Also saving an SVG version")
        fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")


if __name__ == "__main__":
    main()
