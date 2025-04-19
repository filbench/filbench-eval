import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from scripts.utils import COLORS, PLOT_PARAMS

plt.rcParams.update(PLOT_PARAMS)


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Plot impact of LM size.")
    parser.add_argument("--input_path", type=Path, help="Path to the leaderboard results.")
    parser.add_argument("--output_path", type=Path, default="plots/impact_of_lm_size.pdf", help="Path to save the results.")
    parser.add_argument("--max_params", type=int, default=400, help="Set the maximum param size to show in graph.")
    parser.add_argument("--figsize", type=int, nargs=2, default=[10, 10], help="Matplotlib figure size.")
    args = parser.parse_args()
    # fmt: on

    df = pd.read_csv(args.input_path)
    df = df[df["Incomplete"]]  # It is inversed because of how toggle works in Gradio
    df = df[df["# Parameters"] != -1]  # Only keep models where # Parameters != -1
    df = df[["Model", "Average", "# Parameters", "Multilingual"]]
    df = df[df["# Parameters"] <= args.max_params]
    df = df.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=args.figsize)

    colors = {
        "Multilingual": COLORS.get("warm_blue"),
        "SEA-Focused": COLORS.get("warm_crest"),
    }
    for category, color in colors.items():
        mask = df["Multilingual"] == category
        ax.scatter(
            df.loc[mask, "# Parameters"],
            df.loc[mask, "Average"],
            s=120,
            alpha=0.9,
            color=color,
            label=category,
        )

    ax.set_xlabel("\# Parameters (B)")
    ax.set_ylabel("FilBench Score")
    ax.grid(color="gray", alpha=0.2, which="both")
    # ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(args.output_path, bbox_inches="tight")


if __name__ == "__main__":
    main()
