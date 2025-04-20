import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.utils import COLORS, PLOT_PARAMS

plt.rcParams.update(PLOT_PARAMS)


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Plot impact of LM size.")
    parser.add_argument("--input_path", type=Path, help="Path to the leaderboard results.")
    parser.add_argument("--output_path", type=Path, default="plots/impact_of_lm_size.pdf", help="Path to save the results.")
    parser.add_argument("--max_params", type=int, default=400, help="Set the maximum param size to show in graph.")
    parser.add_argument("--figsize", type=int, nargs=2, default=[10, 10], help="Matplotlib figure size.")
    parser.add_argument("--svg", action="store_true", default=False, help="If set, will also save an SVG version.")
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

    # z = np.polyfit(df["# Parameters"], df["Average"], 1)
    # p = np.poly1d(z)
    # ax.plot(df["# Parameters"], p(df["# Parameters"]), "r--", alpha=0.7)

    ax.set_xlabel("\# Parameters (B)")
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
