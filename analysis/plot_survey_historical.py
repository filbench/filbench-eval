import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.utils import PLOT_PARAMS

plt.rcParams.update(PLOT_PARAMS)


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Plot paper data through the years.")
    parser.add_argument("--input_path", type=Path, help="Path to the paper annotations file.")
    parser.add_argument("--output_path", type=Path, default="plots/survey_historical.pdf", help="Output path to save the charts.")
    parser.add_argument("--aggregate", action="store_true", default=False, help="If set, do some aggregation.")
    parser.add_argument("--figsize", type=int, nargs=2, default=[10, 6], help="Matplotlib figure size.")
    parser.add_argument("--svg", action="store_true", default=False, help="If set, will also save an SVG version.")
    args = parser.parse_args()
    # fmt: on

    df = pd.read_csv(args.input_path, skiprows=1)
    df = df[df["Include in paper?"]].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=args.figsize)
    df["GPT-4 output"] = df["GPT-4 output"].apply(
        lambda x: str(x).title().replace("Nlp", "NLP")
    )
    # Remove 2024
    df = df[df["year"] != 2024].reset_index(drop=True)
    year_counts = df.groupby(["year", "GPT-4 output"]).size().unstack(fill_value=0)
    colormap = plt.get_cmap("tab20", len(year_counts.columns))
    colors = [colormap(i) for i in range(len(year_counts.columns))]
    year_counts.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="k")
    # handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        ncol=4,
        title="NLP Sub-Field",
        loc="lower center",
        bbox_to_anchor=(0, -2.0),
        frameon=False,
    )
    ax.set_xticklabels(year_counts.index.astype(int), rotation=90)
    ax.set_xlabel("Year")

    output_path = Path(args.output_path)
    plt.tight_layout()

    fig.savefig(output_path, bbox_inches="tight")
    if args.svg:
        print("Also saving an SVG version")
        fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")


if __name__ == "__main__":
    main()
