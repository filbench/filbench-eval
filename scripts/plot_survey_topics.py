import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.utils import COLORS, PLOT_PARAMS

plt.rcParams.update(PLOT_PARAMS)


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Plot paper data through the years.")
    parser.add_argument("--input_path", type=Path, help="Path to the paper annotations file.")
    parser.add_argument("--output_path", type=Path, default="plots/survey_topics.pdf", help="Output path to save the charts.")
    parser.add_argument("--aggregate", action="store_true", default=False, help="If set, do some aggregation.")
    parser.add_argument("--figsize", type=int, nargs=2, default=[6, 6], help="Matplotlib figure size.")
    parser.add_argument("--svg", action="store_true", default=False, help="If set, will also save an SVG version.")
    args = parser.parse_args()
    # fmt: on

    df = pd.read_csv(args.input_path, skiprows=1)
    df = df[df["Include in paper?"]].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=args.figsize)
    df["GPT-4 output"] = df["GPT-4 output"].apply(
        lambda x: str(x).title().replace("Nlp", "NLP")
    )
    df["GPT-4 output"].value_counts().sort_values().plot.barh(
        ax=ax, color=COLORS.get("warm_blue"), edgecolor="k"
    )

    ax.set_title("NLP Sub-Field")
    ax.set_ylabel("")
    ax.set_xlabel(
        "Number of papers published\nthat includes any Philippine language\n(2000-2023)"
    )

    output_path = Path(args.output_path)
    plt.tight_layout()

    fig.savefig(output_path, bbox_inches="tight")
    if args.svg:
        print("Also saving an SVG version")
        fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")


if __name__ == "__main__":
    main()
