import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.utils import COLORS, PLOT_PARAMS

plt.rcParams.update(PLOT_PARAMS)

# From Ely: https://docs.google.com/spreadsheets/d/1oW_4X1XLwFDyRhLnaq8OkZ84DaIfF-JsNy2DdTv0ji4/edit?usp=sharing
GENERATION_RESULTS = {
    "CohereLabs/aya-expanse-32b": {
        "mult": "Multilingual",
        "w_ceb": {0: 3.45, 1: 9.05, 3: 9.30, 5: 9.63},
    },
    "gpt-4o-2024-08-06": {
        "mult": "Multilingual",
        "w_ceb": {0: 18.45, 1: 27.65, 3: 28.66, 5: 29.08},
    },
    "gpt-4o-mini": {
        "mult": "Multilingual",
        "w_ceb": {0: 13.30, 1: 23.33, 3: 27.00, 5: 27.37},
    },
    "sail/Sailor2-20B-Chat": {
        "mult": "SEA-specific",
        "w_ceb": {0: 7.79, 1: 11.31, 3: 11.60, 5: 12.29},
    },
    "Qwen/Qwen-2.5-7B-Instruct": {
        "mult": "Multilingual",
        "w_ceb": {0: 2.79, 1: 7.03, 3: 7.54, 5: 7.91},
    },
    "SeaLLMs/SeaLLMs-v3-1.5B-Chat": {
        "mult": "SEA-specific",
        "w_ceb": {0: 1.55, 1: 5.65, 3: 6.58, 5: 6.74},
    },
}


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Plot k-shot generation results.")
    parser.add_argument("--output_path", type=Path, default="plots/generation_results.pdf", help="Path to save the results.")
    parser.add_argument("--figsize", type=int, nargs=2, default=[6, 8], help="Matplotlib figure size.")
    parser.add_argument("--svg", action="store_true", default=False, help="If set, will also save an SVG version.")
    args = parser.parse_args()
    # fmt: on

    fig, ax = plt.subplots(figsize=args.figsize)
    colors = {
        "Multilingual": COLORS.get("warm_blue"),
        "SEA-specific": COLORS.get("warm_crest"),
    }

    for model, data in GENERATION_RESULTS.items():
        category = data.get("mult")
        scores = data.get("w_ceb")

        kshot = list(scores.keys())
        genscore = list(scores.values())
        ax.plot(
            kshot,
            genscore,
            color=colors.get(category),
            label=category,
            marker="o",
            linestyle="--",
        )

    ax.grid(color="gray", alpha=0.2, which="both")
    ax.set_xticks([0, 1, 3, 5])
    ax.set_xlabel("Number of demonstrations ($k$-shot)")
    ax.set_ylabel("Generation Score")
    ax.set_ylim(top=35)

    output_path = Path(args.output_path)

    plt.tight_layout()

    fig.savefig(output_path, bbox_inches="tight")
    if args.svg:
        print("Also saving an SVG version")
        fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")


if __name__ == "__main__":
    main()
