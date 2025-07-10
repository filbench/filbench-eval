# /// script
# dependencies = ["matplotlib"]
# ///
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from analysis.utils import COLORS, PLOT_PARAMS

plt.rcParams.update(PLOT_PARAMS)

# From Ely: https://docs.google.com/spreadsheets/d/1oW_4X1XLwFDyRhLnaq8OkZ84DaIfF-JsNy2DdTv0ji4/edit?usp=sharing
GENERATION_RESULTS = {
    "CohereLabs/aya-expanse-32b": {
        "mult": "Multilingual",
        "w_ceb": {0: 4.58, 1: 9.21, 3: 9.92, 5: 13.64},
    },
    "gpt-4o-2024-08-06": {
        "mult": "Multilingual",
        "w_ceb": {0: 46.48, 1: 45.08, 3: 50.52, 5: 56.33},
    },
    "gpt-4o-mini": {
        "mult": "Multilingual",
        "w_ceb": {0: 23.28, 1: 36.02, 3: 40.46, 5: 46.96},
    },
    "sail/Sailor2-20B-Chat": {
        "mult": "SEA-specific",
        "w_ceb": {0: 19.47, 1: 18.02, 3: 19.10, 5: 21.90},
    },
    "Qwen/Qwen-2.5-7B-Instruct": {
        "mult": "Multilingual",
        "w_ceb": {0: 4.19, 1: 6.86, 3: 7.49, 5: 10.58},
    },
    "SeaLLMs/SeaLLMs-v3-1.5B-Chat": {
        "mult": "SEA-specific",
        "w_ceb": {0: 1.68, 1: 3.61, 3: 3.80, 5: 7.34},
    },
    "SeaLLMs/SeaLLMs-v3-7B-Chat": {
        "mult": "SEA-specific",
        "w_ceb": {0: 3.62, 1: 7.02, 3: 7.88, 5: 11.28},
    },
    "neulab/Pangea-7B": {
        "mult": "Multilingual",
        "w_ceb": {0: 3.15, 1: 6.11, 3: 5.37, 5: 8.83},
    },
    "aisingapore/llama3.1-8b-cpt-sealionv3-instruct": {
        "mult": "SEA-specific",
        "w_ceb": {0: 10.05, 1: 15.09, 3: 14.68, 5: 21.01},
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
    ax.set_ylim(top=60)

    output_path = Path(args.output_path)

    plt.tight_layout()

    fig.savefig(output_path, bbox_inches="tight")
    if args.svg:
        print("Also saving an SVG version")
        fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")


if __name__ == "__main__":
    main()
