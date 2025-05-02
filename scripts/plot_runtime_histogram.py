import argparse
from pathlib import Path
from scipy.interpolate import make_interp_spline
import numpy as np


import matplotlib.pyplot as plt
from scripts.utils import COLORS, PLOT_PARAMS

plt.rcParams.update(PLOT_PARAMS)


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Plot performance trends")
    parser.add_argument("--output_path", type=Path, default="plots/runtime_histogram.pdf", help="Path to save the results.")
    parser.add_argument("--bins", type=int, default=10, help="Number of bins in the histogram")
    parser.add_argument("--figsize", type=int, nargs=2, default=[6, 6], help="Matplotlib figure size.")
    parser.add_argument("--svg", action="store_true", default=False, help="If set, will also save an SVG version.")
    args = parser.parse_args()
    # fmt: on

    runtime_in_min = {
        "balita_tgl_mcf": 88,
        "belebele_ceb_mcf": 7 + 56 / 60,
        "belebele_fil_mcf": 8,
        "cebuaner_ceb_mcf": 7 + 10 / 60,
        "dengue_filipino_fil": 11 + 10 / 60,
        "firecs_fil_mcf": 9 + 44 / 60,
        "global_mmlu_all_tgl_mcf": 26 + 57 / 60,
        "include_tgl_mcf": 6 + 29 / 60,
        "newsphnli_fil_mcf": 46 + 38 / 60,
        "ntrex128_fil": 9 + 49 / 60,
        "sib200_ceb_mcf": 6 + 21 / 60,
        "sib200_tgl_mcf": 5 + 55 / 60,
        "stingraybench_tgl_mcf": 6 + 17 / 60,
        "tatoeba_ceb": 6 + 34 / 60,
        "tatoeba_tgl": 7 + 24 / 60,
        "tico19_tgl": 8 + 16 / 60,
        "tlunifiedner_tgl_mcf": 6 + 43 / 60,
        "universalner_tgl_mcf": 6 + 16 / 60,
        "universalner_ceb_mcf": 5 + 52 / 60,
        "kalahi_tgl_mcf": 7,
        "readability_ceb_mcf": 7 + 28 / 60,
    }

    fig, ax = plt.subplots(figsize=args.figsize)
    runtimes = list(runtime_in_min.values())
    print(f"Max: {max(runtimes)}, Min: {min(runtimes)}, Mean: {np.mean(runtimes)}")
    print(f"Total runtime if sequential: {sum(runtimes)}")

    ax.hist(
        runtimes,
        bins=args.bins,
        color=COLORS.get("slate"),
        edgecolor="black",
        alpha=0.7,
    )
    ax.set_xlabel("Runtime (minutes)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()

    # Save the plot
    plt.savefig(args.output_path)
    if args.svg:
        plt.savefig(args.output_path.with_suffix(".svg"))


if __name__ == "__main__":
    main()
