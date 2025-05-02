import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from scripts.utils import COLORS, PLOT_PARAMS

plt.rcPrams.update(PLOT_PARAMS)


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Plot performance trends")
    parser.add_argument("--output_path", type=Path, default="plots/runtime_histogram.pdf", help="Path to save the results.")
    parser.add_argument("--figsize", type=int, nargs=2, default=[6, 6], help="Matplotlib figure size.")
    parser.add_argument("--svg", action="store_true", default=False, help="If set, will also save an SVG version.")
    args = parser.parse_args()
    # fmt: on

    runtime_in_min = {
        "balita_tgl_mcf": 60,
        "belebele_ceb_mcf": 8,
        "belebele_fil_mcf": 8,
        "cebuaner_ceb_mcf": 7,
        "dengue_filipino_fil": 11,
        "firecs_fil_mcf": 10,
        "global_mmlu_all_tgl_mcf": 27,
        "include_tgl_mcf": 6,
        "newsphnli_fil_mcf": 60,
        "ntrex128_fil": 10,
        "sib200_ceb_mcf": 6,
        "sib200_tgl_mcf": 6,
        "stingraybench_tgl_mcf": 6,
        "tatoeba_ceb": 7,
        "tatoeba_tgl": 7,
        "tico19_tgl": 8,
        "tlunifiedner_tgl_mcf": 7,
        "universalner_tgl_mcf": 6,
        "universalner_ceb_mcf": 6,
        "kalahi_tgl_mcf": 7,
        "readability_ceb_mcf": 7,
    }


if __name__ == "__main__":
    main()
