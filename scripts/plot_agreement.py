import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset

from scripts.utils import COLORS, PLOT_PARAMS

plt.rcParams.update(PLOT_PARAMS)


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Check agreement and plot.")
    parser.add_argument("-m", "--model_names", nargs="+", default=[], help="Model names to check agreement on.")
    parser.add_argument("-t", "--task", nargs="+", default=[], help="Tasks to check model agreement on.")
    parser.add_argument("-n", "--num_samples", type=int, default=100, help="Number of samples for computing agreement (higher is more reliable).")
    parser.add_argument("--output_path", type=Path, default="plots/impact_of_lm_size.pdf", help="Path to save the results.")
    parser.add_argument("--figsize", type=int, nargs=2, default=[6, 6], help="Matplotlib figure size.")
    parser.add_argument("--svg", action="store_true", default=False, help="If set, will also save an SVG version.")
    args = parser.parse_args()
    # fmt: on


if __name__ == "__main__":
    main()
