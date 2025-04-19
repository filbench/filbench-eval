import argparse
import pandas
from pathlib import Path
import matplotlib.pyplot as plt

from plot.utils import COLORS, PLOT_PARAMS

plt.rcParams.update(PLOT_PARAMS)


def get_args():
    parser = argparse.ArgumentParser(description="Plot impact of LM size.")
    # fmt: off
    parser.add_argument("--input_path", type=Path, help="Path to the leaderboard results.")
    parser.add_argument("--figsize", type=int, nargs=2, default=[10, 10], help="Matplotlib figure size.")
    # fmt: on
    return parser.parse_args()


def main():
    pass


if __name__ == "__main__":
    main()
