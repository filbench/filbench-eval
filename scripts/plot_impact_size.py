import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from scripts.utils import COLORS, PLOT_PARAMS

plt.rcParams.update(PLOT_PARAMS)


def main():
    parser = argparse.ArgumentParser(description="Plot impact of LM size.")
    parser.add_argument(
        "--input_path", type=Path, help="Path to the leaderboard results."
    )
    parser.add_argument(
        "--figsize", type=int, nargs=2, default=[10, 10], help="Matplotlib figure size."
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input_path)
    breakpoint()


if __name__ == "__main__":
    main()
