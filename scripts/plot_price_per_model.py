import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.utils import COLORS, PLOT_PARAMS

plt.rcParams.update(PLOT_PARAMS)


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Plot cost-efficiency chart.")
    parser.add_argument("--input_path", type=Path, help="Path to the leaderboard results.")
    parser.add_argument("--price_per_token_data", type=Path, help="Path to the CSV containing per-token price.")
    parser.add_argument("--figsize", type=int, nargs=2, default=[6, 6], help="Matplotlib figure size.")
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
    breakpoint()


if __name__ == "__main__":
    main()
