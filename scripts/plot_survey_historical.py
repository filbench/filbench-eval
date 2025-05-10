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
