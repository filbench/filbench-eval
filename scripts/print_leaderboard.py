import argparse
from pathlib import Path

import pandas as pd


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Print leaderboard LaTeX.")
    parser.add_argument("--input_path", type=Path, help="Path to the leaderboard results.")
    parser.add_argument("--max_params", type=int, default=400, help="Set the maximum param size to show in graph.")
    parser.add_argument("--top_n", type=int, default=None, help="Print only top n results.")
    parser.add_argument("--aggregate", action="store_true", help="If set, will only show the aggregate results.")
    args = parser.parse_args()
    # fmt: on

    df = pd.read_csv(args.input_path)
    df = df[df["Incomplete"]]  # It is inversed because of how toggle works in Gradio
    df = df[df["# Parameters"] <= args.max_params]

    if args.top_n:
        df = df.head(args.top_n)

    if args.aggregate:
        cols = ["Model", "Average"]
        df = df[cols + [col for col in df.columns if "agg_" in col]]

    df = df.rename(columns={"Average": "FilBench Score"})
    print(df.to_latex(index=False, float_format="%.2f"))


if __name__ == "__main__":
    main()
