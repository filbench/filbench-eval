import argparse
from pathlib import Path

import hashlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datasets import load_dataset

from scripts.utils import COLORS, PLOT_PARAMS

plt.rcParams.update(PLOT_PARAMS)


def format_model(model_name: str) -> str:
    return model_name.replace("/", "__")


def format_task(task_name: str) -> str:
    return task_name.replace("|", "_")


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Check agreement and plot.")
    parser.add_argument("-m", "--model_names", nargs="+", default=[], help="Model names to check agreement on.")
    parser.add_argument("-t", "--task_names", nargs="+", default=[], help="Tasks to check model agreement on (e.g., filbench|balita_tgl_mcf|0).")
    parser.add_argument("-n", "--num_samples", type=int, default=100, help="Number of samples for computing agreement (higher is more reliable).")
    parser.add_argument("--output_path", type=Path, default="plots/impact_of_lm_size.pdf", help="Path to save the results.")
    parser.add_argument("--figsize", type=int, nargs=2, default=[6, 6], help="Matplotlib figure size.")
    parser.add_argument("--svg", action="store_true", default=False, help="If set, will also save an SVG version.")
    args = parser.parse_args()
    # fmt: on

    task_model_results: dict[str, dict[str, pd.DataFrame]] = {}
    for model in args.model_names:
        results_ds_name = f"UD-Filipino/details_{format_model(model)}_private"
        for task in args.task_names:
            df = load_dataset(
                results_ds_name,
                format_task(task),
                split="latest",
            ).to_pandas()
            df["prompt_hash"] = df["example"].apply(
                lambda x: hashlib.sha256(x.encode()).hexdigest()
            )
            df["mcf_predictions"] = df["predictions"].apply(
                lambda x: np.argmax([bool(idx[1]) for idx in x])
            )
            df["gold"] = df["gold_index"].apply(lambda x: int(x[0]))
            df = df[
                [
                    "prompt_hash",
                    "instruction",
                    "example",
                    "mcf_predictions",
                    "gold",
                ]
            ]


if __name__ == "__main__":
    main()
