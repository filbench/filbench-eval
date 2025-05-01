import argparse
from pathlib import Path

import hashlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datasets import load_dataset
from functools import reduce
from statsmodels.stats.inter_rater import fleiss_kappa
from typing import Union

from scripts.utils import COLORS, PLOT_PARAMS

plt.rcParams.update(PLOT_PARAMS)


MODEL_SET = {
    "sea": [
        "aisingapore/Llama-SEA-LION-v3-70B-IT",
        "sail/Sailor2-20B-Chat",
        "sail/Sailor2-8B-Chat",
        "aisingapore/gemma2-9b-cpt-sea-lionv3-instruct",
        "SeaLLMs/SeaLLMs-v3-7B-Chat",
        "aisingapore/llama3.1-8b-cpt-sea-lionv3-instruct",
        "SeaLLMs/SeaLLMs-v3-1.5B-Chat",
    ]
}

TASK_SET = {
    "cn": [
        "filbench|balita_tgl_mcf|0",
        "filbench|cebuaner_ceb_mcf|0",
        "filbench|dengue_filipino_fil|absent|0",
        "filbench|dengue_filipino_fil|dengue|0",
        "filbench|dengue_filipino_fil|health|0",
        "filbench|dengue_filipino_fil|mosquito|0",
        "filbench|dengue_filipino_fil|sick|0",
        "filbench|firecs_fil_mcf|0",
        "filbench|sib200_tgl_mcf|0",
        "filbench|sib200_ceb_mcf|0",
        "filbench|tlunifiedner_tgl_mcf|0",
        "filbench|universalner_tgl_mcf|0",
        "filbench|universalner_ceb_mcf|0",
    ]
}


def format_model(model_name: str) -> str:
    return model_name.replace("/", "__")


def format_task(task_name: str) -> str:
    return task_name.replace("|", "_")


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Check agreement and plot.")
    parser.add_argument("--model_set", type=str, choices=["sea", "top5"], default="sea", help="Model set to check agreement on.")
    parser.add_argument("--task_set", nargs="+", choices=["cn", "ck"], default="cn", help="Task set to check model agreement on.")
    parser.add_argument("--output_path", type=Path, default="plots/agreement_mcf.pdf", help="Path to save the results.")
    parser.add_argument("--cache", type=Path, default="data/agreement_mcf.jsonl", help="Path to save the results.")
    parser.add_argument("--figsize", type=int, nargs=2, default=[6, 6], help="Matplotlib figure size.")
    parser.add_argument("--svg", action="store_true", default=False, help="If set, will also save an SVG version.")
    args = parser.parse_args()
    # fmt: on

    model_names = MODEL_SET[args.model_set]
    task_names = TASK_SET[args.task_set]

    task_model_results = get_task_model_results(model_names, task_names)
    task_agreement_table = combine_model_results(task_model_results)
    task_fleiss_kappa: dict[str, Union[str, int, float]] = [
        {
            "task": task,
            "n_samples": len(agreement_table),
            "fleiss_kappa": fleiss_kappa(
                prepare_data_for_fleiss_kappa(
                    agreement_table.drop(
                        columns=["prompt_hash", "instruction", "example", "gold"]
                    )
                )
            ),
        }
        for task, agreement_table in task_agreement_table.items()
    ]
    breakpoint()


def get_task_model_results(
    model_names: list[str], task_names: list[str]
) -> dict[str, dict[str, pd.DataFrame]]:
    task_model_results: dict[str, dict[str, pd.DataFrame]] = {}
    for task in task_names:
        task_model_results[task] = {}
        for model in model_names:
            results_ds_name = f"UD-Filipino/details_{format_model(model)}_private"
            print(f"Loading results for {model} ({task})...")
            # fmt: off
            df = load_dataset(results_ds_name, format_task(task), split="latest").to_pandas()
            df["prompt_hash"] = df["example"].apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
            df["mcf_predictions"] = df["predictions"].apply(lambda x: np.argmax([bool(idx[1]) for idx in x]))
            df["gold"] = df["gold_index"].apply(lambda x: int(x[0]))
            # fmt: on
            df = df[
                [
                    "prompt_hash",
                    "instruction",
                    "example",
                    "mcf_predictions",
                    "gold",
                ]
            ]

            task_model_results[task][model] = df
    return task_model_results


def combine_model_results(
    task_model_results: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    combined_results = {}

    # Process each task separately
    for task, model_dfs in task_model_results.items():
        if not model_dfs:
            continue

        # Find common prompt_hashes across all models for this task
        prompt_hash_sets = []
        for model, df in model_dfs.items():
            if "prompt_hash" in df.columns:
                prompt_hash_sets.append(set(df["prompt_hash"]))

        # Get intersection of prompt_hashes if there are any models
        if prompt_hash_sets:
            common_prompt_hashes = reduce(
                lambda x, y: x.intersection(y), prompt_hash_sets
            )
        else:
            continue

        # Filter each model's DataFrame to only include common prompt_hashes
        # and select only necessary columns
        filtered_dfs = {}
        for model, df in model_dfs.items():
            if "prompt_hash" in df.columns:
                filtered_df = df[df["prompt_hash"].isin(common_prompt_hashes)].copy()

                # Keep only needed columns
                cols_to_keep = ["prompt_hash", "instruction", "example", "gold"]
                if "mcf_predictions" in filtered_df.columns:
                    cols_to_keep.append("mcf_predictions")

                # Only keep columns that exist in the DataFrame
                valid_cols = [col for col in cols_to_keep if col in filtered_df.columns]
                filtered_df = filtered_df[valid_cols]

                # Rename mcf_predictions column to model name
                if "mcf_predictions" in filtered_df.columns:
                    filtered_df = filtered_df.rename(columns={"mcf_predictions": model})

                filtered_dfs[model] = filtered_df

        # Merge all DataFrames for this task
        if filtered_dfs:
            # Start with the first DataFrame as base
            base_model = next(iter(filtered_dfs))
            result_df = filtered_dfs[base_model].copy()

            # Merge with other models' DataFrames
            for model, df in filtered_dfs.items():
                if model != base_model:
                    # Only merge the model column (renamed from mcf_predictions)
                    if model in df.columns:
                        merge_cols = ["prompt_hash"]
                        result_df = pd.merge(
                            result_df,
                            df[merge_cols + [model]],
                            on=merge_cols,
                            how="inner",
                        )

            combined_results[task] = result_df

    return combined_results


def prepare_data_for_fleiss_kappa(df):
    """
    Convert a dataframe of model predictions to the format required by statsmodels.
    """
    n_categories = len(np.unique(df.values))
    n_samples = df.shape[0]

    # Initialize the table
    table = np.zeros((n_samples, n_categories))

    # Fill the table with counts
    for i in range(n_samples):
        for category in range(n_categories):
            table[i, category] = (df.iloc[i, :] == category).sum()

    return table


if __name__ == "__main__":
    main()
