import json
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from datasets import DownloadMode, load_dataset
from rich.console import Console
from rich.table import Table


def compute_score(hf_path: str) -> dict[str, Any]:
    parsed_results = parse_outputs(hf_path)
    filbench_score, category_scores = compute_filbench_score(
        parsed_results.get("results")
    )

    model_report = deepcopy(parsed_results)
    model_report.update({"category_scores": category_scores})
    model_report.update({"filbench_score": filbench_score})
    return model_report


def pretty_report(model_report: dict[str, Any], output_path: str):
    console = Console()
    config = model_report.get("config", {})
    model_name = config.get("model_name", "Unknown Model")
    filbench_category_scores = model_report.get("category_scores", {})
    console.print(f"[bold]Model Name:[/bold] {model_name}")
    table = Table(title="FilBench Category Scores")
    table.add_column("Category", justify="left", style="cyan", no_wrap=True)
    table.add_column("Score", justify="right", style="magenta")
    filbench_score = model_report.get("filbench_score", "N/A")
    console.print(f"[bold]FilBench Score:[/bold] {filbench_score}")

    for category, score in filbench_category_scores.items():
        table.add_row(category, str(score))

    console.print(table)
    console.print(
        f"\n[bold]Submit to leaderboard:[/bold] filbench submit {output_path}"
    )


def parse_outputs(dataset_id: str, force_download: bool = False) -> dict[str, Any]:
    """Parse a dataset ID and output a dataframe containing the relevant fields

    Based from: https://huggingface.co/docs/lighteval/en/saving-and-reading-results

    dataset_id (str): The Hugging Face dataset ID.
    RETURNS (dict[str, Any]): A dictionary containing the metrics and versions for each task.
    """
    ds = load_dataset(
        dataset_id,
        "results",
        trust_remote_code=True,
        download_mode=(
            DownloadMode.FORCE_REDOWNLOAD
            if force_download
            else DownloadMode.REUSE_DATASET_IF_EXISTS
        ),
    )

    # Save all metrics and versions for each task
    metrics = {}
    versions = {}
    for run in ds.keys():
        df = ds[run].to_pandas()
        for task, result in json.loads(df.results.iloc[0]).items():
            if task != "all":
                _, benchmark, n_shots = task.split("|")
                if int(n_shots) == 0:
                    metrics[benchmark] = result

        versions.update(json.loads(df.versions.iloc[0]))

    latest_config = json.loads(ds["latest"].to_pandas().config_general.iloc[0])
    model_config = {
        "model_name": latest_config.get("model_name"),
        "model_dtype": latest_config.get("model_dtype"),
        "model_size": latest_config.get("model_size"),
    }

    return {
        "config": model_config,
        "results": metrics,
        "versions": versions,
    }


def compute_filbench_score(scores: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    results = {}
    for task in Tasks:
        task = task.value
        if scores.get(task.benchmark):
            score = scores.get(task.benchmark).get(task.metric)
            if "acc_" in task.metric:
                score = score * 100.0
            if "rougeL" in task.metric:
                score = score * 100.0
            results[task.benchmark] = score
        else:
            results[task.benchmark] = None

    # Compute weighted average for each category
    aggregate_results = {}
    for task_category in TaskCategory:
        tasks = [task.value for task in Tasks if task.value.category == task_category]
        total_category = sum([task.num_samples for task in tasks])
        weighted_total_category = 0
        for task in tasks:
            if results[task.benchmark]:
                score = results[task.benchmark]
            else:
                score = 0
            weighted_total_category += score * task.num_samples
        aggregate_results[task_category.value] = (
            weighted_total_category / total_category
        )

    average = np.mean(list(aggregate_results.values()))
    return average, aggregate_results


class TaskCategory(Enum):
    CLASSICAL_NLP = "CLASSICAL_NLP"
    READING_COMPREHENSION = "READING_COMPREHENSION"
    GENERATION = "GENERATION"
    CULTURAL_KNOWLEDGE = "CULTURAL_KNOWLEDGE"


@dataclass
class Task:
    benchmark: str  # benchmark name in the results file
    metric: str  # metric to display
    col_name: str  # column name to display
    language: str  # language being evaluated
    category: str  # choice between different task categories
    num_samples: int  # canonical number of examples


class Tasks(Enum):
    # fmt: off
    balita_tgl_mcf = Task("balita_tgl_mcf", "acc_", "🏛️ BalitaNLP", "tgl", TaskCategory.CLASSICAL_NLP, 35_177)
    belebele_ceb_mcf = Task("belebele_ceb_mcf", "acc_", "📖 Belebele (ceb)", "ceb", TaskCategory.READING_COMPREHENSION, 900)
    belebele_fil_mcf = Task("belebele_fil_mcf", "acc_", "📖 Belebele (fil)", "fil", TaskCategory.READING_COMPREHENSION, 900)
    cebuaner_ceb_mcf = Task("cebuaner_ceb_mcf", "acc_", "🏛️ CebuaNER", "ceb", TaskCategory.CLASSICAL_NLP, 1310)
    dengue_filipino_fil = Task("dengue_filipino_fil:_average", "acc_norm", "🏛️ Dengue", "fil", TaskCategory.CLASSICAL_NLP, 4015)
    firecs_fil_mcf = Task("firecs_fil_mcf", "acc_", "🏛️ FiReCS", "fil", TaskCategory.CLASSICAL_NLP, 7340)
    global_mmlu_all_tgl = Task("global_mmlu_all_tgl_mcf:_average", "acc_", "🌏 Global-MMLU", "tgl", TaskCategory.CULTURAL_KNOWLEDGE, 14_042)
    include_tgl_mcf = Task("include_tgl_mcf:_average", "acc_", "🌏 INCLUDE", "tgl", TaskCategory.CULTURAL_KNOWLEDGE, 500)
    kalahi_tgl_mcf = Task("kalahi_tgl_mcf", "acc_", "🌏 KALAHI", "tgl", TaskCategory.CULTURAL_KNOWLEDGE, 150)
    newsphnli_fil_mcf = Task("newsphnli_fil_mcf", "acc_", "📖 NewsPH NLI", "fil", TaskCategory.READING_COMPREHENSION, 90_000)
    ntrex128_fil = Task("ntrex128_fil", "rougeL", "🔢 NTREX-128", "fil", TaskCategory.GENERATION, 1997)
    readability_ceb_mcf = Task("readability_ceb_mcf", "acc_", "📖 Readability (ceb)", "ceb", TaskCategory.READING_COMPREHENSION, 350)
    sib200_ceb_mcf = Task("sib200_ceb_mcf", "acc_", "🏛️ SIB-200 (ceb)", "ceb", TaskCategory.CLASSICAL_NLP, 99)
    sib200_tgl_mcf = Task("sib200_tgl_mcf", "acc_", "🏛️ SIB-200 (tgl)", "tgl", TaskCategory.CLASSICAL_NLP, 99)
    # stingraybench_corr_tgl_mcf = Task("stingraybench_correctness_tgl_mcf", "acc_", "StingrayBench (Correctness)", "tgl", TaskCategory.CULTURAL_KNOWLEDGE, 100)
    stingraybench_sem_appropriateness_tgl_mcf = Task("stingraybench_semantic_appropriateness_tgl_mcf", "acc_", "🌏StingrayBench", "tgl", TaskCategory.CULTURAL_KNOWLEDGE, 100)
    tatoeba_ceb = Task("tatoeba_ceb", "rougeL", "🔢 Tatoeba (ceb)", "ceb", TaskCategory.GENERATION, 377)
    tatoeba_tgl = Task("tatoeba_tgl", "rougeL", "🔢 Tatoeba (tgl)", "tgl", TaskCategory.GENERATION, 2499)
    tico19_tgl = Task("tico19_tgl", "rougeL", "🔢 TICO-19", "tgl", TaskCategory.GENERATION, 971)
    tlunifiedner_tgl_mcf = Task("tlunifiedner_tgl_mcf", "acc_", "🏛️ TLUnified NER", "tgl", TaskCategory.CLASSICAL_NLP, 1579)
    universalner_ceb_mcf = Task("universalner_ceb_mcf", "acc_", "🏛️ Universal NER (ceb)", "ceb", TaskCategory.CLASSICAL_NLP, 49)
    universalner_tgl_mcf = Task("universalner_tgl_mcf", "acc_", "🏛️ Universal NER (tgl)", "tgl", TaskCategory.CLASSICAL_NLP, 56)
    # fmt: on
