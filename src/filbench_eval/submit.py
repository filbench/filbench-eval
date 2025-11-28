import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import typer
from huggingface_hub import HfApi
from rich.console import Console
from rich.table import Table
from wasabi import msg


def submit(
    json_path: Path,
    submissions_dataset: str = "filbench/filbench-results-submission",
    dry_run: bool = False,
) -> dict[str, Any]:
    print(f"Submitting {json_path}...")
    with open(json_path, "r") as f:
        results_dict = json.load(f)

    # fmt: off
    hf_id: str = typer.prompt("🤗 Model Name or HuggingFace ID (e.g., Qwen/Qwen3-32B)")
    contact: str = typer.prompt("✉️  E-mail")
    model_url_input = input("🌐 Model URL (optional, press Enter to skip): ").strip()
    if model_url_input:
        model_url = model_url_input
        typer.echo(f"Using URL: {model_url}")
    else:
        model_url = f"https://huggingface.co/{hf_id}"
        typer.echo(f"Using default URL: {model_url}")

    m_choices = click.Choice(["Multilingual", "SEA-Specific", "Monolingual"])
    multilinguality: click.Choice = typer.prompt("🌎 Multilinguality", show_choices=True, type=m_choices)
    t_choices = click.Choice(["Base", "SFT", "Preference-aligned", "Reasoning"])
    model_type: click.Choice = typer.prompt("⭕ Model Type", show_choices=True, type=t_choices)
    num_params: int = typer.prompt("📈 Number of parameters", type=float)
    # fmt: on

    results_dict.update(
        {
            "display_metadata": {
                "hf_id": hf_id,
                "url": model_url,
                "contact": contact,
                "multilinguality": multilinguality,
                "model_type": model_type,
                "num_params": num_params,
                "submission_date": datetime.now().isoformat(),
                "hash": hashlib.sha256(f"{hf_id}".encode()).hexdigest(),
            }
        }
    )

    msg.text(f"Adding new metadata to {json_path} in the `display_metadata` field.")
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    if not dry_run:
        msg.info(f"Submitting files to {submissions_dataset}")
        api = HfApi()
        commit_message = f"[Submission] FilBench results for {hf_id})"
        commit_description = f"Filbench score: {results_dict.get('filbench_score')},\nCategory Score: {results_dict.get('category_scores')}"
        api.upload_file(
            path_or_fileobj=str(json_path),
            path_in_repo=str(json_path),
            repo_id=submissions_dataset,
            repo_type="dataset",
            commit_message=commit_message,
            commit_description=commit_description,
            create_pr=True,
        )

    return results_dict


def status(display_metadata: dict[str, Any], submissions_dataset: str, dry_run: bool):
    console = Console()
    table = Table(title="Submission Metadata")

    table.add_column("Field", justify="left", style="cyan", no_wrap=True)
    table.add_column("Value", justify="left", style="magenta")

    for key, value in display_metadata.items():
        table.add_row(key, str(value))

    console.print(table)
    if not dry_run:
        console.print(
            f"Created a PR at [bold]{submissions_dataset}/discussions[/bold]!",
        )
    else:
        console.print("Dry-run enabled. Will not submit results.")
