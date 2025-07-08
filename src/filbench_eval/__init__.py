import json
from pathlib import Path

import typer
from wasabi import msg

from .compute_score import compute_score, pretty_report

app = typer.Typer(
    name="filbench",
    help="A lightweight CLI for computing the FilBench score and reporting results to the leaderboard.",
)


@app.command(name="compute-score")
def compute_score_cmd(
    # fmt: off
    hf_path: str = typer.Argument(..., help="Path to the HF dataset containing the results for a given model."),
    output_path: Path = typer.Option(None, help="Path to the output JSON file."),
    # fmt: on
) -> None:
    """Compute the FilBench score for a given model."""
    msg.text(f"Computing score for {hf_path}...")
    model_report = compute_score(hf_path)

    if not output_path:
        output_path = Path(f"scores_{hf_path.replace('/', '___')}.json")
        msg.text(f"Saving model results to: {output_path}")

    with open(output_path, "w") as f:
        json.dump(model_report, f)

    pretty_report(model_report, output_path)


@app.command(name="submit")
def submit_cmd(
    # fmt: off
    json_path: Path = typer.Argument(..., help="Path to the JSON file containing the results."),
    # fmt: on
) -> None:
    """Submit the results to the leaderboard."""
    print(f"Submitting {json_path}...")


if __name__ == "__main__":
    app()
