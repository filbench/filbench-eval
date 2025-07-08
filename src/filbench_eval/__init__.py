import typer
from pathlib import Path

app = typer.Typer(
    name="filbench",
    help="A lightweight CLI for computing the FilBench score and reporting results to the leaderboard.",
)


@app.command()
def compute_score(
    # fmt: off
    hf_path: Path = typer.Argument(..., help="Path to the JSON file containing the results."),
    output_path: Path = typer.Option(..., help="Path to the output JSON file."),
    # fmt: on
) -> None:
    """Compute the FilBench score for a given model."""
    print(f"Computing score for {hf_path}...")


@app.command()
def submit(
    # fmt: off
    json_path: Path = typer.Argument(..., help="Path to the JSON file containing the results."),
    # fmt: on
) -> None:
    """Submit the results to the leaderboard."""
    print(f"Submitting {json_path}...")


if __name__ == "__main__":
    app()
