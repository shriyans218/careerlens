"""CareerLens CLI."""
import json
import sys

import click

from .feature_extraction import extract_features, features_to_vector
from .resume_reader import read_resume
from .gap_analysis import analyze_gap, resolve_career_name


@click.group()
def main():
    """CareerLens: career-fit prediction from the command line."""


@main.command("predict-resume")
@click.argument("file_path", type=click.Path(exists=True))
def predict_resume_cmd(file_path):
    from .main import predict_top_k, predict_tech_roles, tech_bundle

    with open(file_path, "rb") as f:
        raw = f.read()
    text = read_resume(file_path, raw)
    scores = extract_features(text)
    vector = features_to_vector(scores)
    predictions = predict_top_k(vector, k=5)
    tech_predictions = predict_tech_roles(tech_bundle, text, k=5) if tech_bundle else None
    click.echo(json.dumps({"top_5": predictions, "tech_top_5": tech_predictions}, indent=2))


@main.command("predict-scores")
@click.option("--linguistic", type=float, required=True)
@click.option("--musical", type=float, required=True)
@click.option("--bodily", type=float, required=True)
@click.option("--logical-mathematical", type=float, required=True)
@click.option("--spatial-visualization", type=float, required=True)
@click.option("--interpersonal", type=float, required=True)
@click.option("--intrapersonal", type=float, required=True)
@click.option("--naturalist", type=float, required=True)
def predict_scores_cmd(**kwargs):
    from .main import predict_top_k

    ordered = list(kwargs.values())
    predictions = predict_top_k(ordered, k=5)
    click.echo(json.dumps(predictions, indent=2))


@main.command("gap-report")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--career", required=True)
def gap_report_cmd(file_path, career):
    with open(file_path, "rb") as f:
        raw = f.read()
    text = read_resume(file_path, raw)
    scores = extract_features(text)
    resolved = resolve_career_name(career)
    if resolved is None:
        click.echo(f"Could not resolve career: {career}", err=True)
        sys.exit(1)
    report = analyze_gap(scores, resolved, resume_text=text)
    click.echo(json.dumps(report, indent=2))


@main.command("serve")
@click.option("--port", default=8000)
@click.option("--reload", is_flag=True)
def serve_cmd(port, reload):
    import uvicorn

    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=port, reload=reload)


if __name__ == "__main__":
    main()
