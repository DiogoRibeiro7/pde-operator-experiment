# Contributing

Issues and pull requests are welcome, especially ones that make the comparison fairer: a stronger baseline, a better-tuned model, or a bug that changes a number.

## Setting up

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev,docs]"
```

## Before opening a pull request

```bash
ruff check .
pytest
python run_experiment.py --quick      # writes results-quick/, never touches results/
mkdocs build --strict
```

The same checks run in CI on every pull request.

## Changing the results

The files in `results/` and `figures/` are the ones the article quotes. If a change alters them, regenerate everything with the **Reproduce results** workflow (Actions tab), which runs the full pipeline and uploads the new results as an artifact, and explain in the pull request which numbers moved and why.

## Releases

Releases are made by the **Release** workflow, not by hand. Run it from the Actions tab with the new version number: it sets the version in `pyproject.toml` and `CITATION.cff`, runs the checks, commits the bump, tags it, creates the GitHub release and attaches the results and figures. If the repository is enabled on Zenodo, each release is archived there with a DOI.

By participating you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
