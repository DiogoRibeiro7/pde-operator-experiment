# Contributing

Issues and pull requests are welcome, especially ones that make the comparison fairer: a stronger baseline, a better-tuned model, or a bug that changes a number.

## Setting up

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -c requirements-lock.txt   # every package at the versions of the committed results
pip install -e ".[dev,docs]"
```

## Before opening a pull request

```bash
ruff check .
pytest
python run_experiment.py --quick      # writes results-quick/, refuses to touch results/
mkdocs build --strict
```

CI runs the same checks on every pull request, plus the whole pipeline on a quick run, and fails if anything outside `results-quick/` was written.

## Changing the results

The files in `results/` and `figures/` are the ones the article quotes. If a change alters them, regenerate everything with the **Reproduce results** workflow (Actions tab), which runs the full pipeline and uploads the new results as an artifact, and explain in the pull request which numbers moved and why.

## Releases

Releases are made by the **Release** workflow, not by hand. Run it from the Actions tab, on `main`, with the new version number: it sets the version in `pyproject.toml` and `CITATION.cff`, runs the checks and the docs build, commits the bump, tags it, creates the GitHub release with the results and figures attached, and rebuilds the docs site. If a run fails after tagging, running it again with the same version resumes and only creates the release. If the repository is enabled on Zenodo, each release is archived there with a DOI.

## Repository settings the workflows need

These are one-time settings that a workflow cannot make with its default token:

- **Settings → Pages → Source: GitHub Actions**, so the Docs workflow can deploy the site.
- If a branch ruleset protects `main`, add **GitHub Actions** to its bypass list, so the Release workflow can push the version commit.

By participating you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
