"""Repository-level consistency: metadata agrees across files, committed results are complete."""

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTACT = "dfr@esmad.ipp.pt"


def pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_version_agrees_between_pyproject_and_citation():
    version = pyproject()["project"]["version"]
    cff = (ROOT / "CITATION.cff").read_text()
    assert re.search(rf"^version: \"?{re.escape(version)}\"?$", cff, re.M), "CITATION.cff version differs"


def test_licence_agrees_everywhere():
    assert pyproject()["project"]["license"] == "Apache-2.0"
    assert "Apache License" in (ROOT / "LICENSE").read_text()
    assert re.search(r"^license: Apache-2\.0$", (ROOT / "CITATION.cff").read_text(), re.M)
    assert json.loads((ROOT / ".zenodo.json").read_text())["license"] == "Apache-2.0"


def test_contact_email_is_the_same_everywhere():
    assert all(a["email"] == CONTACT for a in pyproject()["project"]["authors"])
    for name in ("SECURITY.md", "CODE_OF_CONDUCT.md", "CITATION.cff"):
        text = (ROOT / name).read_text()
        assert CONTACT in text, name
        others = set(re.findall(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", text)) - {CONTACT}
        assert not others, (name, others)


def test_committed_results_are_complete():
    expected = ["results.json", "arrays.npz", "baselines.json", "timing.json", "derived.json",
                "reference_check.json", "ablation_log_input.json"]
    for name in expected:
        assert (ROOT / "results" / name).is_file(), name
    r = json.loads((ROOT / "results" / "results.json").read_text())
    assert r["environment"]["quick"] is False
    for key in ("solver", "operators", "pinn", "timing"):
        assert key in r
    for k in ("deeponet", "fno"):
        assert len(list((ROOT / "results" / "models").glob(f"{k}_seed*.pkl"))) == r["config"]["n_seeds"]


def test_figures_exist():
    for i, name in enumerate(["instances_and_errors", "accuracy_ladder", "training_size",
                              "distribution_shift", "cost"], start=1):
        assert (ROOT / "figures" / f"fig{i}_{name}.png").is_file()
