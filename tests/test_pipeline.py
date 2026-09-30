"""The glue between the scripts: configuration round trip, regenerated fields, guards, docs numbers."""

import json
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from elliptic1d import runs
from elliptic1d.config import Config, from_dict, quick

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import mkdocs_hooks  # noqa: E402


@pytest.mark.parametrize("cfg", [Config(), quick(Config())], ids=["full", "quick"])
def test_config_round_trips_through_json(cfg):
    assert from_dict(json.loads(json.dumps(asdict(cfg)))) == cfg


def test_committed_config_is_the_default():
    _, cfg = runs.load(ROOT / "results")
    assert cfg == Config()


def test_regenerated_fields_match_the_committed_run():
    _, cfg = runs.load(ROOT / "results")
    _, tests = runs.draws(cfg)
    runs.check_tests(tests, ROOT / "results", cfg.n_grid)


def test_quick_run_refuses_to_overwrite_the_committed_results():
    out = subprocess.run([sys.executable, str(ROOT / "run_experiment.py"), "--quick", "--out", str(ROOT / "results")],
                         capture_output=True, text=True, cwd=ROOT / "tests")
    assert out.returncode != 0 and "must not write into results/" in out.stderr


def test_docs_values_and_tables_build_from_the_committed_results():
    vals = mkdocs_hooks.values()
    assert all(v != "n/a" for v in vals.values())
    for name, fn in mkdocs_hooks.GENERATORS.items():
        assert fn().strip(), name


def test_readme_ablation_numbers_match_the_results():
    ab = json.loads((ROOT / "results" / "ablation_log_input.json").read_text())
    readme = (ROOT / "README.md").read_text()
    assert round(ab["deeponet"]["ratio_a_over_log_a"]) == 2 and "doubles the DeepONet's test error" in readme
    m = re.search(r"multiplies the FNO's by (\d\.\d)", readme)
    assert m and float(m.group(1)) == round(ab["fno"]["ratio_a_over_log_a"], 1)
