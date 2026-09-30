"""The glue between the scripts: configuration round trip, regenerated fields, guards, docs numbers."""

import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
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


def test_regenerated_fields_are_checked_against_the_saved_arrays(tmp_path):
    """check_tests must reject a run whose saved fields differ from the regenerated ones."""
    _, cfg = runs.load(ROOT / "results")
    _, tests = runs.draws(cfg)
    A = dict(np.load(ROOT / "results" / "arrays.npz"))
    for key, bump in (("a_rough", 1e-9), ("contrast_piecewise", 1e-9)):
        tampered = dict(A)
        tampered[key] = A[key] * (1.0 + bump)
        np.savez(tmp_path / "arrays.npz", **tampered)
        with pytest.raises(RuntimeError):
            runs.check_tests(tests, tmp_path, cfg.n_grid)


def test_committed_results_are_recognised_however_the_path_is_spelled(tmp_path):
    sys.path.insert(0, str(ROOT))
    import run_experiment

    link = tmp_path / "link"
    link.symlink_to(ROOT, target_is_directory=True)
    assert run_experiment.is_committed_results(ROOT / "results")
    assert run_experiment.is_committed_results(link / "results")
    assert run_experiment.is_committed_results(ROOT / "tests" / ".." / "results")
    assert not run_experiment.is_committed_results(ROOT / "results-quick")


def test_quick_run_refuses_to_write_into_results_even_through_a_symlink(tmp_path):
    """Run on a throwaway copy of the script, so a broken guard cannot damage the committed results."""
    shutil.copy2(ROOT / "run_experiment.py", tmp_path / "run_experiment.py")
    (tmp_path / "elliptic1d").symlink_to(ROOT / "elliptic1d", target_is_directory=True)
    (tmp_path / "results").mkdir()
    sentinel = tmp_path / "results" / "results.json"
    sentinel.write_text("committed")
    (tmp_path / "alias").symlink_to(tmp_path, target_is_directory=True)
    for out in (tmp_path / "results", tmp_path / "alias" / "results"):
        proc = subprocess.run([sys.executable, str(tmp_path / "run_experiment.py"), "--quick", "--out", str(out)],
                              capture_output=True, text=True, cwd=tmp_path, timeout=120)
        assert proc.returncode != 0 and "must not write into results/" in proc.stderr
    assert sentinel.read_text() == "committed"
    assert sorted(p.name for p in (tmp_path / "results").iterdir()) == ["results.json"]


def test_docs_values_and_tables_build_from_the_committed_results():
    vals = mkdocs_hooks.values()
    assert vals and all(v and v != "n/a" for v in vals.values())
    for name, fn in mkdocs_hooks.GENERATORS.items():
        text = fn()
        assert not re.search(r"\bNone\b|\bnan\b|\binf\b", text), name      # Python values leaking through
        if name == "bibtex":
            assert "version = {" in text and "year    = {20" in text
            continue
        rows = [line for line in text.splitlines() if line.startswith("| ")]
        assert len(rows) >= 4, name          # header, separator and at least two rows
        widths = {line.count(" | ") for line in rows}
        assert len(widths) == 1, name        # every row has as many cells as the header


def test_readme_ablation_numbers_match_the_results():
    ab = json.loads((ROOT / "results" / "ablation_log_input.json").read_text())
    readme = (ROOT / "README.md").read_text()
    assert round(ab["deeponet"]["ratio_a_over_log_a"]) == 2 and "doubles the DeepONet's test error" in readme
    m = re.search(r"multiplies the FNO's by (\d\.\d)", readme)
    assert m and float(m.group(1)) == round(ab["fno"]["ratio_a_over_log_a"], 1)
