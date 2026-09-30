"""Check the reference solution itself: how much does it move if the quadrature grid is halved?

    python check_reference.py                        # seconds
    python check_reference.py --results results-quick

Writes reference_check.json into the results folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from elliptic1d import fields, runs, solver

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default=str(ROOT / "results"), help="folder written by run_experiment.py")
    args = ap.parse_args()
    res_dir = Path(args.results)
    _, cfg = runs.load(res_dir)
    rng = np.random.default_rng(cfg.seed + 1)
    out = {}
    for fam in (cfg.train_family,) + cfg.shift_families:
        if fam.kind != "smooth":
            out[fam.name] = "exact (closed form for piecewise-constant a)"
            continue
        p = fields.sample(fam, rng, 50)
        u_full = solver.exact(p, cfg.n_grid, n_fine=cfg.n_fine)
        u_half = solver.exact(p, cfg.n_grid, n_fine=(cfg.n_fine - 1) // 2 + 1)
        rel = np.linalg.norm(u_full - u_half, axis=1) / np.linalg.norm(u_full, axis=1)
        out[fam.name] = {"max_relative_change_when_halving_quadrature_grid": float(rel.max())}
    with open(res_dir / "reference_check.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
