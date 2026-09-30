"""Check the reference solution itself: how much does it move if the quadrature grid is halved?

    python check_reference.py

Writes results/reference_check.json.
"""

from __future__ import annotations

import json
import os

import numpy as np

from elliptic1d import fields, solver
from elliptic1d.config import Config


def main() -> None:
    cfg = Config()
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
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "reference_check.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
