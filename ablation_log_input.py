"""Ablation: do the operators do better reading log a or a?

    python ablation_log_input.py                        # about three minutes
    python ablation_log_input.py --results results-quick

Trains each operator twice on the same training pairs (the first run's subset
at the headline training-set size), once reading a and once reading log a,
with the run's budgets, and scores both on the training-family test fields.
The log-a runs reproduce the first run of run_experiment.py exactly.
Writes ablation_log_input.json into the results folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from elliptic1d import fields, operators, runs, solver

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default=str(ROOT / "results"), help="folder written by run_experiment.py")
    args = ap.parse_args()
    res_dir = Path(args.results)
    _, cfg = runs.load(res_dir)
    pool, tests = runs.draws(cfg)
    runs.check_tests(tests, res_dir, cfg.n_grid)
    test = tests["train"]
    x = solver.grid(cfg.n_grid)
    u_pool = solver.fd(pool, cfg.n_grid, cfg.f_const)
    u_exact = solver.exact(test, cfg.n_grid, cfg.n_fine, f=cfg.f_const)
    idx = np.sort(np.random.default_rng(1000).choice(runs.pool_size(cfg), cfg.n_train, replace=False))
    p_tr = fields.take(pool, idx)
    out = {"note": "one run per setting: the first run's training subset and seeds; FNO training is "
                   "reproducible bit for bit only on the same CPU model and software versions",
           "environment": runs.environment()}
    for kind in ("deeponet", "fno"):
        out[kind] = {}
        for feature in ("a", "log_a"):
            op = operators.fit(kind, cfg, p_tr, u_pool[idx], x, seed=int(kind == "fno"), feature=feature)
            pred = operators.predict(op, cfg, test, cfg.n_grid)
            e = np.linalg.norm(pred - u_exact, axis=1) / np.linalg.norm(u_exact, axis=1)
            out[kind][feature] = {"mean": float(e.mean()), "median": float(np.median(e))}
            print(kind, feature, out[kind][feature], flush=True)
        out[kind]["ratio_a_over_log_a"] = out[kind]["a"]["mean"] / out[kind]["log_a"]["mean"]
    with open(res_dir / "ablation_log_input.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
