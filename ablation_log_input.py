"""Ablation: do the operators do better reading log a or a?

    python ablation_log_input.py     # about four minutes; writes results/ablation_log_input.json

Trains each operator twice on the same 1,000 pairs (the first run's subset in
the training-size sweep), once reading a and once reading log a, with the
configured budgets, and scores both on the 200 training-family test fields.
The log-a runs reproduce the first run of run_experiment.py exactly.
"""

from __future__ import annotations

import json
import os

import numpy as np

from baselines import regenerate
from elliptic1d import fields, operators, solver
from elliptic1d.config import Config

RES = "results"


def main() -> None:
    cfg = Config()
    rng = np.random.default_rng(cfg.seed)
    n_pool = max(cfg.n_train_sweep + (cfg.n_train,))
    pool = fields.sample(cfg.train_family, rng, n_pool)
    test = regenerate(cfg)["train"]
    x = solver.grid(cfg.n_grid)
    u_pool = solver.fd(pool, cfg.n_grid, cfg.f_const)
    u_exact = solver.exact(test, cfg.n_grid, cfg.n_fine)
    idx = np.sort(np.random.default_rng(1000).choice(n_pool, cfg.n_train, replace=False))
    p_tr = fields.take(pool, idx)
    out = {}
    for kind in ("deeponet", "fno"):
        out[kind] = {}
        for feature in ("a", "log_a"):
            op = operators.fit(kind, cfg, p_tr, u_pool[idx], x, seed=int(kind == "fno"), feature=feature)
            pred = operators.predict(op, cfg, test, cfg.n_grid)
            e = np.linalg.norm(pred - u_exact, axis=1) / np.linalg.norm(u_exact, axis=1)
            out[kind][feature] = {"mean": float(e.mean()), "median": float(np.median(e))}
            print(kind, feature, out[kind][feature], flush=True)
        out[kind]["ratio_a_over_log_a"] = out[kind]["a"]["mean"] / out[kind]["log_a"]["mean"]
    with open(os.path.join(RES, "ablation_log_input.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
