"""Ablation: should the operators read 1/a instead of log a?

    python ablation_inverse_input.py                        # about a quarter of an hour
    python ablation_inverse_input.py --results results-quick

The exact solution depends on a only through running integrals of 1/a,
u = C I0 - I1 with I0 = int 1/a, I1 = int s/a and C = I1(1)/I0(1): linear in
1/a apart from the scalar C, and homogeneous of degree one in 1/a. It is also
exactly as skewed as a: g is a centred Gaussian (or has N(0, sigma^2) levels),
so exp(-g) and exp(g) have the same distribution, pointwise and as fields.
Reading a against reading 1/a therefore changes the structure of the map with
the input distribution held fixed; reading log a against reading 1/a trades
one for the other.

Trains each operator on each input with the main run's three training
subsets, seeds and budgets (so the log-a runs are the main run's headline
runs, retrained) and scores every model on the test fields of all four
families. Writes ablation_inverse_input.json, and the per-field errors in
ablation_inverse_input.npz, into the results folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from elliptic1d import fields, operators, runs, solver

ROOT = Path(__file__).resolve().parent
FEATURES = ("log_a", "a", "inv_a")
KINDS = ("deeponet", "fno")


def rel_l2(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - ref, axis=1) / np.linalg.norm(ref, axis=1)


def summary(e: np.ndarray) -> dict:
    return {"mean": float(e.mean()), "median": float(np.median(e)),
            "p90": float(np.quantile(e, 0.9)), "max": float(e.max())}


def input_statistics(pool: dict, tests: dict, x: np.ndarray) -> dict:
    """How skewed each input is, and how far each test family pushes it, in the
    standardised units the networks see (one global mean and std from the pool)."""
    out = {}
    for feat in FEATURES:
        v = operators.FEATURES[feat](pool, x)
        mu, sd = v.mean(), v.std()
        out[feat] = {"pool_skewness": float(stats.skew(v.ravel())),
                     "pool_abs_z_q999": float(np.quantile(np.abs((v - mu) / sd), 0.999)),
                     "test_max_abs_z": {name: float(np.abs((operators.FEATURES[feat](p, x) - mu) / sd).max())
                                        for name, p in tests.items()}}
    # The level of 1/a carries part of its skew; the solution scales with that level
    # exactly, and the level-free ratio C does not depend on it.
    b = fields.inv_a(pool, x)
    I0, I1 = np.trapezoid(b, x, axis=1), np.trapezoid(b * x, x, axis=1)
    out["functionals_of_inv_a"] = {"skewness_I0_at_1": float(stats.skew(I0)),
                                   "skewness_I1_at_1": float(stats.skew(I1)),
                                   "skewness_C": float(stats.skew(I1 / I0)),
                                   "skewness_inv_a_over_I0": float(stats.skew((b / I0[:, None]).ravel()))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default=str(ROOT / "results"), help="folder written by run_experiment.py")
    args = ap.parse_args()
    res_dir = Path(args.results)
    main_run, cfg = runs.load(res_dir)
    pool, tests = runs.draws(cfg)
    runs.check_tests(tests, res_dir, cfg.n_grid)
    x = solver.grid(cfg.n_grid)
    u_pool = solver.fd(pool, cfg.n_grid, cfg.f_const)
    exact = {name: solver.exact(p, cfg.n_grid, cfg.n_fine, f=cfg.f_const) for name, p in tests.items()}
    n_pool = runs.pool_size(cfg)

    errors = {}                       # (kind, feature, family) -> one array of per-field errors per run
    per_run = {k: {f: {name: [] for name in tests} for f in FEATURES} for k in KINDS}
    for s in range(cfg.n_seeds):
        idx = np.sort(np.random.default_rng(1000 + s).choice(n_pool, cfg.n_train, replace=False))
        p_tr = fields.take(pool, idx)
        for kind in KINDS:
            for feat in FEATURES:
                op = operators.fit(kind, cfg, p_tr, u_pool[idx], x, seed=100 * s + (kind == "fno"), feature=feat)
                for name, p in tests.items():
                    e = rel_l2(operators.predict(op, cfg, p, cfg.n_grid), exact[name])
                    errors.setdefault((kind, feat, name), []).append(e)
                    per_run[kind][feat][name].append({"seed": s, **summary(e),
                                                      "final_train_loss": op.info["history"][-1][1]})
                print(f"seed {s} {kind:8s} {feat:6s} "
                      + "  ".join(f"{name} {per_run[kind][feat][name][-1]['mean']:.2e}" for name in tests)
                      + f"  ({op.info['seconds']:.0f}s)", flush=True)

    mm = {k: {f: {name: float(np.mean([r["mean"] for r in per_run[k][f][name]])) for name in tests}
              for f in FEATURES} for k in KINDS}
    ratios = {k: {name: {"inv_a_over_log_a": mm[k]["inv_a"][name] / mm[k]["log_a"][name],
                         "a_over_log_a": mm[k]["a"][name] / mm[k]["log_a"][name],
                         "a_over_inv_a": mm[k]["a"][name] / mm[k]["inv_a"][name]}
                  for name in tests} for k in KINDS}

    # Where 1/a wins or loses against log a, field by field, in the first run.
    contrast = {name: (lambda v: v.max(axis=1) / v.min(axis=1))(fields.a(p, x)) for name, p in tests.items()}
    per_field = {k: {name: {
        "share_inv_a_better_than_log_a": float(np.mean(errors[(k, "inv_a", name)][0] < errors[(k, "log_a", name)][0])),
        "spearman_log_error_ratio_vs_contrast": float(stats.spearmanr(
            np.log(errors[(k, "inv_a", name)][0] / errors[(k, "log_a", name)][0]), contrast[name]).statistic)}
        for name in tests} for k in KINDS}

    reproduced = {k: {"this_run": [r["mean"] for r in per_run[k]["log_a"]["train"]],
                      "main_run": [r["mean"] for r in main_run["operators"]["sweep"][k][str(cfg.n_train)]]}
                  for k in KINDS}

    out = {"note": "three runs per setting with the main run's training subsets, seeds and budgets; "
                   "errors are relative L2 against the exact solution on the 129-point grid. FNO training "
                   "is reproducible bit for bit only on the same CPU model and software versions, so its "
                   "log-a runs can differ slightly from the main run's on other hardware",
           "environment": runs.environment(),
           "mean_error_mean_over_runs": mm,
           "ratios_of_mean_errors": ratios,
           "per_field_first_run": per_field,
           "input_statistics": input_statistics(pool, tests, x),
           "log_a_vs_main_run_train_family": reproduced,
           "per_run": per_run}
    with open(res_dir / "ablation_inverse_input.json", "w") as fh:
        json.dump(out, fh, indent=1)
    np.savez_compressed(res_dir / "ablation_inverse_input.npz",
                        **{f"{k}_{f}_{name}": np.stack(v) for (k, f, name), v in errors.items()},
                        **{f"contrast_{name}": c for name, c in contrast.items()})


if __name__ == "__main__":
    main()
