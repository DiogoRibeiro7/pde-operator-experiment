"""Run the whole comparison and write results/results.json and results/arrays.npz.

    python run_experiment.py            # full run (about an hour on a 2-core CPU)
    python run_experiment.py --quick    # tiny smoke run (a couple of minutes), writes results-quick/

The other scripts in the repository (baselines.py, timing.py,
ablation_log_input.py, check_reference.py) read these outputs and add their
own result files; make_figures.py draws the figures from all of them.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import sys
import time
from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy

from elliptic1d import fields, operators, pinn, solver
from elliptic1d.config import Config, quick

OUT = "results"          # set in main(); --quick writes to results-quick/ so it never overwrites the full run


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rel_l2(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - ref, axis=-1) / np.linalg.norm(ref, axis=-1)


def summary(e: np.ndarray) -> dict:
    e = np.asarray(e, dtype=float)
    return {"mean": float(e.mean()), "median": float(np.median(e)),
            "p90": float(np.quantile(e, 0.9)), "max": float(e.max()), "n": int(e.size)}


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def median_time(fn, repeats: int) -> float:
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true", help="tiny configuration for a smoke test")
    ap.add_argument("--out", default=None, help="output folder (default: results, or results-quick with --quick)")
    args = ap.parse_args()
    cfg = quick(Config()) if args.quick else Config()
    OUT = args.out or ("results-quick" if args.quick else "results")
    os.makedirs(OUT, exist_ok=True)
    res: dict = {
        "config": asdict(cfg),
        "environment": {
            "python": sys.version.split()[0], "jax": jax.__version__, "optax": optax.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__,
            "cpu": cpu_name(), "cpu_count": os.cpu_count(), "quick": args.quick,
        },
    }

    def save() -> None:
        with open(os.path.join(OUT, "results.json"), "w") as fh:
            json.dump(res, fh, indent=1)

    # ------------------------------------------------------------------ data
    rng = np.random.default_rng(cfg.seed)
    n_pool = max(cfg.n_train_sweep + (cfg.n_train,))
    families = {cfg.train_family.name: cfg.train_family, **{f.name: f for f in cfg.shift_families}}
    pool = fields.sample(cfg.train_family, rng, n_pool)
    tests = {name: fields.sample(fam, rng, cfg.n_test) for name, fam in families.items()}
    x = solver.grid(cfg.n_grid)
    t0 = time.perf_counter()
    u_pool = solver.fd(pool, cfg.n_grid, cfg.f_const)
    res["data"] = {"n_pool": n_pool, "seconds_fd_labels_pool": time.perf_counter() - t0}
    exact = {name: {n: solver.exact(p, n, cfg.n_fine) for n in cfg.test_resolutions}
             for name, p in tests.items()}
    stats = {}
    for name, p in tests.items():
        a = fields.a(p, x)
        contrast = a.max(axis=1) / a.min(axis=1)
        stats[name] = {"contrast_median": float(np.median(contrast)),
                       "contrast_p90": float(np.quantile(contrast, 0.9)),
                       "a_min": float(a.min()), "a_max": float(a.max())}
    res["data"]["families"] = stats
    log(f"data ready: pool {n_pool}, test {cfg.n_test} per family")

    # ------------------------------------------------- 1. classical solver
    conv = {}
    for name, p in tests.items():
        sub = fields.take(p, slice(0, 50))
        conv[name] = {}
        for n in cfg.fd_convergence_grids:
            e = rel_l2(solver.fd(sub, n, cfg.f_const), solver.exact(sub, n, cfg.n_fine))
            conv[name][n] = summary(e)
    fd129 = {name: summary(rel_l2(solver.fd(p, cfg.n_grid, cfg.f_const), exact[name][cfg.n_grid]))
             for name, p in tests.items()}
    res["solver"] = {"convergence": conv, "fd_at_training_grid": fd129}
    save()
    log("solver convergence done")

    # ------------------------------------- 2. operators: training-set sweep
    kinds = ("deeponet", "fno")
    sweep = {k: {} for k in kinds}
    headline = {k: [] for k in kinds}
    for n_tr in cfg.n_train_sweep:
        for k in kinds:
            sweep[k][n_tr] = []
        for s in range(cfg.n_seeds):
            idx = np.sort(np.random.default_rng(1000 + s).choice(n_pool, n_tr, replace=False))
            p_tr = fields.take(pool, idx)
            for k in kinds:
                op = operators.fit(k, cfg, p_tr, u_pool[idx], x, seed=100 * s + (k == "fno"))
                e = rel_l2(operators.predict(op, cfg, tests["train"], cfg.n_grid),
                           exact["train"][cfg.n_grid])
                row = {"seed": s, **summary(e), "train_seconds": op.info["seconds"],
                       "compile_seconds_est": op.info["compile_seconds_est"],
                       "final_train_loss": op.info["history"][-1][1], "n_params": op.info["n_params"]}
                sweep[k][n_tr].append(row)
                log(f"sweep {k:8s} n={n_tr:5d} seed={s}: mean {row['mean']:.2e} "
                    f"median {row['median']:.2e} ({row['train_seconds']:.0f}s)")
                if n_tr == cfg.n_train:
                    headline[k].append(op)
            res["operators"] = {"sweep": sweep}
            save()
    if not all(len(v) == cfg.n_seeds for v in headline.values()):
        raise RuntimeError("n_train must be one of n_train_sweep")
    os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
    for k in kinds:
        for s, op in enumerate(headline[k]):
            with open(os.path.join(OUT, "models", f"{k}_seed{s}.pkl"), "wb") as fh:
                pickle.dump({"kind": k, "params": jax.tree_util.tree_map(np.asarray, op.params),
                             "scaler": asdict(op.scaler), "n_train": cfg.n_train}, fh)

    # -------------------------- 3. operators: distribution shift, resolution
    shift, resol, per_sample, preds129 = {k: {} for k in kinds}, {k: {} for k in kinds}, {}, {}
    for k in kinds:
        for name, p in tests.items():
            shift[k][name] = []
            for s, op in enumerate(headline[k]):
                pr = operators.predict(op, cfg, p, cfg.n_grid)
                e = rel_l2(pr, exact[name][cfg.n_grid])
                shift[k][name].append({"seed": s, **summary(e)})
                if s == 0:
                    per_sample[f"{k}_{name}"] = e
                    preds129[f"{k}_{name}"] = pr
        for name in ("train", "rough"):
            resol[k][name] = {}
            for n in cfg.test_resolutions:
                resol[k][name][n] = [
                    {"seed": s, **summary(rel_l2(operators.predict(op, cfg, tests[name], n), exact[name][n]))}
                    for s, op in enumerate(headline[k])]
    res["operators"].update({"shift": shift, "resolution": resol})
    save()
    log("shift and resolution evaluation done")

    # ------------------------------------------------------------ 4. PINNs
    forms = {"train": ("strong", "mixed"), "rough": ("strong", "mixed"),
             "high-contrast": ("strong", "mixed"), "piecewise": ("mixed",)}
    pinn_res, pinn_preds = {}, {}
    for name in tests:
        pinn_res[name] = {}
        for form in forms[name]:
            rows = []
            for i in range(cfg.n_pinn_instances):
                params, info = pinn.fit(cfg, fields.take(tests[name], i), seed=7 + i, form=form)
                u = pinn.predict(params, x)
                e = float(rel_l2(u, exact[name][cfg.n_grid][i]))
                rows.append({"instance": i, "rel_l2": e, "seconds": info["seconds"],
                             "seconds_adam": info["seconds_adam"], "seconds_lbfgs": info["seconds_lbfgs"],
                             "final_loss": (info["loss_lbfgs"] or info["loss_adam"])[-1][1],
                             "compile_seconds_est": info["compile_seconds_est"],
                             "n_params": info["n_params"]})
                if i == 0:
                    pinn_preds[f"pinn_{form}_{name}"] = u
                log(f"PINN {name:13s} {form:6s} #{i}: rel L2 {e:.2e} ({info['seconds']:.0f}s)")
            pinn_res[name][form] = rows
            res["pinn"] = pinn_res
            save()
    # the operators and the solver on exactly the same instances, for a paired comparison
    paired = {}
    for name in tests:
        ii = slice(0, cfg.n_pinn_instances)
        paired[name] = {
            "fd": rel_l2(solver.fd(fields.take(tests[name], ii), cfg.n_grid, cfg.f_const),
                         exact[name][cfg.n_grid][ii]).tolist(),
            **{k: np.mean([rel_l2(operators.predict(op, cfg, fields.take(tests[name], ii), cfg.n_grid),
                                  exact[name][cfg.n_grid][ii]) for op in headline[k]], axis=0).tolist()
               for k in kinds},
        }
    res["pinn_paired_instances"] = paired
    save()

    # ---------------------------------------------------------- 5. timing
    # Every method is timed from its own input arrays, already in memory: the
    # harmonic cell coefficients for the solver, standardised log a at the
    # sensors (DeepONet) or on the grid (FNO). Evaluating the random fields is
    # a property of this synthetic benchmark, not of any method, so it is excluded.
    p200 = tests["train"]
    a_half = solver.cell_harmonic_a(p200, cfg.n_grid)
    fd_one_times = []
    for i in range(a_half.shape[0]):
        t0 = time.perf_counter()
        solver.fd_solve_one(a_half[i], cfg.f_const)
        fd_one_times.append(time.perf_counter() - t0)
    timing = {
        "protocol": "from in-memory input arrays; field evaluation excluded; single = one field at a time",
        "fd_single_seconds": float(np.median(fd_one_times)),
        "fd_batch_seconds_per_instance":
            median_time(lambda: solver.fd_solve_batch(a_half, cfg.f_const), 5) / cfg.n_test,
        "fd_labels_seconds_per_instance": res["data"]["seconds_fd_labels_pool"] / n_pool,
        "field_to_cell_coefficients_seconds_per_instance":
            median_time(lambda: solver.cell_harmonic_a(p200, cfg.n_grid), 5) / cfg.n_test,
    }
    xs = jnp.asarray(x, jnp.float32)
    fdj = solver.fd_solver_jax()
    a32 = jnp.asarray(a_half, jnp.float32)
    jax.block_until_ready(fdj(a32[:1], cfg.f_const))
    jax.block_until_ready(fdj(a32, cfg.f_const))
    timing["fd_jax_single_seconds"] = median_time(lambda: jax.block_until_ready(fdj(a32[:1], cfg.f_const)), 200)
    timing["fd_jax_batch_seconds_per_instance"] = median_time(
        lambda: jax.block_until_ready(fdj(a32, cfg.f_const)), 20) / cfg.n_test
    for k in kinds:
        op = headline[k][0]
        f = operators._apply_jit[k]
        g = op.scaler.g(operators.model_input(k, cfg, p200, cfg.n_grid))
        g1 = g[:1]
        jax.block_until_ready(f(op.params, g1, xs))              # compile both shapes first
        jax.block_until_ready(f(op.params, g, xs))
        timing[f"{k}_single_seconds"] = median_time(lambda: jax.block_until_ready(f(op.params, g1, xs)), 200)
        timing[f"{k}_batch_seconds_per_instance"] = median_time(
            lambda: jax.block_until_ready(f(op.params, g, xs)), 20) / cfg.n_test
        timing[f"{k}_train_seconds"] = [row["train_seconds"] for row in sweep[k][cfg.n_train]]
    for form in ("strong", "mixed"):
        timing[f"pinn_{form}_train_seconds_train_family"] = [r["seconds"] for r in pinn_res["train"][form]]
    res["timing"] = timing
    save()
    log("timing done")

    # ---------------------------------------------------------- 6. arrays
    arrays = {"x": x, "sweep_sizes": np.array(cfg.n_train_sweep)}
    for name, p in tests.items():
        arrays[f"a_{name}"] = fields.a(fields.take(p, slice(0, 5)), x)
        arrays[f"exact_{name}"] = exact[name][cfg.n_grid][:5]
        arrays[f"fd_{name}"] = solver.fd(fields.take(p, slice(0, 5)), cfg.n_grid, cfg.f_const)
        a = fields.a(p, x)
        arrays[f"contrast_{name}"] = a.max(axis=1) / a.min(axis=1)
    for key, e in per_sample.items():
        arrays[f"err_{key}"] = e
    for key, pr in preds129.items():
        arrays[f"pred_{key}"] = pr[:5]
    for key, u in pinn_preds.items():
        arrays[key] = u
    np.savez_compressed(os.path.join(OUT, "arrays.npz"), **arrays)
    log("all done")


if __name__ == "__main__":
    main()
