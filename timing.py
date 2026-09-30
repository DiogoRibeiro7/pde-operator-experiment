"""Time every method under one protocol, using the trained models saved by run_experiment.py.

    python timing.py                        # about 12 minutes; run it on an otherwise idle machine
    python timing.py --reuse-train-times    # under a minute: training times from results.json
    python timing.py --results results-quick

Protocol: every method receives the same input, a NumPy float64 array holding
the coefficient a at the grid points (the DeepONet reads its sensors, every
other point), and returns a NumPy array with u at the grid points. Everything
between the two is inside the timed region: the solver's harmonic means; the
operators' conversion to float32, transfer, logarithm and standardisation, and
the conversion of their output back to NumPy. Generating the random fields is
not timed.

For the solver and the DeepONet the faster of two reasonable implementations is
reported, and both are kept in results/timing.json:
  solver    LAPACK dgtsv (SciPy) vs banded solve, one field; NumPy vs JAX Thomas, batched
  DeepONet  trunk recomputed on every call (as trained) vs trunk evaluated once for the
            fixed output grid and cached, which is the standard way to deploy it
For reference, timing.json also records the operators with their input already
on the device as float32 ("device_resident"), which is the most favourable case
for them; the figures and the article use the common protocol.

Training costs (labels, every operator run and every PINN field of the main
run, with the same seeds) are measured again here by default, so that every
time in timing.json comes from the same machine; the spread across runs and
fields is kept next to the means. The trained models are discarded.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg.lapack import dgtsv

from elliptic1d import deeponet, fields, fno, nets, operators, pinn, runs, solver

ROOT = Path(__file__).resolve().parent


def median_time(fn, repeats: int) -> float:
    fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def load_model(res_dir: Path, kind: str, seed: int = 0) -> dict:
    with open(res_dir / "models" / f"{kind}_seed{seed}.pkl", "rb") as fh:
        m = pickle.load(fh)
    m["params"] = jax.tree_util.tree_map(jnp.asarray, m["params"])
    return m


def fd_dgtsv(a: np.ndarray, f: float = 1.0) -> np.ndarray:
    ah = 2.0 * a[:-1] * a[1:] / (a[:-1] + a[1:])
    h2 = (1.0 / (a.size - 1)) ** 2
    diag = (ah[:-1] + ah[1:]) / h2
    off = -ah[1:-1] / h2
    u = np.zeros(a.size)
    u[1:-1] = dgtsv(off, diag, off, np.full(a.size - 2, f))[3]
    return u


def fd_banded(a: np.ndarray, f: float = 1.0) -> np.ndarray:
    return solver.fd_solve_one(solver.nodal_harmonic_a(a), f)


def training_times(cfg, pool: dict, tests: dict, n_runs: int, n_pinn: int) -> dict:
    """Wall-clock training cost on this machine, compilation included, for the same runs,
    seeds and fields as run_experiment.py (the first `n_runs` operator runs, the first
    `n_pinn` training-family fields). The trained models are discarded."""
    N, f = cfg.n_grid, cfg.f_const
    x = solver.grid(N)
    a_half = solver.cell_harmonic_a(pool, N)              # evaluating the fields is not timed
    t0 = time.perf_counter()
    u_pool = solver.fd_solve_batch(a_half, f)
    out = {"source": "measured by timing.py on this machine", "n_operator_runs": n_runs, "n_pinn_fields": n_pinn,
           "fd_labels_seconds_per_instance": (time.perf_counter() - t0) / runs.pool_size(cfg)}
    for kind in ("deeponet", "fno"):
        out[f"{kind}_train_seconds"] = []
    for s in range(n_runs):
        idx = np.sort(np.random.default_rng(1000 + s).choice(runs.pool_size(cfg), cfg.n_train, replace=False))
        p_tr = fields.take(pool, idx)
        for kind in ("deeponet", "fno"):
            op = operators.fit(kind, cfg, p_tr, u_pool[idx], x, seed=100 * s + (kind == "fno"))
            out[f"{kind}_train_seconds"].append(op.info["seconds"])
    for form in ("strong", "mixed"):
        out[f"pinn_{form}_train_seconds_train_family"] = [
            pinn.fit(cfg, fields.take(tests["train"], i), seed=7 + i, form=form)[1]["seconds"] for i in range(n_pinn)]
    return out


TRAIN_KEYS = ("fd_labels_seconds_per_instance", "deeponet_train_seconds", "fno_train_seconds",
              "pinn_strong_train_seconds_train_family", "pinn_mixed_train_seconds_train_family")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default=str(ROOT / "results"), help="folder written by run_experiment.py")
    ap.add_argument("--reuse-train-times", action="store_true",
                    help="take training times from results.json instead of measuring them on this machine")
    ap.add_argument("--operator-runs", type=int, default=None,
                    help="operator training runs to time (default: all runs of the main run)")
    ap.add_argument("--pinn-fields", type=int, default=None,
                    help="PINNs per form to time (default: all PINN fields of the main run)")
    args = ap.parse_args()
    res_dir = Path(args.results)
    r, cfg = runs.load(res_dir)
    N, f = cfg.n_grid, cfg.f_const
    pool, tests = runs.draws(cfg)
    runs.check_tests(tests, res_dir, N)
    p = tests["train"]
    x = solver.grid(N)
    a = fields.a(p, x)                                   # (n_test, N) float64: the common input
    a1 = a[:1]                                           # one field, same dtype and container
    xs = jnp.asarray(x, jnp.float32)
    sens = np.linspace(0, N - 1, cfg.n_sensors).round().astype(int)
    assert np.allclose(x[sens], deeponet.sensors(cfg))
    n_b = a.shape[0]
    T: dict = {"protocol": "input: NumPy float64 a at the grid points (DeepONet: its sensors); output: NumPy u "
                           "at the grid points; all conversion and preprocessing inside the timed region; field "
                           "generation excluded; single = one field per call, batch = all test fields per call "
                           "(per-field time)",
               "n_grid": N, "n_fields_batch": int(n_b), "environment": runs.environment()}

    # ---------------------------------------------------------------- solver
    assert np.allclose(fd_dgtsv(a[0], f), fd_banded(a[0], f))
    T["fd_single_dgtsv"] = median_time(lambda: fd_dgtsv(a[0], f), 2000)
    T["fd_single_banded"] = median_time(lambda: fd_banded(a[0], f), 2000)
    for n_big in (257, 513):                               # finer grids, for the PINN comparison
        a_big = fields.a(fields.take(p, 0), solver.grid(n_big))[0]
        T[f"fd_single_dgtsv_{n_big}"] = median_time(lambda: fd_dgtsv(a_big, f), 2000)
    T["fd_batch_numpy"] = median_time(lambda: solver.fd_solve_batch(solver.nodal_harmonic_a(a), f), 50) / n_b
    thomas = jax.jit(lambda aa: solver._thomas_jax(2.0 * aa[:, :-1] * aa[:, 1:] / (aa[:, :-1] + aa[:, 1:]), f))
    assert np.allclose(np.asarray(thomas(a)), solver.fd_solve_batch(solver.nodal_harmonic_a(a), f), rtol=1e-3)
    T["fd_batch_jax"] = median_time(lambda: np.asarray(thomas(a)), 200) / n_b

    # -------------------------------------------------------------- DeepONet
    m = load_model(res_dir, "deeponet")
    P, sc = m["params"], m["scaler"]
    mu, sd, us = sc["g_mean"], sc["g_std"], sc["u_scale"]

    @jax.jit
    def don_full(aa):
        g = (jnp.log(aa[:, sens]) - mu) / sd
        return deeponet.apply(P, g, xs) * us

    basis = xs[:, None] * (1.0 - xs[:, None]) * nets.mlp_apply(P["trunk"], (2.0 * xs - 1.0)[:, None],
                                                                final_act=jnp.tanh)      # (N, p), once
    bias = xs * (1.0 - xs) * P["b0"]

    @jax.jit
    def don_cached(aa):
        g = (jnp.log(aa[:, sens]) - mu) / sd
        return (nets.mlp_apply(P["branch"], g) @ basis.T + bias) * us

    assert np.allclose(np.asarray(don_full(a)), np.asarray(don_cached(a)), rtol=1e-4, atol=1e-6)
    T["deeponet_single_full"] = median_time(lambda: np.asarray(don_full(a1)), 2000)
    T["deeponet_single_cached_trunk"] = median_time(lambda: np.asarray(don_cached(a1)), 2000)
    T["deeponet_batch_full"] = median_time(lambda: np.asarray(don_full(a)), 200) / n_b
    T["deeponet_batch_cached_trunk"] = median_time(lambda: np.asarray(don_cached(a)), 200) / n_b

    # ------------------------------------------------------------------- FNO
    m = load_model(res_dir, "fno")
    Pf, scf = m["params"], m["scaler"]

    @jax.jit
    def fno_run(aa):
        return fno.apply(Pf, (jnp.log(aa) - scf["g_mean"]) / scf["g_std"], xs) * scf["u_scale"]

    T["fno_single"] = median_time(lambda: np.asarray(fno_run(a1)), 1000)
    T["fno_batch"] = median_time(lambda: np.asarray(fno_run(a)), 50) / n_b

    # ------------------- reference only: operators with their input already on the device
    a1_dev, a_dev = jnp.asarray(a1, jnp.float32), jnp.asarray(a, jnp.float32)
    T["device_resident"] = {
        "note": "input already a float32 JAX array, output left on the device; not the common protocol",
        "deeponet_single_cached_trunk": median_time(lambda: jax.block_until_ready(don_cached(a1_dev)), 2000),
        "deeponet_batch_cached_trunk": median_time(lambda: jax.block_until_ready(don_cached(a_dev)), 200) / n_b,
        "fno_single": median_time(lambda: jax.block_until_ready(fno_run(a1_dev)), 1000),
        "fno_batch": median_time(lambda: jax.block_until_ready(fno_run(a_dev)), 50) / n_b,
        "fd_batch_jax": median_time(lambda: jax.block_until_ready(thomas(a_dev)), 200) / n_b,
    }

    # --------------------------------------- arithmetic per field (XLA's own count)
    def flops(fn, arg):
        ca = fn.lower(arg).compile().cost_analysis()
        ca = ca[0] if isinstance(ca, (list, tuple)) else ca
        return float(ca.get("flops", float("nan")))

    T["flops_per_field"] = {"deeponet_cached_trunk": flops(don_cached, a1_dev), "fno": flops(fno_run, a1_dev),
                            "fd_tridiagonal_estimate": 8.0 * (N - 2)}

    # ---------------------------------------------------------- best of each
    T["best"] = {
        "fd_single": min(T["fd_single_dgtsv"], T["fd_single_banded"]),
        "fd_batch": min(T["fd_batch_numpy"], T["fd_batch_jax"]),
        "deeponet_single": min(T["deeponet_single_full"], T["deeponet_single_cached_trunk"]),
        "deeponet_batch": min(T["deeponet_batch_full"], T["deeponet_batch_cached_trunk"]),
        "fno_single": T["fno_single"],
        "fno_batch": T["fno_batch"],
    }
    if args.reuse_train_times:
        tr = {"source": "results.json (the main run's machine)", "environment": r["environment"],
              **{k: r["timing"][k] for k in TRAIN_KEYS}}
    else:
        n_runs = min(args.operator_runs or cfg.n_seeds, cfg.n_seeds)
        n_pinn = min(args.pinn_fields or cfg.n_pinn_instances, cfg.n_pinn_instances)
        tr = training_times(cfg, pool, tests, n_runs, n_pinn)
    T["training"] = tr
    label = tr["fd_labels_seconds_per_instance"] * cfg.n_train
    op_s = {k: np.asarray(tr[f"{k}_train_seconds"]) + label for k in ("deeponet", "fno")}
    pn_s = {fm: np.asarray(tr[f"pinn_{fm}_train_seconds_train_family"]) for fm in ("strong", "mixed")}
    offline = {k: float(v.mean()) for k, v in op_s.items()}
    pinn_s = {fm: float(v.mean()) for fm, v in pn_s.items()}
    B = T["best"]
    T["offline_seconds"] = offline
    T["offline_seconds_range"] = {k: [float(v.min()), float(v.max())] for k, v in op_s.items()}
    T["pinn_seconds_per_field"] = pinn_s
    T["pinn_seconds_per_field_range"] = {fm: [float(v.min()), float(v.max())] for fm, v in pn_s.items()}
    # how many PINN fields one operator training costs: from the most favourable pairing
    # for the operator (its fastest run against the slowest PINN) to the least favourable
    T["breakeven_fields_vs_pinn_range"] = {
        k: [float(op_s[k].min() / max(v.max() for v in pn_s.values())),
            float(op_s[k].max() / min(v.min() for v in pn_s.values()))] for k in op_s}
    T["breakeven_queries_vs_fd"] = {
        mode: {k: (offline[k] / (B[f"fd_{mode}"] - B[f"{k}_{mode}"]) if B[f"fd_{mode}"] > B[f"{k}_{mode}"] else None)
               for k in ("deeponet", "fno")} for mode in ("single", "batch")}
    T["breakeven_fields_vs_pinn"] = {fm: {k: offline[k] / pinn_s[fm] for k in ("deeponet", "fno")} for fm in pinn_s}
    q = 1e6
    T["million_queries_seconds_single"] = {
        "fd": q * B["fd_single"], **{k: offline[k] + q * B[f"{k}_single"] for k in ("deeponet", "fno")},
        **{f"pinn_{fm}": q * s for fm, s in pinn_s.items()}}
    with open(res_dir / "timing.json", "w") as fh:
        json.dump(T, fh, indent=1)
    print(json.dumps(T, indent=1))


if __name__ == "__main__":
    main()
