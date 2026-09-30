"""Time every method under one protocol, using the trained models saved by run_experiment.py.

    python timing.py      # a few minutes; run it on an otherwise idle machine

Protocol: every method receives the coefficient a at the 129 grid points (the
DeepONet uses its 65 sensors, every other point) and returns u at the 129 grid
points. Whatever each method needs to do to its input is inside the timed
region: harmonic means for the solver, log and standardisation for the
operators. Generating the random fields is not.

For each method the faster of two reasonable implementations is reported, and
both are kept in results/timing.json:
  solver    LAPACK dgtsv (SciPy) vs banded solve, one field; NumPy vs JAX Thomas, batched
  DeepONet  trunk recomputed on every call (as trained) vs trunk evaluated once for the
            fixed output grid and cached, which is the standard way to deploy it
"""

from __future__ import annotations

import json
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg.lapack import dgtsv

from baselines import regenerate
from elliptic1d import deeponet, fields, fno, nets, solver
from elliptic1d.config import Config

RES = "results"
N = 129


def median_time(fn, repeats: int) -> float:
    fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def load(kind: str, seed: int = 0) -> dict:
    with open(os.path.join(RES, "models", f"{kind}_seed{seed}.pkl"), "rb") as fh:
        m = pickle.load(fh)
    m["params"] = jax.tree_util.tree_map(jnp.asarray, m["params"])
    return m


def fd_dgtsv(a: np.ndarray) -> np.ndarray:
    ah = 2.0 * a[:-1] * a[1:] / (a[:-1] + a[1:])
    h2 = (1.0 / (a.size - 1)) ** 2
    diag = (ah[:-1] + ah[1:]) / h2
    off = -ah[1:-1] / h2
    u = np.zeros(a.size)
    u[1:-1] = dgtsv(off, diag, off, np.ones(a.size - 2))[3]
    return u


def fd_banded(a: np.ndarray) -> np.ndarray:
    return solver.fd_solve_one(solver.nodal_harmonic_a(a))


def main() -> None:
    cfg = Config()
    p = regenerate(cfg)["train"]
    x = solver.grid(N)
    a = fields.a(p, x)                                   # (200, 129) float64: the common input
    a32 = jnp.asarray(a, jnp.float32)
    a1_np, a1 = a[0], a32[:1]
    xs = jnp.asarray(x, jnp.float32)
    sens = np.arange(0, N, (N - 1) // (cfg.n_sensors - 1))
    assert np.allclose(x[sens], deeponet.sensors(cfg))
    T: dict = {"protocol": "input: a at the 129 grid points (DeepONet: its 65 sensors); output: u at the "
                           "129 grid points; input preprocessing inside the timed region; field generation "
                           "excluded; single = one field per call, batch = 200 fields per call (per-field time)",
               "n_fields_batch": int(a.shape[0])}

    # ---------------------------------------------------------------- solver
    assert np.allclose(fd_dgtsv(a1_np), fd_banded(a1_np))
    T["fd_single_dgtsv"] = median_time(lambda: fd_dgtsv(a1_np), 2000)
    T["fd_single_banded"] = median_time(lambda: fd_banded(a1_np), 2000)
    for n_big in (257, 513):                               # finer grids, for the PINN comparison
        a_big = fields.a(fields.take(p, 0), solver.grid(n_big))[0]
        T[f"fd_single_dgtsv_{n_big}"] = median_time(lambda: fd_dgtsv(a_big), 2000)
    T["fd_batch_numpy"] = median_time(lambda: solver.fd_solve_batch(solver.nodal_harmonic_a(a)), 50) / a.shape[0]
    thomas = jax.jit(lambda aa: solver._thomas_jax(2.0 * aa[:, :-1] * aa[:, 1:] / (aa[:, :-1] + aa[:, 1:]), 1.0))
    T["fd_batch_jax"] = median_time(lambda: jax.block_until_ready(thomas(a32)), 200) / a.shape[0]

    # -------------------------------------------------------------- DeepONet
    m = load("deeponet")
    P, sc = m["params"], m["scaler"]
    mu, sd, us = sc["g_mean"], sc["g_std"], sc["u_scale"]

    @jax.jit
    def don_full(aa):
        g = (jnp.log(aa[:, sens]) - mu) / sd
        return deeponet.apply(P, g, xs) * us

    basis = xs[:, None] * (1.0 - xs[:, None]) * nets.mlp_apply(P["trunk"], (2.0 * xs - 1.0)[:, None],
                                                                final_act=jnp.tanh)      # (129, p), once
    bias = xs * (1.0 - xs) * P["b0"]

    @jax.jit
    def don_cached(aa):
        g = (jnp.log(aa[:, sens]) - mu) / sd
        return (nets.mlp_apply(P["branch"], g) @ basis.T + bias) * us

    assert np.allclose(np.asarray(don_full(a32)), np.asarray(don_cached(a32)), rtol=1e-4, atol=1e-6)
    T["deeponet_single_full"] = median_time(lambda: jax.block_until_ready(don_full(a1)), 2000)
    T["deeponet_single_cached_trunk"] = median_time(lambda: jax.block_until_ready(don_cached(a1)), 2000)
    T["deeponet_batch_full"] = median_time(lambda: jax.block_until_ready(don_full(a32)), 200) / a.shape[0]
    T["deeponet_batch_cached_trunk"] = median_time(lambda: jax.block_until_ready(don_cached(a32)), 200) / a.shape[0]

    # ------------------------------------------------------------------- FNO
    m = load("fno")
    P, sc = m["params"], m["scaler"]
    mu, sd, us = sc["g_mean"], sc["g_std"], sc["u_scale"]

    @jax.jit
    def fno_run(aa):
        return fno.apply(P, (jnp.log(aa) - mu) / sd, xs) * us

    T["fno_single"] = median_time(lambda: jax.block_until_ready(fno_run(a1)), 1000)
    T["fno_batch"] = median_time(lambda: jax.block_until_ready(fno_run(a32)), 50) / a.shape[0]

    # --------------------------------------- arithmetic per field (XLA's own count)
    def flops(fn, arg):
        ca = fn.lower(arg).compile().cost_analysis()
        ca = ca[0] if isinstance(ca, (list, tuple)) else ca
        return float(ca.get("flops", float("nan")))

    T["flops_per_field"] = {"deeponet_cached_trunk": flops(don_cached, a1), "fno": flops(fno_run, a1),
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
    r = json.load(open(os.path.join(RES, "results.json")))
    n_train = r["config"]["n_train"]
    label = r["timing"]["fd_labels_seconds_per_instance"] * n_train
    offline = {k: float(np.mean(r["timing"][f"{k}_train_seconds"])) + label for k in ("deeponet", "fno")}
    pinn = {f: float(np.mean(r["timing"][f"pinn_{f}_train_seconds_train_family"])) for f in ("strong", "mixed")}
    B = T["best"]
    T["offline_seconds"] = offline
    T["pinn_seconds_per_field"] = pinn
    T["breakeven_queries_vs_fd"] = {
        mode: {k: (offline[k] / (B[f"fd_{mode}"] - B[f"{k}_{mode}"]) if B[f"fd_{mode}"] > B[f"{k}_{mode}"] else None)
               for k in ("deeponet", "fno")} for mode in ("single", "batch")}
    T["breakeven_fields_vs_pinn"] = {f: {k: offline[k] / pinn[f] for k in ("deeponet", "fno")} for f in pinn}
    q = 1e6
    T["million_queries_seconds_single"] = {
        "fd": q * B["fd_single"], **{k: offline[k] + q * B[f"{k}_single"] for k in ("deeponet", "fno")},
        **{f"pinn_{f}": q * s for f, s in pinn.items()}}
    with open(os.path.join(RES, "timing.json"), "w") as fh:
        json.dump(T, fh, indent=1)
    print(json.dumps(T, indent=1))


if __name__ == "__main__":
    main()
