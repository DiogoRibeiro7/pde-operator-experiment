"""Solver baselines on the same footing as the learned models.

    python baselines.py      # a couple of minutes; writes results/baselines.json

run_experiment.py measures the finite-difference error at the solver's own
nodes. The learned models are scored on the 129-point grid. This script puts
every method on that common grid (the solver's nodal solution is carried onto
it by piecewise-linear interpolation, the natural reconstruction for a
second-order scheme), and adds a second solver variant that sees only point
values of a, exactly the information the operators get:

  cell   a_{i+1/2} = harmonic mean of a over the cell (as in run_experiment.py);
         uses a between the nodes (4-point Gauss or exact integrals)
  nodal  a_{i+1/2} = harmonic mean of a(x_i) and a(x_{i+1}); point values only

It regenerates the test fields from the same seed and checks them against
results/arrays.npz before using them.
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import spearmanr

from elliptic1d import fields, solver
from elliptic1d.config import Config

RES = "results"
N_GRID = 129


def rel_l2(pred, ref):
    return np.linalg.norm(pred - ref, axis=-1) / np.linalg.norm(ref, axis=-1)


def regenerate(cfg: Config):
    """The same draws as run_experiment.py, in the same order."""
    rng = np.random.default_rng(cfg.seed)
    n_pool = max(cfg.n_train_sweep + (cfg.n_train,))
    fields.sample(cfg.train_family, rng, n_pool)            # the training pool (not needed here)
    fams = {cfg.train_family.name: cfg.train_family, **{f.name: f for f in cfg.shift_families}}
    return {name: fields.sample(fam, rng, cfg.n_test) for name, fam in fams.items()}


def fd_on_common_grid(p: dict, n: int, variant: str) -> np.ndarray:
    u = solver.fd(p, n) if variant == "cell" else solver.fd_nodal(p, n)
    return solver.to_grid(u, N_GRID)


def smallest_matching_n(ns, errs, level, n_max=N_GRID):
    """Smallest grid (up to the 129-point evaluation grid) whose mean error is at or below `level`.
    Beyond 129 points, grids that do not nest in the evaluation grid pay an interpolation
    penalty, so the curve is no longer monotone and the question is not well posed."""
    for n, e in zip(ns, errs, strict=True):
        if n <= n_max and e <= level:
            return int(n)
    return None


def main() -> None:
    cfg = Config()
    r = json.load(open(os.path.join(RES, "results.json")))
    A = np.load(os.path.join(RES, "arrays.npz"))
    tests = regenerate(cfg)
    x = solver.grid(N_GRID)
    for name, p in tests.items():
        if not np.allclose(fields.a(fields.take(p, slice(0, 5)), x), A[f"a_{name}"], rtol=0, atol=1e-12):
            raise RuntimeError(f"regenerated {name} fields do not match arrays.npz")
    exact = {name: solver.exact(p, N_GRID, cfg.n_fine) for name, p in tests.items()}
    out: dict = {"note": "all errors on the common 129-point grid; solver solutions interpolated linearly"}

    # --- 1. both solver variants at 129 points, every family, every field
    out["fd129"] = {}
    for name, p in tests.items():
        c = fields.a(p, x)
        contrast = c.max(axis=1) / c.min(axis=1)
        row = {}
        for v in ("cell", "nodal"):
            e = rel_l2(fd_on_common_grid(p, N_GRID, v), exact[name])
            row[v] = {"mean": float(e.mean()), "median": float(np.median(e)),
                      "p90": float(np.quantile(e, 0.9)), "max": float(e.max()),
                      "worst_over_mean": float(e.max() / e.mean()),
                      "spearman_with_contrast": float(spearmanr(contrast, e)[0])}
            if v == "nodal":
                for k in ("deeponet", "fno"):
                    if f"err_{k}_{name}" in A.files:
                        ratio = A[f"err_{k}_{name}"] / e
                        row[f"{k}_over_nodal_spearman_with_contrast"] = float(spearmanr(contrast, ratio)[0])
        for k in ("deeponet", "fno"):
            eo = A[f"err_{k}_{name}"]
            row[f"{k}_first_run"] = {"worst_over_mean": float(eo.max() / eo.mean()),
                                     "spearman_with_contrast": float(spearmanr(contrast, eo)[0])}
        out["fd129"][name] = row

    # --- 2. convergence on the common grid, and the grid size that matches each model
    ns = list(range(5, 601)) + [1025, 2049]
    p_tr = tests["train"]
    conv = {v: {"all200": [], "pinn_fields": []} for v in ("cell", "nodal")}
    for n in ns:
        for v in ("cell", "nodal"):
            u = fd_on_common_grid(p_tr, n, v)
            e = rel_l2(u, exact["train"])
            conv[v]["all200"].append(float(e.mean()))
            conv[v]["pinn_fields"].append(float(e[: cfg.n_pinn_instances].mean()))
    out["convergence"] = {"n": ns, **conv}
    dy = [n for n in ns if n >= 33 and ((n - 1) & (n - 2)) == 0]      # 33, 65, 129, ... (n - 1 a power of 2)
    out["convergence_slope_vs_h"] = {
        v: float(np.polyfit(np.log([1.0 / (n - 1) for n in dy]),
                            np.log([conv[v]["all200"][ns.index(n)] for n in dy]), 1)[0])
        for v in ("cell", "nodal")}

    op_mean = {k: float(np.mean([row["mean"] for row in r["operators"]["shift"][k]["train"]]))
               for k in ("deeponet", "fno")}
    pinn_mean = {form: float(np.mean([row["rel_l2"] for row in r["pinn"]["train"][form]]))
                 for form in ("strong", "mixed")}
    out["model_mean_error_train"] = {**op_mean, **{f"pinn_{f}": v for f, v in pinn_mean.items()}}
    out["matching_n"] = {}
    for v in ("cell", "nodal"):
        out["matching_n"][v] = {
            **{k: smallest_matching_n(ns, conv[v]["all200"], m) for k, m in op_mean.items()},
            **{f"pinn_{f}": smallest_matching_n(ns, conv[v]["pinn_fields"], m) for f, m in pinn_mean.items()},
        }
    out["fd_on_pinn_fields_129"] = {v: conv[v]["pinn_fields"][ns.index(N_GRID)] for v in ("cell", "nodal")}

    # --- 2b. order of convergence of each variant at its own nodes, every family (50 fields each)
    out["order_at_own_nodes"] = {}
    grids = [33, 65, 129, 257, 513, 1025, 2049]
    for name, p in tests.items():
        sub = fields.take(p, slice(0, 50))
        row = {}
        for v in ("cell", "nodal"):
            errs = []
            for n in grids:
                u = solver.fd(sub, n) if v == "cell" else solver.fd_nodal(sub, n)
                errs.append(float(rel_l2(u, solver.exact(sub, n, cfg.n_fine)).mean()))
            row[v] = {"grids": grids, "mean_error": errs,
                      "slope_vs_h": float(np.polyfit(np.log([1.0 / (n - 1) for n in grids]), np.log(errs), 1)[0])}
        out["order_at_own_nodes"][name] = row

    # --- 3. where the rough family's extra variation sits, relative to the FNO's retained modes
    # FNO mode m on the padded period 9/8 has angular frequency 2*pi*m*8/9; the fields use pi*k.
    k_max = 2 * (cfg.fno_modes - 1) * 8 / 9
    spec = {}
    for fam in (cfg.train_family, cfg.shift_families[0]):
        s2 = fields.spectral_weights(fam) ** 2
        k = np.arange(1, s2.size + 1)
        spec[fam.name] = {"var_fraction_k_le_5": float(s2[k <= 5].sum() / s2.sum()),
                          "var_fraction_k_6_to_kmax": float(s2[(k >= 6) & (k <= k_max)].sum() / s2.sum()),
                          "var_fraction_above_kmax": float(s2[k > k_max].sum() / s2.sum())}
    out["fno_mode_coverage"] = {"k_max_retained": k_max, **spec}

    with open(os.path.join(RES, "baselines.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "convergence"}, indent=1))


if __name__ == "__main__":
    main()
