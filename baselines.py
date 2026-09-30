"""Solver baselines on the same footing as the learned models.

    python baselines.py                        # under a minute; writes results/baselines.json
    python baselines.py --results results-quick

run_experiment.py measures the finite-difference error at the solver's own
nodes. The learned models are scored on the 129-point grid. This script puts
every method on that common grid (the solver's nodal solution is carried onto
it by piecewise-linear interpolation), and adds a second solver variant that
is given only point values of a at its own grid points, the kind of input the
operators get:

  cell   a_{i+1/2} = harmonic mean of a over the cell (as in run_experiment.py);
         uses a between the nodes (4-point Gauss or exact integrals)
  nodal  a_{i+1/2} = harmonic mean of a(x_i) and a(x_{i+1}); point values only

A point-value solver on, say, 27 points reads a at 27 points, most of which are
not among the FNO's 129. `matching_n_nested` therefore also asks the stricter
question: which grids that are subsets of the FNO's grid (5, 9, 17, 33, 65)
match each operator.

It regenerates the test fields from the run's configuration and checks them
against the saved arrays before using them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from elliptic1d import fields, runs, solver

ROOT = Path(__file__).resolve().parent


def rel_l2(pred, ref):
    return np.linalg.norm(pred - ref, axis=-1) / np.linalg.norm(ref, axis=-1)


def fd_variant(p: dict, n: int, variant: str, f: float) -> np.ndarray:
    return solver.fd(p, n, f) if variant == "cell" else solver.fd_nodal(p, n, f)


def smallest_matching_n(ns, errs, level, n_max):
    """Smallest grid (up to the evaluation grid) whose mean error is at or below `level`.
    Beyond the evaluation grid, grids that do not nest in it pay an interpolation
    penalty, so the curve is no longer monotone and the question is not well posed."""
    for n, e in zip(ns, errs, strict=True):
        if n <= n_max and e <= level:
            return int(n)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default=str(ROOT / "results"), help="folder written by run_experiment.py")
    args = ap.parse_args()
    res_dir = Path(args.results)
    r, cfg = runs.load(res_dir)
    N = cfg.n_grid
    f = cfg.f_const
    A = np.load(res_dir / "arrays.npz")
    _, tests = runs.draws(cfg)
    runs.check_tests(tests, res_dir, N)
    x = solver.grid(N)
    exact = {name: solver.exact(p, N, cfg.n_fine, f=f) for name, p in tests.items()}
    out: dict = {"note": f"all errors on the common {N}-point grid; solver solutions interpolated linearly"}

    # --- 1. both solver variants on the evaluation grid, every family, every field
    out["fd129"] = {}
    for name, p in tests.items():
        c = fields.a(p, x)
        contrast = c.max(axis=1) / c.min(axis=1)
        row = {}
        for v in ("cell", "nodal"):
            e = rel_l2(solver.to_grid(fd_variant(p, N, v, f), N), exact[name])
            row[v] = {"mean": float(e.mean()), "median": float(np.median(e)),
                      "p90": float(np.quantile(e, 0.9)), "max": float(e.max()),
                      "worst_over_mean": float(e.max() / e.mean()),
                      "spearman_with_contrast": float(spearmanr(contrast, e)[0])}
            if v == "nodal":
                for k in ("deeponet", "fno"):
                    ratio = A[f"err_{k}_{name}"] / e
                    row[f"{k}_first_run_over_nodal_spearman_with_contrast"] = float(spearmanr(contrast, ratio)[0])
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
            e = rel_l2(solver.to_grid(fd_variant(p_tr, n, v, f), N), exact["train"])
            conv[v]["all200"].append(float(e.mean()))
            conv[v]["pinn_fields"].append(float(e[: cfg.n_pinn_instances].mean()))
    out["convergence"] = {"n": ns, **conv}
    # Only grids that nest in the evaluation grid carry no interpolation error, so the
    # order of convergence on the common grid is fitted on those (n - 1 = 128, 256, ...).
    nested = [n for n in ns if n >= N and (n - 1) % (N - 1) == 0]
    out["convergence_slope_vs_h_nested"] = {
        "grids": nested,
        **{v: float(np.polyfit(np.log([1.0 / (n - 1) for n in nested]),
                               np.log([conv[v]["all200"][ns.index(n)] for n in nested]), 1)[0])
           for v in ("cell", "nodal")}}

    op_mean = {k: float(np.mean([row["mean"] for row in r["operators"]["shift"][k]["train"]]))
               for k in ("deeponet", "fno")}
    pinn_mean = {form: float(np.mean([row["rel_l2"] for row in r["pinn"]["train"][form]]))
                 for form in ("strong", "mixed")}
    out["model_mean_error_train"] = {**op_mean, **{f"pinn_{fm}": v for fm, v in pinn_mean.items()}}
    out["matching_n"] = {}
    for v in ("cell", "nodal"):
        out["matching_n"][v] = {
            **{k: smallest_matching_n(ns, conv[v]["all200"], m, N) for k, m in op_mean.items()},
            **{f"pinn_{fm}": smallest_matching_n(ns, conv[v]["pinn_fields"], m, N) for fm, m in pinn_mean.items()},
        }
    out["fd_on_pinn_fields_129"] = {v: conv[v]["pinn_fields"][ns.index(N)] for v in ("cell", "nodal")}
    # the stricter question: only grids whose nodes are a subset of the N-point grid
    subgrids = [n for n in ns if n < N and (N - 1) % (n - 1) == 0]
    out["matching_n_nested"] = {"grids": subgrids, **{
        v: {k: smallest_matching_n(subgrids, [conv[v]["all200"][ns.index(n)] for n in subgrids], m, N)
            for k, m in op_mean.items()} for v in ("cell", "nodal")}}

    # --- 2a. the same question under the easier convention: cell-average solver scored at its
    # own nodes (no interpolation), smallest whole number of points, searched up to N
    own = {}
    for n in range(5, N + 1):
        e = rel_l2(solver.fd(p_tr, n, f), solver.exact(p_tr, n, cfg.n_fine, f=f)).mean()
        for k, m in op_mean.items():
            if k not in own and e <= m:
                own[k] = n
        if len(own) == len(op_mean):
            break
    out["matching_n_at_own_nodes_cell"] = {k: own.get(k) for k in op_mean}

    # --- 2b. order of convergence of each variant at its own nodes, every family (50 fields each)
    out["order_at_own_nodes"] = {}
    grids = [33, 65, 129, 257, 513, 1025, 2049]
    for name, p in tests.items():
        sub = fields.take(p, slice(0, 50))
        row = {}
        for v in ("cell", "nodal"):
            errs = [float(rel_l2(fd_variant(sub, n, v, f), solver.exact(sub, n, cfg.n_fine, f=f)).mean())
                    for n in grids]
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

    with open(res_dir / "baselines.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "convergence"}, indent=1))


if __name__ == "__main__":
    main()
