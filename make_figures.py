"""Draw the article's figures from the files in results/.

    python make_figures.py                          # reads results/, writes figures/
    python make_figures.py --results results-quick  # writes results-quick/figures/

Reads results.json and arrays.npz (run_experiment.py), baselines.json
(baselines.py) and timing.json (timing.py). Also writes derived.json into the
results folder: the few numbers computed from the results rather than read off
them (convergence slopes, training-set slopes, error-versus-contrast
statistics), so the article can cite them too.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RES, FIG = str(ROOT / "results"), str(ROOT / "figures")      # set in main()

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
C = {"fd": "#3d3c39", "pinn": "#eb6834", "deeponet": "#1baf7a", "fno": "#2a78d6"}
LABEL = {"fd": "Finite differences", "pinn": "PINN", "deeponet": "DeepONet", "fno": "FNO"}
FAMILIES = ["train", "rough", "high-contrast", "piecewise"]
FAMILY_TITLE = {"train": "Training family", "rough": "Rough (ℓ = 0.05)",
                "high-contrast": "High contrast (σ = 1.6)", "piecewise": "Piecewise constant"}
CREDIT = "Figure: Diogo Ribeiro · own experiment, reproducible from the accompanying code"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelcolor": INK2, "axes.edgecolor": MUTED,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 200, "lines.linewidth": 1.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def nice_log_axis(ax, which: str, lo: float, hi: float) -> None:
    """Log axis with ticks at 1-2-5 steps, printed as plain numbers."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    ticks = [m * 10.0**k for k in range(int(np.floor(np.log10(lo))) - 1, int(np.ceil(np.log10(hi))) + 1)
             for m in (1, 2, 5) if lo <= m * 10.0**k <= hi]
    axis = ax.yaxis if which == "y" else ax.xaxis
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def spread(values: list[float], min_ratio: float = 1.45) -> list[float]:
    """Nudge label positions apart in log space so stacked labels do not collide."""
    order = np.argsort(values)
    pos = np.log(np.asarray(values, dtype=float))
    out = pos.copy()
    step = np.log(min_ratio)
    for a, b in zip(order[:-1], order[1:], strict=True):
        if out[b] - out[a] < step:
            out[b] = out[a] + step
    return list(np.exp(out))


def header(fig, title: str, subtitle: str) -> None:
    fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.012, 0.935, subtitle, ha="left", va="top", fontsize=10.5, color=INK2)
    fig.text(0.988, 0.012, CREDIT, ha="right", va="bottom", fontsize=8, color=MUTED)


def seed_mean(rows: list[dict], key: str) -> float:
    return float(np.mean([r[key] for r in rows]))


def main() -> None:
    global RES, FIG
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default=str(ROOT / "results"), help="folder with the result files")
    ap.add_argument("--figures", default=None,
                    help="output folder (default: figures/ for results/, otherwise <results>/figures)")
    args = ap.parse_args()
    RES = os.path.abspath(args.results)
    FIG = args.figures or (str(ROOT / "figures") if Path(RES) == ROOT / "results" else os.path.join(RES, "figures"))
    os.makedirs(FIG, exist_ok=True)
    r = json.load(open(os.path.join(RES, "results.json")))
    A = np.load(os.path.join(RES, "arrays.npz"))
    derived: dict = {}
    x = A["x"]
    op = r["operators"]

    # ----------------------------------------------------------------- fig 1
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.35]})
    for j, fam in enumerate(FAMILIES):
        ax = axes[0, j]
        ax.plot(x, A[f"a_{fam}"][0], color=INK2, lw=1.6)
        ax.set_yscale("log")
        av = A[f"a_{fam}"][0]
        nice_log_axis(ax, "y", av.min() * 0.9, av.max() * 1.1)
        ax.set_title(FAMILY_TITLE[fam])
        if j == 0:
            ax.set_ylabel("coefficient a(x)")
        ax = axes[1, j]
        ue = A[f"exact_{fam}"][0]
        curves = [("fd", A[f"fd_{fam}"][0], "-", None)]
        if f"pinn_strong_{fam}" in A.files:
            curves.append(("pinn", A[f"pinn_strong_{fam}"], "--", "strong form"))
        curves.append(("pinn", A[f"pinn_mixed_{fam}"], "-", "flux form"))
        curves += [("deeponet", A[f"pred_deeponet_{fam}"][0], "-", None),
                   ("fno", A[f"pred_fno_{fam}"][0], "-", None)]
        for key, u, ls, _form in curves:
            err = np.maximum(np.abs(u - ue), 1e-10)
            ax.plot(x, err, color=C[key], ls=ls, lw=1.5 if ls == "-" else 1.4)
        ax.set_yscale("log")
        ax.set_ylim(1e-8, 1)
        ax.set_xlabel("x")
        if j == 0:
            ax.set_ylabel("|u − u_exact|")
    handles = [plt.Line2D([], [], color=C["fd"], label="Finite differences (129 pts)"),
               plt.Line2D([], [], color=C["pinn"], ls="--", label="PINN, strong form"),
               plt.Line2D([], [], color=C["pinn"], label="PINN, flux form"),
               plt.Line2D([], [], color=C["deeponet"], label="DeepONet"),
               plt.Line2D([], [], color=C["fno"], label="FNO")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.9), ncol=5, fontsize=10)
    header(fig, "One instance from each coefficient family, and the pointwise error of every method",
           "Top: the coefficient (log scale). Bottom: absolute error against the exact solution. "
           "Operators were trained only on the first family.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.84))
    fig.savefig(os.path.join(FIG, "fig1_instances_and_errors.png"))
    plt.close(fig)

    # ----------------------------------------------------------------- fig 2
    # second-order convergence at the solver's own nodes (run_experiment.py)
    derived["fd_convergence_slope_by_family"] = {}
    for fam, cf in r["solver"]["convergence"].items():
        nf = np.array(sorted(int(k) for k in cf))
        ef = np.array([cf[str(n)]["mean"] for n in nf])
        keep = nf >= 33
        derived["fd_convergence_slope_by_family"][fam] = float(
            np.polyfit(np.log(1.0 / (nf[keep] - 1)), np.log(ef[keep]), 1)[0])
    # matching, on the common 129-point grid (baselines.py)
    bl = json.load(open(os.path.join(RES, "baselines.json")))
    ns = np.array(bl["convergence"]["n"])
    dense = ns <= 65                    # coarser grids, read on the 129-point grid by interpolation
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for v, ls in (("nodal", "-"), ("cell", "--")):
        e = np.array(bl["convergence"][v]["all200"])
        ax.plot(ns[dense], e[dense], ls, color=C["fd"], lw=2 if v == "nodal" else 1.5, zorder=3)
        dy = np.isin(ns, [5, 9, 17, 33, 65])
        ax.plot(ns[dy], e[dy], "o", color=C["fd"], ms=5, mfc="white", mew=1.4, zorder=3)
        i129 = int(np.where(ns == 129)[0][0])
        ax.plot([129], [e[i129]], "s", color=C["fd"], ms=6, mfc="white" if v == "cell" else C["fd"], mew=1.4, zorder=3)
    ax.text(122, 5.3e-5, "129 points:\nno interpolation", fontsize=9, color=INK2, ha="right", va="center")
    ax.text(5.3, 4e-5, "finite differences, error against the exact solution\n"
            "solid: given a at its grid points only (as the FNO is)\n"
            "dashed: given cell averages of 1/a", color=C["fd"], fontsize=9.5, va="bottom")
    levels = bl["model_mean_error_train"]
    specs = (("fno", "FNO", C["fno"]), ("deeponet", "DeepONet", C["deeponet"]))
    ys = spread([levels[k] for k, *_ in specs], min_ratio=6.0)
    for (key, lab, col), ylab in zip(specs, ys, strict=True):
        v = levels[key]
        nn, nc = bl["matching_n"]["nodal"][key], bl["matching_n"]["cell"][key]
        ax.axhline(v, color=col, lw=1.6, zorder=2)
        ax.plot([nn], [v], "o", color=col, ms=8, zorder=4, mec="white", mew=1.2)
        ax.plot([nc], [v], "o", color=col, ms=6, zorder=4, mfc="white", mew=1.4)
        ax.annotate(f"{lab}  {v:.1e}\nmatched with {nn} points\n({nc} with cell averages)",
                    xy=(1.0, v), xycoords=("axes fraction", "data"), xytext=(1.02, ylab),
                    textcoords=("axes fraction", "data"), va="center", fontsize=9.5, color=INK2,
                    annotation_clip=False, arrowprops=dict(arrowstyle="-", color=col, lw=1.0, shrinkA=0, shrinkB=2))
    ticks = [5, 9, 17, 33, 65, 129]
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(n) for n in ticks])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("grid points used by the finite-difference solver")
    ax.set_ylabel("mean relative L² error, training family")
    ax.set_xlim(4.4, 150)
    ax.set_ylim(1.5e-5, 0.4)
    header(fig, "How many grid points does the solver need to match each operator?",
           "All errors on the same 129-point grid and the same 200 test fields; coarser solver solutions are "
           "interpolated linearly onto it.")
    fig.tight_layout(rect=(0, 0.03, 0.76, 0.88))
    fig.savefig(os.path.join(FIG, "fig2_accuracy_ladder.png"))
    plt.close(fig)

    # ----------------------------------------------------------------- fig 3
    fig, ax = plt.subplots(figsize=(10, 5.4))
    derived["sweep"] = {}
    lo_y, hi_y = np.inf, -np.inf
    for key in ("deeponet", "fno"):
        sizes = np.array(sorted(int(k) for k in op["sweep"][key]))
        means = []
        for n in sizes:
            rows = op["sweep"][key][str(n)]
            vals = [row["mean"] for row in rows]
            ax.plot([n] * len(vals), vals, "o", color=C[key], ms=5, mfc="white", mew=1.2, alpha=0.9)
            means.append(np.mean(vals))
        means = np.array(means)
        ax.plot(sizes, means, "-", color=C[key], lw=2)
        b = np.polyfit(np.log(sizes), np.log(means), 1)[0]
        derived["sweep"][key] = {"sizes": sizes.tolist(), "seed_mean_of_mean_error": means.tolist(),
                                 "loglog_slope": float(b)}
        ax.text(sizes[-1] * 1.12, means[-1], f"{LABEL[key]}\n(log-log slope {b:.2f})", va="center",
                color=INK2, fontsize=10)
        ys_all = [row["mean"] for n in sizes for row in op["sweep"][key][str(n)]]
        lo_y, hi_y = min(lo_y, min(ys_all)), max(hi_y, max(ys_all))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    nice_log_axis(ax, "y", lo_y * 0.8, hi_y * 1.25)
    ax.set_xlabel("number of training pairs (a, u)")
    ax.set_ylabel("mean relative L² error (training family)")
    ax.set_xlim(right=sizes[-1] * 2.2)
    header(fig, "Error against training-set size",
           "Hollow points are independent runs (different initialisations and, below 2,000 pairs, "
           "different subsets); the line joins their means.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.88))
    fig.savefig(os.path.join(FIG, "fig3_training_size.png"))
    plt.close(fig)

    # ----------------------------------------------------------------- fig 4
    fig, ax = plt.subplots(figsize=(11, 6.2))
    methods = [("fd", "Finite differences, cell averages"), ("fd_nodal", "Finite differences, point values"),
               ("pinn_strong", "PINN, strong form"), ("pinn_mixed", "PINN, flux form"),
               ("deeponet", "DeepONet"), ("fno", "FNO")]
    offs = np.linspace(-0.33, 0.33, len(methods))
    derived["shift"] = {}
    for j, fam in enumerate(FAMILIES):
        derived["shift"][fam] = {}
        for (key, _lab), dx in zip(methods, offs, strict=True):
            col = C["pinn"] if key.startswith("pinn") else C["fd"] if key.startswith("fd") else C[key]
            if key.startswith("fd"):
                s = bl["fd129"][fam]["cell" if key == "fd" else "nodal"]
                mid, lo, hi = s["median"], None, s["p90"]
            elif key.startswith("pinn"):
                form = key.split("_")[1]
                if form not in r["pinn"][fam]:
                    ax.text(j + dx, 3e-6, "n/a", ha="center", fontsize=8.5, color=MUTED, rotation=90)
                    continue
                e = np.array([row["rel_l2"] for row in r["pinn"][fam][form]])
                mid, lo, hi = float(np.median(e)), float(e.min()), float(e.max())
            else:
                rows = op["shift"][key][fam]
                mid, lo, hi = seed_mean(rows, "median"), None, seed_mean(rows, "p90")
            derived["shift"][fam][key] = {"median": mid, "low": lo, "high": hi}
            ax.plot([j + dx, j + dx], [lo if lo else mid, hi], color=col, lw=1.6, alpha=0.8)
            hollow = key in ("pinn_strong", "fd_nodal")
            ax.plot(j + dx, mid, "o", ms=8, color=col, mfc="white" if hollow else col, mew=1.8,
                    mec=col if hollow else "white", zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(range(len(FAMILIES)))
    ax.set_xticklabels([FAMILY_TITLE[f] for f in FAMILIES])
    ax.set_ylabel("relative L² error")
    ax.set_ylim(1e-6, 5)
    for j in range(1, len(FAMILIES)):
        ax.axvline(j - 0.5, color=GRID, lw=1)
    ax.grid(axis="x", visible=False)
    handles = [plt.Line2D([], [], color=C["fd"], marker="o", ls="", ms=8, label="Finite differences, cell averages"),
               plt.Line2D([], [], color=C["fd"], marker="o", ls="", ms=8, mfc="white", mew=1.8,
                          label="Finite differences, point values"),
               plt.Line2D([], [], color=C["pinn"], marker="o", ls="", ms=8, mfc="white", mew=1.8,
                          label="PINN, strong form"),
               plt.Line2D([], [], color=C["pinn"], marker="o", ls="", ms=8, label="PINN, flux form"),
               plt.Line2D([], [], color=C["deeponet"], marker="o", ls="", ms=8, label="DeepONet"),
               plt.Line2D([], [], color=C["fno"], marker="o", ls="", ms=8, label="FNO")]
    ax.legend(handles=handles, loc="lower left", ncol=3, fontsize=9.5, bbox_to_anchor=(0, 1.0))
    header(fig, "What happens when the coefficient leaves the training distribution",
           "Dots: median. Whiskers: up to the 90th percentile over 200 test fields (solver; operators, "
           "averaged over 3 runs), or min–max over 5 fields (PINNs).")
    fig.tight_layout(rect=(0, 0.03, 1, 0.86))
    fig.savefig(os.path.join(FIG, "fig4_distribution_shift.png"))
    plt.close(fig)

    # ----------------------------------------------------------------- fig 5
    T = json.load(open(os.path.join(RES, "timing.json")))
    B, offline, pinn_s = T["best"], T["offline_seconds"], T["pinn_seconds_per_field"]
    Q = np.logspace(0, 6, 200)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    lines = {"fd": Q * B["fd_single"],
             "pinn": Q * pinn_s["mixed"],
             "deeponet": offline["deeponet"] + Q * B["deeponet_single"],
             "fno": offline["fno"] + Q * B["fno_single"]}
    ax.fill_between(Q, Q * pinn_s["mixed"], Q * pinn_s["strong"], color=C["pinn"], alpha=0.25, lw=0)
    ax.plot(Q, Q * pinn_s["strong"], color=C["pinn"], lw=1.2)
    for key, y in lines.items():
        ax.plot(Q, y, color=C[key], lw=2)
    names = {"fd": "Finite differences", "pinn": "PINN, one per field\n(flux to strong form)",
             "deeponet": "DeepONet", "fno": "FNO"}
    ends = spread([lines[k][-1] for k in lines], min_ratio=2.2)
    for (key, y), ylab in zip(lines.items(), ends, strict=True):
        ax.annotate(names[key], xy=(Q[-1], y[-1]), xytext=(Q[-1] * 1.4, ylab), va="center",
                    fontsize=10, color=INK2, arrowprops=dict(arrowstyle="-", color=C[key], lw=1.0))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of coefficient fields to solve for")
    ax.set_ylabel("total wall-clock time (s)")
    ax.set_xlim(1, Q[-1] * 30)
    header(fig, "Total cost of answering many queries, including training",
           "One field per call, fastest measured implementation of each method; "
           "the operators pay for data and training up front.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.88))
    fig.savefig(os.path.join(FIG, "fig5_cost.png"))
    plt.close(fig)

    # where the operators' errors concentrate (first run of each operator)
    from scipy.stats import spearmanr
    derived["error_vs_contrast"] = {}
    for k in ("deeponet", "fno"):
        e, c = A[f"err_{k}_train"], A["contrast_train"]
        worst = np.argsort(e)[-10:]
        derived["error_vs_contrast"][k] = {
            "spearman": float(spearmanr(c, e)[0]), "max_error": float(e.max()),
            "median_contrast_worst10": float(np.median(c[worst])), "median_contrast_all": float(np.median(c))}

    with open(os.path.join(RES, "derived.json"), "w") as fh:
        json.dump(derived, fh, indent=1)
    print(json.dumps(derived, indent=1))


if __name__ == "__main__":
    main()
