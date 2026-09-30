"""MkDocs hooks: bring the figures into the site and build the result tables from the result files.

The site never carries hand-copied numbers. Every table marked
<!-- results:NAME --> in a page is generated here from results/*.json at build
time, so the site always agrees with the committed results.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FAMILIES = ["train", "rough", "high-contrast", "piecewise"]
FAMILY_LABEL = {"train": "Training family", "rough": "Rough", "high-contrast": "High contrast",
                "piecewise": "Piecewise constant"}


def sci(v: float | None) -> str:
    if v is None:
        return "n/a"
    mant, exp = f"{v:.1e}".split("e")
    return f"{mant} × 10<sup>{int(exp)}</sup>"


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def load(name: str) -> dict:
    return json.loads((RES / name).read_text())


def table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or ("l" + "r" * (len(header) - 1))
    sep = ["---:" if a == "r" else ":---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def accuracy_table() -> str:
    r, b = load("results.json"), load("baselines.json")
    rows = [["Finite differences, cell averages (129 pts)"] + [sci(b["fd129"][f]["cell"]["mean"]) for f in FAMILIES],
            ["Finite differences, point values (129 pts)"] + [sci(b["fd129"][f]["nodal"]["mean"]) for f in FAMILIES]]
    for form, label in (("strong", "PINN, strong form (5 fields)"), ("mixed", "PINN, flux form (5 fields)")):
        rows.append([label] + [sci(mean(x["rel_l2"] for x in r["pinn"][f][form])) if form in r["pinn"][f] else "n/a"
                               for f in FAMILIES])
    for k, label in (("deeponet", "DeepONet (3 runs)"), ("fno", "FNO (3 runs)")):
        rows.append([label] + [sci(mean(x["mean"] for x in r["operators"]["shift"][k][f])) for f in FAMILIES])
    return table(["Method"] + [FAMILY_LABEL[f] for f in FAMILIES], rows)


def matching_table() -> str:
    b, d = load("baselines.json"), load("derived.json")
    m = b["model_mean_error_train"]
    rows = []
    for k, label in (("fno", "FNO"), ("deeponet", "DeepONet")):
        rows.append([label, sci(m[k]), str(b["matching_n"]["nodal"][k]), str(b["matching_n"]["cell"][k]),
                     f"{d['matching_n_at_solver_nodes_cell'][k]:.0f}"])
    return table(["Operator", "Mean error", "Point-value solver", "Cell-average solver",
                  "Cell-average, scored at its own nodes"], rows)


def timing_table() -> str:
    t = load("timing.json")
    best = t["best"]
    us = lambda s: f"{s * 1e6:,.1f} µs"  # noqa: E731
    rows = [["Finite differences (point values)", us(best["fd_single"]), us(best["fd_batch"]), "none"],
            ["DeepONet", us(best["deeponet_single"]), us(best["deeponet_batch"]),
             f"{t['offline_seconds']['deeponet']:.0f} s"],
            ["FNO", us(best["fno_single"]), us(best["fno_batch"]), f"{t['offline_seconds']['fno']:.0f} s"],
            ["PINN, flux form", "—", "—", f"{t['pinn_seconds_per_field']['mixed']:.0f} s per field"],
            ["PINN, strong form", "—", "—", f"{t['pinn_seconds_per_field']['strong']:.0f} s per field"]]
    return table(["Method", "One field per call", "Per field, 200 per call", "Training"], rows, "lrrr")


def sweep_table() -> str:
    d = load("derived.json")["sweep"]
    sizes = d["fno"]["sizes"]
    rows = [[label] + [sci(v) for v in d[k]["seed_mean_of_mean_error"]] + [f"{d[k]['loglog_slope']:.2f}"]
            for k, label in (("deeponet", "DeepONet"), ("fno", "FNO"))]
    return table(["Operator"] + [f"{n:,} pairs" for n in sizes] + ["Log-log slope"], rows)


GENERATORS = {"accuracy": accuracy_table, "matching": matching_table, "timing": timing_table, "sweep": sweep_table}


def on_pre_build(config, **kwargs):
    out = Path(config["docs_dir"]) / "figures"
    out.mkdir(exist_ok=True)
    for png in sorted((ROOT / "figures").glob("*.png")):
        shutil.copy2(png, out / png.name)


def on_page_markdown(markdown, page, config, files, **kwargs):
    for name, fn in GENERATORS.items():
        marker = f"<!-- results:{name} -->"
        if marker in markdown:
            markdown = markdown.replace(marker, fn())
    return markdown
