"""MkDocs hooks: bring the figures into the site and fill in result tables and numbers.

Every table marked <!-- results:NAME --> and every number marked
<!-- value:NAME --> in a page is generated here from results/*.json (and the
citation from pyproject.toml and CITATION.cff) at build time, so the site
agrees with the committed results and metadata without hand-copied numbers.
"""

from __future__ import annotations

import json
import re
import shutil
import tomllib
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
    c = r["config"]
    n, n_test, n_pinn, n_runs = c["n_grid"], c["n_test"], c["n_pinn_instances"], c["n_seeds"]
    rows = [[f"Finite differences, cell averages ({n} pts, {n_test} fields)"]
            + [sci(b["fd129"][f]["cell"]["mean"]) for f in FAMILIES],
            [f"Finite differences, point values ({n} pts, {n_test} fields)"]
            + [sci(b["fd129"][f]["nodal"]["mean"]) for f in FAMILIES],
            [f"Finite differences, cell averages ({n} pts), on the {n_pinn} PINN fields"]
            + [sci(mean(r["pinn_paired_instances"][f]["fd"])) for f in FAMILIES]]
    for form, label in (("strong", "PINN, strong form"), ("mixed", "PINN, flux form")):
        rows.append([f"{label} ({n_pinn} fields)"]
                    + [sci(mean(x["rel_l2"] for x in r["pinn"][f][form])) if form in r["pinn"][f] else "n/a"
                       for f in FAMILIES])
    for k, label in (("deeponet", "DeepONet"), ("fno", "FNO")):
        rows.append([f"{label} ({n_runs} runs, {n_test} fields)"]
                    + [sci(mean(x["mean"] for x in r["operators"]["shift"][k][f])) for f in FAMILIES])
    return table(["Method"] + [FAMILY_LABEL[f] for f in FAMILIES], rows)


def count(v) -> str:
    return "n/a" if v is None else str(v)


def matching_table() -> str:
    b = load("baselines.json")
    m = b["model_mean_error_train"]
    rows = []
    for k, label in (("fno", "FNO"), ("deeponet", "DeepONet")):
        rows.append([label, sci(m[k]), count(b["matching_n"]["nodal"][k]), count(b["matching_n"]["cell"][k]),
                     count(b["matching_n_nested"]["nodal"][k]), count(b["matching_n_at_own_nodes_cell"][k])])
    return table(["Operator", "Mean error", "Point-value solver", "Cell-average solver",
                  "Point-value solver, sub-grids of the FNO's grid only", "Cell-average, scored at its own nodes"],
                 rows)


def timing_table() -> str:
    t = load("timing.json")
    best, tr = t["best"], t["training"]
    us = lambda s: f"{s * 1e6:,.1f} µs"  # noqa: E731

    def spread(mean_s, rng, unit=""):
        return f"{mean_s:.0f} s{unit} ({rng[0]:.0f}–{rng[1]:.0f})"

    runs, fields = tr.get("n_operator_runs"), tr.get("n_pinn_fields")
    rows = [["Finite differences (point values)", us(best["fd_single"]), us(best["fd_batch"]), "none"],
            ["DeepONet", us(best["deeponet_single"]), us(best["deeponet_batch"]),
             spread(t["offline_seconds"]["deeponet"], t["offline_seconds_range"]["deeponet"])],
            ["FNO", us(best["fno_single"]), us(best["fno_batch"]),
             spread(t["offline_seconds"]["fno"], t["offline_seconds_range"]["fno"])],
            ["PINN, flux form", "—", "—",
             spread(t["pinn_seconds_per_field"]["mixed"], t["pinn_seconds_per_field_range"]["mixed"], " per field")],
            ["PINN, strong form", "—", "—",
             spread(t["pinn_seconds_per_field"]["strong"], t["pinn_seconds_per_field_range"]["strong"], " per field")]]
    main_cpu = load("results.json")["environment"]["cpu"]
    note = (f"\n\nTraining: mean (range) over {runs} operator runs and {fields} PINN fields. All times on this "
            f"page were measured on {t['environment']['cpu']} ({t['environment']['cpu_count']} cores); the "
            f"accuracy results come from the main run on {main_cpu}.")
    return table(["Method", "One field per call", f"Per field, {t['n_fields_batch']} per call", "Training"],
                 rows, "lrrr") + note


def sweep_table() -> str:
    d = load("derived.json")["sweep"]
    sizes = d["fno"]["sizes"]
    rows = [[label] + [sci(v) for v in d[k]["seed_mean_of_mean_error"]] + [f"{d[k]['loglog_slope']:.2f}"]
            for k, label in (("deeponet", "DeepONet"), ("fno", "FNO"))]
    return table(["Operator"] + [f"{n:,} pairs" for n in sizes] + ["Log-log slope"], rows)


def bibtex() -> str:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    cff = (ROOT / "CITATION.cff").read_text()

    def field(name: str) -> str:
        m = re.search(rf'^{name}: "?(.*?)"?$', cff, re.M)
        return m.group(1) if m else ""

    return "\n".join([
        "```bibtex",
        "@software{ribeiro_pde_operator_experiment,",
        "  author  = {Ribeiro, Diogo},",
        f"  title   = {{{field('title')}}},",
        f"  year    = {{{field('date-released')[:4]}}},",
        f"  version = {{{project['version']}}},",
        f"  license = {{{field('license')}}},",
        f"  url     = {{{field('repository-code')}}}",
        "}",
        "```"])


GENERATORS = {"accuracy": accuracy_table, "matching": matching_table, "timing": timing_table,
              "sweep": sweep_table, "bibtex": bibtex}


def values() -> dict[str, str]:
    """Single numbers quoted in the prose."""
    r, b, t = load("results.json"), load("baselines.json"), load("timing.json")
    shift = r["operators"]["shift"]
    op_mean = {k: {f: mean(x["mean"] for x in shift[k][f]) for f in FAMILIES} for k in ("deeponet", "fno")}
    worst = max(op_mean[k][f] / op_mean[k]["train"] for k in op_mean for f in FAMILIES[1:])
    rng = t["breakeven_fields_vs_pinn_range"]
    lo, hi = min(v[0] for v in rng.values()), max(v[1] for v in rng.values())
    out = {"max_shift_factor": f"{worst:.0f}",
           "pinn_fields_range": f"between {lo:.1f} and {hi:.1f}"}
    nf, nd = (count(b["matching_n_nested"]["nodal"][k]) for k in ("fno", "deeponet"))
    out["match_nested_text"] = (f"{nf} points match both" if nf == nd
                                else f"{nf} and {nd} points match them respectively")
    for v in ("nodal", "cell"):
        for k in ("fno", "deeponet"):
            out[f"match_{v}_{k}"] = count(b["matching_n"][v][k])
    for k in ("fno", "deeponet"):
        out[f"match_own_{k}"] = count(b["matching_n_at_own_nodes_cell"][k])
    return out


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
    if "<!-- value:" in markdown:
        vals = values()

        def fill(m):
            if m.group(1) not in vals:
                raise KeyError(f"unknown value marker {m.group(1)!r} in {page.file.src_path}")
            return vals[m.group(1)]

        markdown = re.sub(r"<!-- value:([\w-]+) -->", fill, markdown)
    return markdown
