"""Render presentation copies from published paired contrasts; never rerun statistics.

Matches algorithm/src/gnn_mechanisms/reporting.py:figures selection, ordering,
error bars, sizes, colors, limits and legends. Only display labels change.
Use --plan-only for a standard-library-only audit without importing matplotlib.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AXES = {"graph_signal": "S_G", "kappa": "R_true", "tau": "H",
        "delta": "D_pred", "eta": "D_struct"}
SERIES = {"graph_signal": ["Static-GCN_minus_MLP", "Static-GAT_minus_MLP"],
          "kappa": ["Architecture_Static", "Architecture_Dynamic"],
          "tau": ["Static-GCN_minus_MLP", "Static-GAT_minus_MLP"],
          "delta": ["Dynamics_GCN", "Dynamics_GAT"],
          "eta": ["Dynamics_GCN", "Dynamics_GAT"]}
PHASES = [("S_G", "R_hat", "Architecture_Static"),
          ("D_struct", "D_pred", "Dynamics_GCN"),
          ("S_G", "H", "Static-GCN_minus_MLP")]
ROBUST_SERIES = ["Static-GCN_minus_MLP", "Static-GAT_minus_MLP",
                 "Architecture_Static", "GATv2_minus_GAT_Static"]
LABELS = {"graph_signal": r"$\lambda$", "kappa": r"$\kappa$", "tau": r"$\tau$",
          "delta": r"$\delta$", "eta": r"$\eta$", "noise": r"$\sigma$",
          "n_nodes": r"$N$", "S_G": r"$S_G$", "H": r"$H$",
          "R_hat": r"$\widehat{R}$", "R_true": r"$R_{\mathrm{true}}$ (oracle)",
          "D_struct": r"$D_{\mathrm{struct}}$", "D_pred": r"$D_{\mathrm{pred}}$",
          "architecture_control": r"$\kappa$", "classification": r"$\kappa$",
          "query_dependent": r"$\kappa$", "graph_family": r"Graph family / $\kappa$"}
CONTRAST_LABELS = {
    "Static-GCN_minus_MLP": r"$\mathrm{GCN}_{S}-\mathrm{MLP}$",
    "Static-GAT_minus_MLP": r"$\mathrm{GAT}_{S}-\mathrm{MLP}$",
    "Architecture_Static": r"$\mathrm{GAT}_{S}-\mathrm{GCN}_{S}$",
    "Architecture_Dynamic": r"$\mathrm{GAT}_{D}-\mathrm{GCN}_{D}$",
    "Dynamics_GCN": r"$\mathrm{GCN}_{D}-\mathrm{GCN}_{S}$",
    "Dynamics_GAT": r"$\mathrm{GAT}_{D}-\mathrm{GAT}_{S}$",
    "GATv2_minus_GAT_Static": r"$\mathrm{GATv2}_{S}-\mathrm{GAT}_{S}$"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def numeric(row, key):
    try:
        value = float(row[key])
        return value if math.isfinite(value) else None
    except (ValueError, KeyError, TypeError):
        return None


def series_rows(rows, contrast, key, positions=None):
    selected = [r for r in rows if r["contrast"] == contrast]
    selected.sort(key=lambda r: float(r["value"]) if positions is None else positions[r["value"]])
    data = []
    for row in selected:
        x = (positions[row["value"]] if positions is not None else
             float(row["value"]) if key == "knob" else numeric(row, key))
        if x is not None and numeric(row, "mean_gap") is not None:
            data.append({"condition_id": row["condition_id"], "x": x,
                         **{k: numeric(row, k) for k in ("mean_gap", "ci_low", "ci_high")}})
    return {"contrast": contrast, "points": data}


def build_plans(rows, group):
    plans = []
    for axis, diagnostic in AXES.items():
        subset = [r for r in rows if r["stage"] == "B" and r["axis"] == axis]
        for task in sorted({r["task"] for r in subset}):
            selected = [r for r in subset if r["task"] == task]
            plans.append(dict(group=group, filename=f"response_{axis}_{task}.png", kind="response",
                              axis=axis, diagnostic=diagnostic, task=task,
                              seeds=min(int(r["seeds"]) for r in selected),
                              panels=[[series_rows(selected, c, key) for c in SERIES[axis]]
                                      for key in ("knob", diagnostic)]))
    for x, y, contrast in PHASES:
        seen, points = set(), []
        for row in rows:
            if row["stage"] not in ("C", "D") or row["contrast"] != contrast or row["task"] != "regression":
                continue
            if row["condition_id"] in seen:
                continue
            seen.add(row["condition_id"])
            values = [numeric(row, key) for key in (x, y, "mean_gap")]
            if None not in values:
                points.append(dict(condition_id=row["condition_id"], x=values[0], y=values[1], mean_gap=values[2]))
        if points:
            plans.append(dict(group=group, filename=f"phase_{x}_{y}_{contrast}.png", kind="phase",
                              x=x, y=y, contrast=contrast, points=points))
    for stage in ("sample_size", "robustness"):
        keys = sorted({(r["axis"], r["task"]) for r in rows if r["stage"] == stage})
        for axis, task in keys:
            selected = [r for r in rows if (r["stage"], r["axis"], r["task"]) == (stage, axis, task)]
            labels = list(dict.fromkeys(r["value"] for r in selected))
            if all(numeric({"value": v}, "value") is not None for v in labels):
                labels.sort(key=float)
            positions = {value: i for i, value in enumerate(labels)}
            series = [series_rows(selected, c, "position", positions) for c in ROBUST_SERIES
                      if any(r["contrast"] == c for r in selected)]
            plans.append(dict(group=group, filename=f"{stage}_{axis}_{task}.png", kind="categorical",
                              stage=stage, axis=axis, task=task, labels=labels, series=series,
                              seeds=min(int(r["seeds"]) for r in selected)))
    return plans


def category_label(value):
    value = value.strip('"')
    if "_k" in value:
        family, kappa = value.rsplit("_k", 1)
        return {"heterogeneous": "Heterogeneous", "sbm": "SBM", "knn": "kNN"}.get(family, family) + rf" $\kappa={kappa}$"
    return f"${value}$" if numeric({"v": value}, "v") is not None else value


def draw_series(ax, series, categorical=False):
    points = series["points"]
    ready = [p for p in points if p["ci_low"] is not None and p["ci_high"] is not None]
    label = CONTRAST_LABELS[series["contrast"]]
    if categorical and not ready:
        ax.scatter([p["x"] for p in points], [p["mean_gap"] for p in points], label=label)
    elif not categorical and len(ready) != len(points):
        ax.plot([p["x"] for p in points], [p["mean_gap"] for p in points], marker="o", label=label)
    elif ready:
        low = [max(p["mean_gap"] - p["ci_low"], 0) for p in ready]
        high = [max(p["ci_high"] - p["mean_gap"], 0) for p in ready]
        artist = ax.errorbar([p["x"] for p in ready], [p["mean_gap"] for p in ready],
                             yerr=[low, high], marker="o", capsize=3, label=label)
        assert list(artist.lines[0].get_xdata()) == [p["x"] for p in ready]
        assert list(artist.lines[0].get_ydata()) == [p["mean_gap"] for p in ready]


def render(plan, output, plt):
    ylabel = r"Paired $\Delta\mathrm{AP}$" if plan.get("task") == "classification" else r"Paired $\Delta R^2$"
    if plan["kind"] == "response":
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, series, label in zip(axes, plan["panels"], (plan["axis"], plan["diagnostic"])):
            for item in series:
                draw_series(ax, item)
            ax.axhline(0, color="gray", linewidth=.7)
            ax.set_xlabel(LABELS[label])
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=7)
            ax.grid(alpha=.2)
        fig.suptitle(f'{LABELS[plan["axis"]]}: {plan["task"]}; {plan["seeds"]} seeds', fontsize=11)
    elif plan["kind"] == "phase":
        fig, ax = plt.subplots(figsize=(6, 4))
        points = plan["points"]
        limit = max(max(abs(p["mean_gap"]) for p in points), 1e-4)
        scatter = ax.scatter([p["x"] for p in points], [p["y"] for p in points],
                             c=[p["mean_gap"] for p in points], cmap="RdBu_r", vmin=-limit,
                             vmax=limit, s=35, edgecolors="gray", linewidths=.3)
        ax.set_xlabel(LABELS[plan["x"]])
        ax.set_ylabel(LABELS[plan["y"]])
        ax.set_title(CONTRAST_LABELS[plan["contrast"]] + "; observable diagnostics", fontsize=10)
        fig.colorbar(scatter, ax=ax, label=r"Mean paired $\Delta R^2$")
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        for item in plan["series"]:
            draw_series(ax, item, categorical=True)
        ax.axhline(0, color="gray", linewidth=.7)
        ax.set_xticks(range(len(plan["labels"])), [category_label(v) for v in plan["labels"]],
                      rotation=20 if len(plan["labels"]) > 4 else 0)
        ax.set_xlabel(LABELS[plan["axis"]])
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
        ax.set_title(f'{plan["stage"]}: {LABELS[plan["axis"]]}; {plan["seeds"]} seeds', fontsize=10)
    fig.tight_layout()
    destination = output / plan["group"] / plan["filename"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    output, frozen = ROOT / "dist/assets/figures-latex", ROOT / "dist/data/mechanisms-v2"
    sources, plans = {}, []
    for group in ("main", "robustness"):
        source = frozen / group / "paired_contrasts.csv"
        sources[str(source.relative_to(ROOT))] = sha(source)
        with source.open(encoding="utf-8-sig", newline="") as handle:
            plans.extend(build_plans(list(csv.DictReader(handle)), group))
    originals = {}
    for plan in plans:
        original = frozen / plan["group"] / "figures" / plan["filename"]
        originals[str(original.relative_to(ROOT))] = sha(original)
    assert len(plans) == 14, "Expected exactly the 14 published presentation figures"
    provenance = frozen / "provenance.json"
    sources[str(provenance.relative_to(ROOT))] = sha(provenance)
    manifest = dict(schema_version=1, purpose="LaTeX presentation copies only; no statistical recomputation",
                    source_sha256=sources, original_figure_sha256=originals,
                    plotting_plan_sha256=digest(plans), figures=[])
    if args.plan_only:
        print(json.dumps(dict(figures=len(plans), plotting_plan_sha256=digest(plans),
                              numerical_libraries_imported=False), indent=2))
        return
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib-latex"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"text.usetex": False, "axes.formatter.use_mathtext": True}):
        for plan in plans:
            destination = render(plan, output, plt)
            manifest["figures"].append(dict(path=str(destination.relative_to(ROOT)).replace("\\", "/"),
                sha256=sha(destination), series_data_sha256=digest(plan), data=plan))
            print(f'Rendered {plan["group"]}/{plan["filename"]}', flush=True)
    for path, expected in {**sources, **originals}.items():
        assert sha(ROOT / path) == expected, f"Frozen source changed: {path}"
    manifest.update(completed_at=datetime.now(timezone.utc).isoformat(),
                    matplotlib_version=matplotlib.__version__, frozen_sources_unchanged=True)
    (output / "presentation_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Complete: {len(plans)} presentation figures; frozen sources unchanged.", flush=True)


if __name__ == "__main__":
    main()
