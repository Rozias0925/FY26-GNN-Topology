"""Freeze a training-only external expectation, then compare all five models.

This adapter does not assert the current Shihmen data is sufficient for an
external event study. It requires a verified, aligned training-only panel.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .diagnostics import probe_diagnostics,estimate_relevance
from .models import MODEL_NAMES
from .reporting import CONTRASTS
from .experiment import write_json

OBSERVABLE=["S_X","S_G","H","R_hat","D_struct","D_pred"]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(panel_path,synthetic_folder,destination,gate_epochs=80):
    destination=Path(destination)
    if destination.exists():
        raise FileExistsError("Preregistered expectations are immutable; use a new filename.")
    with np.load(panel_path,allow_pickle=False) as panel:
        required={"features","target","graphs","target_times","metadata"}
        if not required<=set(panel.files):
            raise ValueError(f"Training panel requires {sorted(required)}.")
        metadata=json.loads(str(panel["metadata"].item()))
        if metadata.get("training_only") is not True or "training_cutoff" not in metadata:
            raise ValueError("Explicit training-only provenance and cutoff are required.")
        times=panel["target_times"]
        if np.issubdtype(times.dtype,np.number):
            within=times<=float(metadata["training_cutoff"])
        else:
            within=times.astype("datetime64[ns]")<=np.datetime64(metadata["training_cutoff"])
        if not within.all():
            raise ValueError("Panel contains targets after the declared training cutoff.")
        x,y,graphs=panel["features"].copy(),panel["target"].copy(),panel["graphs"].copy()
        ordered=times if np.issubdtype(times.dtype,np.number) else times.astype("datetime64[ns]")
        if len(times)!=len(x) or not np.all(ordered[1:]>ordered[:-1]):
            raise ValueError("Target times must be strictly increasing and aligned.")
        if not metadata.get("target_alignment"):
            raise ValueError("Explicit target alignment provenance is required.")
        if np.any((graphs!=0)&(graphs!=1)):
            raise ValueError("This adapter expects binary undirected topology.")
    synthetic_folder=Path(synthetic_folder)
    if not (synthetic_folder/"complete.json").exists():
        raise ValueError("Synthetic study must be complete before preregistration.")
    manifest=json.loads((synthetic_folder/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("config",{}).get("purpose")!="research":
        raise ValueError("Smoke outputs cannot support a real-world preregistration.")
    diagnostic=probe_diagnostics(x,y,graphs)
    diagnostic["R_hat"]=estimate_relevance(x,y,graphs,epochs=gate_epochs)
    source=pd.read_csv(synthetic_folder/"metrics.csv",dtype={"condition_id":str})
    if not (source.task=="regression").any():
        raise ValueError("Need regression synthetic maps for these R2 diagnostics.")
    source=source[(source.task=="regression")&(source.regime=="mechanism")&
                  (source.attention=="source")]
    points=source.drop_duplicates(["condition_id","seed"]).groupby("condition_id")[OBSERVABLE].mean()
    if points.empty or not np.isfinite([diagnostic[key] for key in OBSERVABLE]).all():
        raise ValueError("No finite comparable observable diagnostics.")
    scale=np.maximum(points.std(ddof=0).to_numpy(),.05)
    vector=np.array([diagnostic[key] for key in OBSERVABLE])
    distance=np.linalg.norm((points.to_numpy()-vector)/scale,axis=1)
    nearest=points.index[np.argsort(distance)[:min(10,len(points))]].tolist()
    contrasts=pd.read_csv(synthetic_folder/"paired_contrasts.csv",dtype={"condition_id":str})
    local=contrasts[(contrasts.condition_id.isin(nearest))&(contrasts.task=="regression")]
    expectations={}
    for name,group in local.groupby("contrast"):
        unique=group.drop_duplicates("condition_id")
        expectations[name]=dict(mean_local_gap=float(unique.mean_gap.mean()),
            local_gap_min=float(unique.mean_gap.min()),local_gap_max=float(unique.mean_gap.max()),
            interpretation="descriptive nearby-condition range, not a prediction interval")
    within_range={key:bool(points[key].min()<=diagnostic[key]<=points[key].max())
                  for key in OBSERVABLE}
    support=dict(diagnostic_ranges=within_range,node_count=int(x.shape[1]),
        synthetic_node_range=[int(source.n_nodes.min()),int(source.n_nodes.max())],
        node_count_within_range=bool(source.n_nodes.min()<=x.shape[1]<=source.n_nodes.max()))
    payload=dict(schema_version=2,created_at=datetime.now(timezone.utc).isoformat(),
        dataset_id=metadata.get("dataset_id","external_training_panel"),
        training_panel_sha256=file_hash(panel_path),training_provenance=metadata,
        synthetic_manifest_sha256=file_hash(synthetic_folder/"manifest.json"),
        synthetic_metrics_sha256=file_hash(synthetic_folder/"metrics.csv"),
        diagnostics=diagnostic,support=support,nearest_condition_ids=nearest,
        expectation=expectations,primary_score="r2",
        protocol="Five models must all be evaluated; no winner is preregistered.",
        limitations="Snapshot one-step synthetic study does not establish 1/3/6-hour event forecasting validity.")
    write_json(destination,payload)
    return payload


def compare(expectation_path,heldout_path,destination):
    expected=json.loads(Path(expectation_path).read_text(encoding="utf-8"))
    actual=json.loads(Path(heldout_path).read_text(encoding="utf-8"))
    if actual.get("expectation_sha256")!=file_hash(expectation_path):
        raise ValueError("Held-out report must identify the frozen expectation checksum.")
    if actual.get("dataset_id")!=expected["dataset_id"]:
        raise ValueError("External dataset identity mismatch.")
    evaluated=datetime.fromisoformat(actual["evaluated_at"])
    registered=datetime.fromisoformat(expected["created_at"])
    if evaluated<=registered:
        raise ValueError("Held-out evaluation must occur after preregistration.")
    models=actual.get("models",{})
    if not set(MODEL_NAMES)<=set(models):
        raise ValueError("All five models must be reported.")
    rows=[]
    for name,coefficients in CONTRASTS.items():
        gap=sum(float(models[model]["test_r2"])*weight for model,weight in coefficients.items())
        prediction=expected["expectation"].get(name,{})
        local_gap=prediction.get("mean_local_gap")
        rows.append(dict(contrast=name,expected_local_gap=local_gap,observed_gap=gap,
                         same_direction=bool(gap*local_gap>0) if local_gap is not None else None))
    report=dict(dataset_id=actual["dataset_id"],expectation_sha256=file_hash(expectation_path),
                comparisons=rows,support=expected["support"],heldout_report=actual,
                scope="External explanation check; no evidence of operational response optimization.")
    write_json(destination,report)
    return report
