"""Isolated V2 artifacts, frozen source/config provenance, checkpoints and resume."""
import hashlib
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import time
import uuid
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
from .config import plans
from .data import generate
from .diagnostics import measure
from .models import MODEL_NAMES,EXTRA_MODELS
from .training import prepare,train_model
from .reporting import export


def retry_io(operation):
    """Bounded retries for transient Windows access/sharing failures."""
    for attempt in range(9):
        try:
            return operation()
        except OSError as error:
            if getattr(error,"winerror",None) not in {5,32,33} or attempt==8:
                raise
            time.sleep(min(.05*2**attempt,1.))


def write_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    payload=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)
    try:
        retry_io(lambda: temporary.write_text(payload,encoding="utf-8"))
        retry_io(lambda: os.replace(temporary,path))
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass  # Cleanup must not replace the original error.


def read_json(path):
    return json.loads(retry_io(lambda: Path(path).read_text(encoding="utf-8")))


def source_hash():
    digest=hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    runner=Path(__file__).parents[2]/"run_mechanisms.py"
    digest.update(runner.read_bytes())
    return digest.hexdigest()


def workload(config,stages,entries=None):
    plan=list(entries) if entries is not None else plans(config,stages)
    if not plan:
        raise ValueError("Empty experiment plan.")
    seeds=config.get("seeds",[7,19,31,43,59])
    if not seeds or len(set(seeds))!=len(seeds) or any(not isinstance(s,int) or s<0 for s in seeds):
        raise ValueError("Seeds must be distinct nonnegative integers.")
    unique={entry["condition"].identifier:entry["condition"] for entry in plan}
    extras=config.get("gatv2",False)
    model_count=7 if extras else 5
    candidates=len(config.get("training",{}).get("widths",[32,64]))*len(
        config.get("training",{}).get("learning_rates",[.001,.003]))
    if not candidates:
        raise ValueError("No candidate fits.")
    return dict(stages=stages,plan_references=len(plan),unique_conditions=len(unique),
                seeds=seeds,datasets=len(unique)*len(seeds),
                model_evaluations=len(unique)*len(seeds)*model_count,
                candidate_fits=len(unique)*len(seeds)*model_count*candidates,
                purpose=config.get("purpose","research"))


def existing_result(folder):
    files=["result.json","predictions.npz","weights.pt","training.json"]
    return read_json(folder/"result.json") if all((folder/name).is_file() for name in files) else None


def run(config,stages,output,entries=None):
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    plan=list(entries) if entries is not None else plans(config,stages)
    protocol=dict(config=config,stages=stages,source_sha256=source_hash())
    if entries is not None:
        protocol["assigned_plan"]=[dict(stage=e["stage"],axis=e["axis"],value=e["value"],
                                        condition=e["condition"].to_dict()) for e in plan]
    fingerprint=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    manifest_path=output/"manifest.json"
    if manifest_path.exists() and read_json(manifest_path)["fingerprint"]!=fingerprint:
        raise ValueError("Output belongs to a different source/config. Use a new directory.")
    lock=output/"run.lock"
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(descriptor,str(os.getpid()).encode())
    os.close(descriptor)
    try:
        if not manifest_path.exists():
            versions={name:importlib.metadata.version(name) for name in
                      ("numpy","torch","scikit-learn","scipy","networkx","pandas","matplotlib")}
            write_json(manifest_path,dict(**protocol,fingerprint=fingerprint,
                created_at=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),
                packages=versions,workload=workload(config,stages,plan)))
            source=output/"source"
            source.mkdir(exist_ok=True)
            for path in Path(__file__).parent.glob("*.py"):
                shutil.copy2(path,source/path.name)
            shutil.copy2(Path(__file__).parents[2]/"run_mechanisms.py",source/"run_mechanisms.py")
        seeds=config.get("seeds",[7,19,31,43,59])
        models=MODEL_NAMES+(EXTRA_MODELS if config.get("gatv2",False) else [])
        total=workload(config,stages,plan)["model_evaluations"]
        completed_keys=set()
        dataset_cache={}
        records=[]
        for entry in plan:
            condition=entry["condition"]
            identifier=condition.identifier
            for seed in seeds:
                key=(identifier,seed)
                folder=output/"datasets"/identifier/f"seed_{seed}"
                folder.mkdir(parents=True,exist_ok=True)
                cached=dataset_cache.get(key)
                if cached is None:
                    results={name:existing_result(folder/name) for name in models}
                    if all(results.values()) and (folder/"diagnostics.json").exists():
                        diagnostics=read_json(folder/"diagnostics.json")
                    else:
                        dataset=generate(condition,seed)
                        write_json(folder/"generator.json",dict(condition=condition.to_dict(),
                            seed=seed,oracle=dataset.oracle,
                            origin_times=dataset.origin_times.tolist(),
                            target_times=dataset.target_times.tolist(),
                            split_target_times={name:dataset.target_times[ix].tolist()
                                                for name,ix in dataset.splits.items()}))
                        diagnostics=measure(dataset,config.get("diagnostics",{}))
                        write_json(folder/"diagnostics.json",diagnostics)
                        if config.get("save_datasets",False):
                            np.savez_compressed(folder/"dataset.npz",features=dataset.features,
                                target=dataset.target,latent=dataset.latent,true_graphs=dataset.true_graphs,
                                observed_graphs=dataset.observed_graphs,static_graph=dataset.static_graph,
                                origin_times=dataset.origin_times,target_times=dataset.target_times)
                        data=prepare(dataset,config.get("training",{}).get("device","cpu"))
                        fixed_topology=bool(np.all(dataset.observed_graphs==dataset.static_graph))
                        for name in models:
                            if results[name] is not None:
                                continue
                            equivalent=name.replace("Dynamic-","Static-",1)
                            if (config.get("reuse_fixed_topology",False) and fixed_topology
                                    and name.startswith("Dynamic-") and results.get(equivalent)):
                                # This model has identical tensors, initialization and candidate grid.
                                # Preserve every evaluation; record that no extra fits were executed.
                                previous=folder/equivalent
                                result=copy.deepcopy(results[equivalent])
                                result.update(model=name,training_seconds=0.,candidate_fits_executed=0,
                                    training_reuse=dict(kind="exact_fixed_topology_equivalence",
                                                        source_model=equivalent))
                                with np.load(previous/"predictions.npz",allow_pickle=False) as saved:
                                    predictions={key:saved[key] for key in saved.files}
                                weights=torch.load(previous/"weights.pt",map_location="cpu",weights_only=True)
                                candidates=read_json(previous/"training.json")
                            else:
                                result,predictions,weights,candidates=train_model(
                                    name,data,config.get("training",{}),seed)
                                result["candidate_fits_executed"]=len(candidates)
                            destination=folder/name
                            destination.mkdir(exist_ok=True)
                            torch.save(weights,destination/"weights.pt")
                            np.savez_compressed(destination/"predictions.npz",**predictions)
                            write_json(destination/"training.json",candidates)
                            write_json(destination/"result.json",result)
                            results[name]=result
                            completed_keys.add((identifier,seed,name))
                            write_json(output/"status.json",dict(status="running",condition_id=identifier,
                                data_seed=seed,model=name,completed_in_this_invocation=len(completed_keys),
                                planned_model_evaluations=total,updated_at=datetime.now(timezone.utc).isoformat()))
                            print(f"{identifier} seed={seed} {name}: {result['test']}",flush=True)
                    cached=diagnostics,results
                    dataset_cache[key]=cached
                diagnostics,results=cached
                for name,result in results.items():
                    completed_keys.add((identifier,seed,name))
                    row=dict(stage=entry["stage"],axis=entry["axis"],
                        value=json.dumps(entry["value"],sort_keys=True),
                        condition_id=identifier,seed=seed,model=name,
                        **condition.to_dict(),**{key:diagnostics.get(key) for key in
                        ("S_X","S_G","S_G_shuffled","H","R_hat","R_true",
                         "R_estimation_error","D_struct","D_pred")})
                    row.update({f"test_{key}":value for key,value in result["test"].items()})
                    row.update(validation_loss=result["selection"]["validation_loss"],
                               width=result["selection"]["width"],
                               learning_rate=result["selection"]["learning_rate"],
                               best_epoch=result["selection"]["best_epoch"],
                               parameter_count=result["selection"]["parameter_count"],
                               training_seconds=result["training_seconds"])
                    row.update(candidate_fits_executed=result.get("candidate_fits_executed",result["candidates"]),
                               reused_training=bool(result.get("training_reuse")),
                               selected_budget_exhausted=result["selection"].get("budget_exhausted",False))
                    records.append(row)
        payload=export(pd.DataFrame(records),output,config.get("bootstrap_repetitions",2000),
                       config.get("purpose","research"))
        write_json(output/"status.json",dict(status="complete",model_evaluations=len(completed_keys),
                   reference_rows=len(records),updated_at=datetime.now(timezone.utc).isoformat()))
        write_json(output/"complete.json",dict(fingerprint=fingerprint,
                   model_evaluations=len(completed_keys),source_sha256=protocol["source_sha256"]))
        return dict(output=str(output),model_evaluations=len(completed_keys),
                    reference_rows=len(records),contrasts=len(payload["contrasts"]))
    except BaseException as error:
        write_json(output/"status.json",dict(status="interrupted" if isinstance(error,KeyboardInterrupt)
            else "failed",error=str(error),updated_at=datetime.now(timezone.utc).isoformat()))
        raise
    finally:
        lock.unlink(missing_ok=True)
