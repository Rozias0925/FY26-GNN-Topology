"""Resumable, paired full-study queue; bounded Windows-safe process workers."""
from collections import OrderedDict, defaultdict
import ast
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import contextlib
import copy
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback

import pandas as pd

from .config import plans
from .experiment import run, source_hash, read_json, write_json, workload, existing_result, retry_io
from .reporting import export, bootstrap


def now():
    return datetime.now(timezone.utc).isoformat()


def event_record(result):
    # Failed worker records already contain updated_at. Keep it, and add a
    # distinct supervisor timestamp without passing duplicate keyword arguments.
    return {**result,"recorded_at":now()}


def function_ast(path,name):
    for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name==name:
            return ast.dump(node,include_attributes=False)
    raise ValueError(f"Missing cache-safety function {name}")


def import_checkpoint_cache(source,output,jobs,config):
    """Explicit recovery across I/O-only revisions; never alter the old batch."""
    source,output=Path(source).resolve(),Path(output).resolve()
    if source==output:
        raise ValueError("Recovery must use a new output directory.")
    old=read_json(source/"manifest.json")
    if old["config"]!=config:
        raise ValueError("Recovery configuration differs from the original study.")
    if (source/"suite.lock").exists():
        owner=int((source/"suite.lock").read_text(encoding="utf-8"))
        if process_alive(owner):
            raise RuntimeError("Cannot import checkpoints from an active suite.")
    package=Path(__file__).parent
    # Exact computational modules, plus AST equality for orchestration functions
    # that construct examples, fit/select/score and write model checkpoints.
    for name in ("__init__.py","config.py","data.py","graphs.py","models.py",
                 "training.py","diagnostics.py","reporting.py","real_validation.py"):
        if (source/"source"/name).read_bytes()!=(package/name).read_bytes():
            raise ValueError(f"Unsafe checkpoint reuse: {name} changed.")
    for name in ("run","workload","existing_result"):
        if function_ast(source/"source/experiment.py",name)!=function_ast(package/"experiment.py",name):
            raise ValueError(f"Unsafe checkpoint reuse: experiment.{name} changed.")
    for name in ("build_jobs","balanced"):
        if function_ast(source/"source/suite.py",name)!=function_ast(package/"suite.py",name):
            raise ValueError(f"Unsafe checkpoint reuse: suite.{name} changed.")
    imported=0
    for job in jobs:
        previous=source/"jobs"/job["id"]
        if not (previous/"manifest.json").exists():
            continue
        previous_manifest=read_json(previous/"manifest.json")
        if previous_manifest["config"]!=job["config"]:
            raise ValueError(f"Job configuration mismatch: {job['id']}")
        destination=output/"jobs"/job["id"]
        index={}
        for dataset in (previous/"datasets").glob("*/*"):
            if not (dataset/"diagnostics.json").exists():
                continue
            target=destination/"datasets"/dataset.parent.name/dataset.name
            model_folders=[p for p in dataset.iterdir() if p.is_dir() and existing_result(p) is not None]
            if not model_folders:
                continue
            target.mkdir(parents=True,exist_ok=True)
            for name in ("generator.json","diagnostics.json"):
                if (dataset/name).exists():
                    retry_io(lambda: shutil.copy2(dataset/name,target/name))
            for model_folder in model_folders:
                model_target=target/model_folder.name
                model_target.mkdir(parents=True,exist_ok=True)
                for name in ("weights.pt","predictions.npz","training.json","result.json"):
                    original=model_folder/name
                    if not original.stat().st_size:
                        raise ValueError(f"Empty checkpoint: {original}")
                    retry_io(lambda: shutil.copy2(original,model_target/name))
                key=f"{dataset.parent.name}/{dataset.name}/{model_folder.name}"
                index[key]=dict(evaluation_source_sha256=old["source_sha256"],
                    source_result=str((model_folder/"result.json").resolve()),
                    result_sha256=hashlib.sha256((model_folder/"result.json").read_bytes()).hexdigest(),
                    weights_sha256=hashlib.sha256((model_folder/"weights.pt").read_bytes()).hexdigest())
                imported+=1
        if index:
            write_json(destination/"checkpoint_import.json",dict(source=str(previous),models=index))
    provenance=dict(source=str(source),source_sha256=old["source_sha256"],
        original_manifest_sha256=hashlib.sha256((source/"manifest.json").read_bytes()).hexdigest(),
        imported_model_evaluations=imported,imported_at=now(),
        policy="Exact computation modules/AST/config verified; only I/O and supervisor recovery changed.")
    write_json(output/"checkpoint_import.json",provenance)
    return provenance


def process_alive(pid):
    """Read process liveness without sending signals on Windows."""
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return ctypes.get_last_error() != 87  # Access denied is not proof of death.
        try:
            code = ctypes.c_ulong()
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def clear_dead_lock(path):
    path = Path(path)
    if path.exists():
        pid = int(path.read_text(encoding="utf-8").strip())
        if process_alive(pid):
            raise RuntimeError(f"Active worker {pid} owns {path}")
        path.unlink()  # Single known lock file; completed checkpoints are retained.


def balanced(entries):
    """Interleave response axes so all five curves acquire points early."""
    buckets = OrderedDict()
    for entry in entries:
        buckets.setdefault(entry["axis"], []).append(entry)
    result = []
    for index in range(max(map(len, buckets.values()), default=0)):
        result.extend(values[index] for values in buckets.values() if index < len(values))
    return result


def build_jobs(config):
    base = copy.deepcopy(config)
    options = base.pop("suite", {})
    main_stages = options.get("main_stages", ["B", "C", "D", "sample_size"])
    if not main_stages or any(s not in {"A", "B", "C", "D", "sample_size"} for s in main_stages):
        raise ValueError("Invalid main stages.")
    base["gatv2"] = False
    base["stages"] = main_stages
    entries = plans(base, main_stages)
    # These priorities determine scheduling only; hypotheses do not gate execution.
    priority = {"B": 0, "convergence": 1, "sample_size": 2,
                "robustness": 3, "C": 4, "D": 5, "A": 0}
    ordering = balanced([e for e in entries if e["stage"] in {"A", "B"}])
    ordering += [e for e in entries if e["stage"] not in {"A", "B"}]
    groups = OrderedDict()
    for entry in ordering:
        groups.setdefault(entry["condition"].identifier, []).append(entry)
    jobs = []
    for identifier, references in groups.items():
        jobs.append(dict(id=f"main/{identifier}", group="main", config=copy.deepcopy(base),
                         entries=references, priority=min(priority[e["stage"]] for e in references)))
    if options.get("convergence", True):
        longer = copy.deepcopy(base)
        longer["training"]["epochs"] = options.get("convergence_epochs", 400)
        if longer["training"]["epochs"] <= base["training"].get("epochs", 200):
            raise ValueError("Convergence budget must exceed the primary budget.")
        for entry in plans(longer, ["A"]):
            jobs.append(dict(id=f"convergence/{entry['condition'].identifier}", group="convergence",
                             config=copy.deepcopy(longer), entries=[entry], priority=1))
    if options.get("robustness", True):
        robust = copy.deepcopy(base)
        robust["gatv2"] = True
        for entry in plans(robust, ["robustness"]):
            jobs.append(dict(id=f"robustness/{entry['condition'].identifier}", group="robustness",
                             config=copy.deepcopy(robust), entries=[entry], priority=3))
    jobs.sort(key=lambda job: job["priority"])
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate queue job IDs.")
    for job in jobs:
        job["stages"] = list(dict.fromkeys(e["stage"] for e in job["entries"]))
        job["workload"] = workload(job["config"], job["stages"], job["entries"])
    return jobs


def plan_summary(config, jobs):
    groups = {}
    for name in dict.fromkeys(j["group"] for j in jobs):
        selected = [j for j in jobs if j["group"] == name]
        groups[name] = dict(conditions=len(selected),
            model_evaluations=sum(j["workload"]["model_evaluations"] for j in selected),
            candidate_evaluations=sum(j["workload"]["candidate_fits"] for j in selected))
    return dict(workers=config.get("suite", {}).get("workers", 2), groups=groups,
        jobs=len(jobs), model_evaluations=sum(g["model_evaluations"] for g in groups.values()),
        candidate_evaluations=sum(g["candidate_evaluations"] for g in groups.values()),
        reuse_policy="Identical fixed-topology dynamic problems reuse static candidate checkpoints; explicitly labelled.",
        priority=["B balanced axes", "400-epoch A sensitivity", "sample_size", "robustness", "C", "D"])


def worker(job, output, expected_source):
    folder = Path(output) / "jobs" / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "worker.stdout.log").open("a", encoding="utf-8", buffering=1) as stdout, \
         (folder / "worker.stderr.log").open("a", encoding="utf-8", buffering=1) as stderr, \
         contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            if source_hash() != expected_source:
                raise RuntimeError("Source changed after the suite was frozen; use a new output directory.")
            clear_dead_lock(folder / "run.lock")
            result = run(job["config"], job["stages"], folder, entries=job["entries"])
            return dict(job=job["id"], status="complete", **result)
        except Exception as error:
            traceback.print_exc()
            failure = dict(job=job["id"], status="failed", error=str(error), updated_at=now())
            write_json(folder / "queue_error.json", failure)
            return failure


def progress(jobs, output, active, completed, failures):
    counts = defaultdict(int)
    running = []
    for job in jobs:
        folder = Path(output) / "jobs" / job["id"]
        count = len(list((folder / "datasets").glob("*/*/*/result.json")))
        counts[job["group"]] += count
        if job["id"] in active:
            detail = dict(job=job["id"], completed_models=count,
                          planned_models=job["workload"]["model_evaluations"])
            status_path = folder / "status.json"
            if status_path.exists():
                detail["worker"] = read_json(status_path)
            running.append(detail)
    value=dict(status="running", updated_at=now(), active_jobs=running,
        completed_jobs=len(completed), planned_jobs=len(jobs), failures=failures,
        completed_model_evaluations=sum(counts.values()),
        planned_model_evaluations=sum(j["workload"]["model_evaluations"] for j in jobs),
        completed_models_by_group=dict(counts))
    imported=Path(output)/"checkpoint_import.json"
    if imported.exists():
        value["imported_model_evaluations"]=read_json(imported)["imported_model_evaluations"]
        value["new_model_evaluations"]=max(0,value["completed_model_evaluations"]-value["imported_model_evaluations"])
    return value


def read_metrics_csv(path):
    """Condition hashes are identifiers, even when they look like numbers."""
    return pd.read_csv(path, keep_default_na=False, dtype={"condition_id": str})


def refresh_reports(jobs, output, completed, config, *, evaluation_source=None):
    """Export only completely paired condition jobs; mark every partial report."""
    # A report-only repair retains the source of the completed evaluations.
    pipeline_source=evaluation_source or source_hash()
    for group in dict.fromkeys(j["group"] for j in jobs):
        assigned = [j for j in jobs if j["group"] == group]
        finished = [j for j in assigned if j["id"] in completed]
        if not finished:
            continue
        tables=[]
        for job in finished:
            folder=Path(output)/"jobs"/job["id"]
            table=read_metrics_csv(folder/"metrics.csv")
            cache=read_json(folder/"checkpoint_import.json")["models"] if (folder/"checkpoint_import.json").exists() else {}
            origins=[]
            for row in table.itertuples():
                key=f"{row.condition_id}/seed_{row.seed}/{row.model}"
                original=cache.get(key)
                if original is None and str(row.reused_training).lower()=="true" and row.model.startswith("Dynamic-"):
                    original=cache.get(f"{row.condition_id}/seed_{row.seed}/{row.model.replace('Dynamic-','Static-',1)}")
                origins.append(original["evaluation_source_sha256"] if original else pipeline_source)
            table["evaluation_source_sha256"]=origins
            tables.append(table)
        frame = pd.concat(tables, ignore_index=True)
        destination = Path(output) / "reports" / group
        destination.mkdir(parents=True, exist_ok=True)
        done = len(finished) == len(assigned)
        write_json(destination / "status.json", dict(status="updating", updated_at=now()))
        payload = export(frame, destination, config.get("bootstrap_repetitions", 2000),
                         config.get("purpose", "research"))
        payload.update(complete=done, completed_conditions=len(finished),
                       planned_conditions=len(assigned), suite_source_sha256=pipeline_source)
        write_json(destination / "website_results_v2.json", payload)
        write_json(destination / "manifest.json", dict(config=assigned[0]["config"],
            source_sha256=pipeline_source, completed_conditions=len(finished),
            planned_conditions=len(assigned), suite_output=str(Path(output).resolve()),
            evaluation_source_sha256s=sorted(frame.evaluation_source_sha256.unique().tolist())))
        unique = frame.drop_duplicates(["condition_id", "seed", "model"])
        convergence = unique[["condition_id", "axis", "seed", "model", "best_epoch",
                              "selected_budget_exhausted", "reused_training", "candidate_fits_executed"]]
        convergence.to_csv(destination / "convergence_flags.csv", index=False)
        write_json(destination / "status.json", dict(status="complete" if done else "partial",
            updated_at=now(), completed_conditions=len(finished), planned_conditions=len(assigned),
            unique_model_evaluations=len(unique), reference_rows=len(frame),
            candidate_fits_executed=int(pd.to_numeric(unique.candidate_fits_executed).sum())))
        if done:
            write_json(destination / "complete.json", dict(source_sha256=pipeline_source,
                                                          model_evaluations=len(unique)))
        if group == "convergence":
            baseline = Path(config.get("suite", {}).get("stage_a_baseline", "algorithm/results/v2_stage_A"))
            if (baseline / "complete.json").exists():
                old = read_metrics_csv(baseline / "metrics.csv")
                joined = old.merge(unique, on=["condition_id", "seed", "model"], suffixes=("_200", "_extended"))
                if len(joined):
                    joined["r2_change"] = joined.test_r2_extended - joined.test_r2_200
                    joined["validation_loss_change"] = joined.validation_loss_extended - joined.validation_loss_200
                    joined.to_csv(destination / "budget_sensitivity.csv", index=False)
                    intervals = []
                    for (axis, model), block in joined.groupby(["axis_200", "model"]):
                        intervals.append(dict(axis=axis, model=model,
                            **bootstrap(block.r2_change.to_numpy(), config.get("bootstrap_repetitions", 2000))))
                    pd.DataFrame(intervals).to_csv(destination / "budget_sensitivity_paired.csv", index=False)


def execute(config, output, resume_from=None):
    jobs = build_jobs(config)
    workers = config.get("suite", {}).get("workers", 2)
    if not isinstance(workers, int) or not 1 <= workers <= max(1, os.cpu_count() or 1):
        raise ValueError("Invalid worker count.")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).parents[2] / "run_simulation_suite.py"
    manifest = output / "manifest.json"
    if resume_from is None and manifest.exists():
        resume_from=read_json(manifest).get("recovery_source")
    protocol = dict(config=config, source_sha256=source_hash(),
                    runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest())
    if resume_from:
        protocol["recovery_source"]=str(Path(resume_from).resolve())
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if manifest.exists() and read_json(manifest)["fingerprint"] != fingerprint:
        raise ValueError("Suite output belongs to different code/config; use a new directory.")
    lock = output / "suite.lock"
    clear_dead_lock(lock)
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    awake = False
    try:
        if os.name == "nt":
            # A process-scoped power request; no permanent power settings are changed.
            awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        if not manifest.exists():
            if resume_from:
                write_json(output/"status.json",dict(status="recovering_checkpoints",source=protocol["recovery_source"],updated_at=now()))
                import_checkpoint_cache(resume_from,output,jobs,config)
            write_json(manifest, dict(**protocol, fingerprint=fingerprint, created_at=now(),
                                     workload=plan_summary(config, jobs)))
            source = output / "source"
            source.mkdir(exist_ok=True)
            for path in Path(__file__).parent.glob("*.py"):
                shutil.copy2(path, source / path.name)
            shutil.copy2(runner, source / runner.name)
            shutil.copy2(Path(__file__).parents[2] / "run_mechanisms.py", source / "run_mechanisms.py")
        write_json(output / "queue_plan.json", [dict(id=j["id"], group=j["group"], priority=j["priority"],
            workload=j["workload"], condition=j["entries"][0]["condition"].to_dict(),
            references=[dict(stage=e["stage"],axis=e["axis"],value=e["value"]) for e in j["entries"]]) for j in jobs])
        completed, failures = set(), []
        pending = []
        for job in jobs:
            marker = output / "jobs" / job["id"] / "complete.json"
            if marker.exists() and read_json(marker).get("source_sha256") == protocol["source_sha256"]:
                completed.add(job["id"])
            else:
                pending.append(job)
        refresh_reports(jobs, output, completed, config)
        active = {}
        write_json(output / "status.json", progress(jobs, output, set(), completed, failures))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            position = 0
            while position < len(pending) or active:
                while position < len(pending) and len(active) < workers:
                    job = pending[position]
                    active[pool.submit(worker, job, str(output), protocol["source_sha256"])] = job
                    position += 1
                write_json(output / "status.json", progress(jobs, output,
                    {j["id"] for j in active.values()}, completed, failures))
                done, _ = wait(active, timeout=15, return_when=FIRST_COMPLETED)
                for future in done:
                    job = active.pop(future)
                    try:
                        result = future.result()
                    except Exception as error:
                        result = dict(job=job["id"], status="failed", error=str(error))
                    if result["status"] == "complete":
                        completed.add(job["id"])
                    else:
                        failures.append(result)
                    with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(event_record(result), ensure_ascii=False) + "\n")
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                if done:
                    refresh_reports(jobs, output, completed, config)
        status = progress(jobs, output, set(), completed, failures)
        status.update(status="failed" if failures else "complete", sleep_inhibition_requested=awake)
        write_json(output / "status.json", status)
        if not failures:
            write_json(output / "complete.json", dict(fingerprint=fingerprint,
                source_sha256=protocol["source_sha256"], model_evaluations=status["completed_model_evaluations"]))
        return status
    except BaseException as error:
        write_json(output / "status.json", dict(status="interrupted" if isinstance(error, KeyboardInterrupt)
            else "failed", error=str(error), updated_at=now()))
        raise
    finally:
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.unlink(missing_ok=True)
