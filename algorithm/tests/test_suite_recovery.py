"""Regression checks for failed-event logging, file contention and safe reuse."""
from concurrent.futures import Future
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from gnn_mechanisms import experiment
from gnn_mechanisms.config import Condition
from gnn_mechanisms.suite import event_record,build_jobs,execute,import_checkpoint_cache


class RecoveryTests(unittest.TestCase):
    def test_failed_worker_timestamp_is_serializable_without_duplicate_keywords(self):
        failure=dict(job="example",status="failed",error="sharing violation",updated_at="worker-time")
        record=event_record(failure)
        self.assertEqual(record["updated_at"],"worker-time")
        self.assertIn("recorded_at",record)
        self.assertNotIn("recorded_at",failure)
        self.assertEqual(json.loads(json.dumps(record))["error"],failure["error"])

    @staticmethod
    def sharing_error():
        error=PermissionError("temporary Windows access/sharing failure")
        error.winerror=5
        return error

    def test_atomic_json_retries_transient_replace_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"status.json"
            experiment.write_json(path,dict(value="old"))
            original=os.replace
            attempts=[]
            def replace(source,target):
                attempts.append(str(source))
                if len(attempts)<3:
                    self.assertEqual(json.loads(path.read_text())["value"],"old")
                    raise self.sharing_error()
                return original(source,target)
            with patch("gnn_mechanisms.experiment.os.replace",side_effect=replace), \
                 patch("gnn_mechanisms.experiment.time.sleep"):
                experiment.write_json(path,dict(value="new"))
            self.assertEqual(len(attempts),3)
            self.assertEqual(experiment.read_json(path)["value"],"new")
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    def test_persistent_contention_preserves_previous_json_and_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"status.json"
            experiment.write_json(path,dict(value="old"))
            with patch("gnn_mechanisms.experiment.os.replace",side_effect=self.sharing_error()), \
                 patch("gnn_mechanisms.experiment.time.sleep"):
                with self.assertRaises(PermissionError):
                    experiment.write_json(path,dict(value="new"))
            self.assertEqual(experiment.read_json(path)["value"],"old")

    def test_json_reader_retries_sharing_failures(self):
        with patch("gnn_mechanisms.experiment.Path.read_text",side_effect=[self.sharing_error(),'{"value":1}']), \
             patch("gnn_mechanisms.experiment.time.sleep"):
            self.assertEqual(experiment.read_json("example.json"),dict(value=1))

    def test_one_failed_job_does_not_abort_other_jobs_or_fake_completion(self):
        config=dict(purpose="smoke",generator=Condition(n_nodes=30,n_steps=20,degree=6).to_dict(),
            seeds=[7],training=dict(epochs=2,widths=[8],learning_rates=[.003]),
            multi_axis_conditions=2,suite=dict(workers=1,main_stages=["D"],convergence=False,robustness=False))
        class ImmediatePool:
            def __init__(self,**kwargs): self.number=0
            def __enter__(self): return self
            def __exit__(self,*args): return False
            def submit(self,function,job,*args):
                self.number+=1
                future=Future()
                if self.number==1:
                    future.set_result(dict(job=job["id"],status="failed",error="example failure",updated_at="worker-time"))
                else:
                    future.set_result(dict(job=job["id"],status="complete"))
                return future
        with tempfile.TemporaryDirectory() as directory, \
             patch("gnn_mechanisms.suite.ProcessPoolExecutor",ImmediatePool), \
             patch("gnn_mechanisms.suite.refresh_reports"):
            root=Path(directory)
            status=execute(config,root)
            self.assertEqual(status["status"],"failed")
            self.assertEqual(status["completed_jobs"],1)
            self.assertEqual(len(status["failures"]),1)
            self.assertEqual(len((root/"events.jsonl").read_text().splitlines()),2)
            self.assertFalse((root/"complete.json").exists())

    def test_checkpoint_import_preserves_weights_and_rejects_model_changes(self):
        config=dict(purpose="smoke",generator=Condition(n_nodes=30,n_steps=20,degree=6).to_dict(),
            seeds=[7],training=dict(epochs=2,patience=10,widths=[8],learning_rates=[.003],threads=2),
            diagnostics=dict(folds=2,gate_epochs=1),bootstrap_repetitions=100,reuse_fixed_topology=True,
            multi_axis_conditions=1,suite=dict(workers=1,main_stages=["D"],convergence=False,robustness=False))
        jobs=build_jobs(config)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old=root/"old";new=root/"new"
            (old/"source").mkdir(parents=True)
            for source in Path(experiment.__file__).parent.glob("*.py"):
                shutil.copy2(source,old/"source"/source.name)
            experiment.write_json(old/"manifest.json",dict(config=config,source_sha256=experiment.source_hash()))
            job=jobs[0]
            original=old/"jobs"/job["id"]
            experiment.run(job["config"],job["stages"],original,entries=job["entries"])
            before={str(p.relative_to(old)):p.read_bytes() for p in old.glob("jobs/*/*/datasets/*/*/*/weights.pt")}
            imported=import_checkpoint_cache(old,new,jobs,config)
            self.assertEqual(imported["imported_model_evaluations"],5)
            for relative,weights in before.items():
                self.assertEqual((new/relative).read_bytes(),weights)
                self.assertEqual((old/relative).read_bytes(),weights)
            model=old/"source/models.py"
            model.write_text(model.read_text(encoding="utf-8")+"\n",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"models.py changed"):
                import_checkpoint_cache(old,root/"unsafe",jobs,config)
