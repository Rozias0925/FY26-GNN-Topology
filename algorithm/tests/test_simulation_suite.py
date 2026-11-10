"""Complete-study routing, fixed-topology equivalence, budgets and lock safety."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gnn_mechanisms.config import Condition
from gnn_mechanisms.data import generate
from gnn_mechanisms.experiment import run, workload, read_json
from gnn_mechanisms.suite import build_jobs, plan_summary, process_alive, clear_dead_lock
from gnn_mechanisms.training import prepare, train_model
import pandas as pd


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(purpose="smoke", generator=Condition(n_nodes=30,n_steps=20,degree=6).to_dict(),
            seeds=[7], training=dict(epochs=2,patience=10,widths=[8],learning_rates=[.003],threads=2),
            diagnostics=dict(folds=2,gate_epochs=1), bootstrap_repetitions=100,
            multi_axis_conditions=2,sample_sizes=[20,30],reuse_fixed_topology=True,
            suite=dict(workers=2,main_stages=["B","C","D","sample_size"],
                       robustness=True,convergence=True,convergence_epochs=4))

    def test_conditions_are_shared_without_merging_training_budgets(self):
        jobs=build_jobs(self.config)
        identifiers=[job["id"] for job in jobs]
        self.assertEqual(len(identifiers),len(set(identifiers)))
        main=[j for j in jobs if j["group"]=="main"]
        self.assertTrue(any(len(j["entries"])>1 for j in main))
        self.assertTrue(all(not j["config"]["gatv2"] for j in main))
        self.assertTrue(all(j["config"]["training"]["epochs"]==2 for j in main))
        extra=[j for j in jobs if j["group"]=="robustness"]
        self.assertTrue(all(j["config"]["gatv2"] for j in extra))
        self.assertEqual(len(extra),14)
        longer=[j for j in jobs if j["group"]=="convergence"]
        self.assertEqual(len(longer),6)
        self.assertTrue(all(j["config"]["training"]["epochs"]==4 for j in longer))
        self.assertEqual(self.config["training"]["epochs"],2)
        total=plan_summary(self.config,jobs)
        self.assertEqual(total["model_evaluations"],sum(j["workload"]["model_evaluations"] for j in jobs))

    def test_response_axes_are_interleaved_and_hypotheses_do_not_gate_jobs(self):
        jobs=build_jobs(self.config)
        axes={e["axis"] for j in jobs[:8] for e in j["entries"] if e["stage"]=="B"}
        self.assertEqual(axes,{"graph_signal","kappa","tau","delta","eta"})
        self.assertEqual(sum(j["entries"][0]["stage"]=="D" for j in jobs),2)

    def test_assigned_plan_and_exact_reuse_preserve_model_evaluation(self):
        c=Condition(**self.config["generator"])
        entry=dict(stage="A",axis="verification",value=None,condition=c)
        self.assertEqual(workload(self.config,["A"],[entry])["model_evaluations"],5)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            output=run(self.config,["A"],root,entries=[entry])
            self.assertEqual(output["model_evaluations"],5)
            folder=root/"datasets"/c.identifier/"seed_7"
            static=read_json(folder/"Static-GCN"/"result.json")
            dynamic=read_json(folder/"Dynamic-GCN"/"result.json")
            self.assertEqual(dynamic["candidate_fits_executed"],0)
            self.assertEqual(dynamic["training_reuse"]["source_model"],"Static-GCN")
            self.assertEqual(static["test"],dynamic["test"])
            # An independently fitted dynamic model must be bitwise equivalent.
            independent,_,weights,_=train_model("Dynamic-GCN",prepare(generate(c,7)),self.config["training"],7)
            self.assertEqual(independent["selection"],dynamic["selection"])
            saved=torch.load(folder/"Dynamic-GCN"/"weights.pt",weights_only=True)
            for key in weights:
                torch.testing.assert_close(weights[key],saved[key],rtol=0,atol=0)
            before=(folder/"MLP"/"weights.pt").stat().st_mtime_ns
            run(self.config,["A"],root,entries=[entry])
            self.assertEqual(before,(folder/"MLP"/"weights.pt").stat().st_mtime_ns)
            changed=dict(stage="A",axis="changed",value=None,condition=c)
            with self.assertRaisesRegex(ValueError,"different"):
                run(self.config,["A"],root,entries=[changed])

    def test_changing_topology_does_not_use_equivalence_reuse(self):
        c=Condition(n_nodes=30,n_steps=20,degree=6,delta=.2)
        entry=dict(stage="A",axis="drift",value=None,condition=c)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            run(self.config,["A"],root,entries=[entry])
            result=read_json(root/"datasets"/c.identifier/"seed_7"/"Dynamic-GCN"/"result.json")
            self.assertNotIn("training_reuse",result)
            self.assertEqual(result["candidate_fits_executed"],1)

    def test_shared_response_and_phase_references_export_without_duplicate_fits(self):
        c=Condition(**self.config["generator"])
        entries=[dict(stage="B",axis="graph_signal",value=.75,condition=c),
                 dict(stage="C",axis="signal_relevance",value={"graph_signal":.75,"kappa":0},condition=c)]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            result=run(self.config,["B","C"],root,entries=entries)
            self.assertEqual(result["model_evaluations"],5)
            self.assertEqual(result["reference_rows"],10)
            frame=pd.read_csv(root/"metrics.csv")
            self.assertEqual(len(frame.drop_duplicates(["condition_id","seed","model"])),5)
            payload=read_json(root/"website_results_v2.json")
            self.assertTrue(any(name.startswith("response_") for name in payload["figures"]))
            self.assertEqual(sum(name.startswith("phase_") for name in payload["figures"]),3)

    def test_epoch_ceiling_is_reported_using_validation_training_only(self):
        c=Condition(**self.config["generator"])
        result,_,_,candidates=train_model("MLP",prepare(generate(c,7)),self.config["training"],7)
        self.assertTrue(result["selection"]["budget_exhausted"])
        self.assertEqual(result["selection"]["stopped_reason"],"epoch_limit")
        self.assertEqual(candidates[0]["epochs_run"],2)

    def test_active_lock_is_preserved_and_only_dead_lock_is_removed(self):
        self.assertTrue(process_alive(os.getpid()))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"run.lock"
            path.write_text(str(os.getpid()),encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError,"Active"):
                clear_dead_lock(path)
            self.assertTrue(path.exists())
            with patch("gnn_mechanisms.suite.process_alive",return_value=False):
                clear_dead_lock(path)
            self.assertFalse(path.exists())

    def test_invalid_sensitivity_budget_is_rejected_before_execution(self):
        config=copy.deepcopy(self.config)
        config["suite"]["convergence_epochs"]=2
        with self.assertRaisesRegex(ValueError,"exceed"):
            build_jobs(config)


if __name__=="__main__":
    unittest.main()
