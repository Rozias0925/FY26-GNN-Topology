"""Mechanism invariants, leakage guards, sparse aggregation and paired inference."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from gnn_mechanisms.config import Condition,plans
from gnn_mechanisms.data import generate,time_split,smooth
from gnn_mechanisms.graphs import initial_graph,rewire,edge_turnover,consensus,source_weights,relevance_entropy
from gnn_mechanisms.models import SnapshotModel,pack_graphs,graph_tensors,MODEL_NAMES
from gnn_mechanisms.diagnostics import measure,diagnostic_target,smoothness,chronological_folds
from gnn_mechanisms.training import prepare,train_model
from gnn_mechanisms.reporting import bootstrap,paired
from gnn_mechanisms.real_validation import freeze,compare,file_hash


class MechanismTests(unittest.TestCase):
    def setUp(self):
        self.condition=Condition(n_nodes=30,n_steps=20,degree=6)

    def test_target_time_split_has_no_boundary_crossing(self):
        origins,targets,splits=time_split(40)
        self.assertEqual(targets[splits["train"]].tolist(),list(range(2,25)))
        self.assertEqual(targets[splits["validation"]].tolist(),list(range(25,33)))
        self.assertEqual(targets[splits["test"]].tolist(),list(range(33,41)))
        np.testing.assert_array_equal(targets,origins+1)
        _,contemporaneous,same=time_split(40,horizon=0)
        self.assertEqual(contemporaneous[same["train"]].tolist(),list(range(1,25)))

    def test_regular_normalization_equals_uniform_aggregation(self):
        graph=initial_graph(30,6,"regular",np.random.default_rng(7))
        np.testing.assert_array_equal(graph.sum(1),np.full(30,6))
        packed=pack_graphs(graph)
        np.testing.assert_allclose(packed["gcn_weights"],1/7,atol=1e-7)
        scores=np.random.default_rng(7).normal(size=30)
        weights=source_weights(graph,scores,0)
        np.testing.assert_allclose(weights.sum(1),1)
        np.testing.assert_allclose(weights[weights>0],1/7)
        np.testing.assert_allclose(relevance_entropy(weights),0,atol=1e-12)

    def test_entropy_sign_range_and_concentration(self):
        graph=initial_graph(30,6,"regular",np.random.default_rng(7))
        scores=np.random.default_rng(9).normal(size=30)
        values=[relevance_entropy(source_weights(graph,scores,k)).mean() for k in (0,.5,1,2,4)]
        self.assertTrue(all(0<=v<=1 for v in values))
        self.assertTrue(all(right>left for left,right in zip(values,values[1:])))

    def test_swaps_preserve_degrees_density_and_realized_turnover(self):
        graph=initial_graph(100,10,"regular",np.random.default_rng(7))
        changed=rewire(graph,.2,np.random.default_rng(9))
        np.testing.assert_array_equal(changed.sum(1),graph.sum(1))
        np.testing.assert_array_equal(changed,changed.T)
        self.assertFalse(np.diag(changed).any())
        self.assertEqual(changed.sum(),graph.sum())
        self.assertAlmostEqual(edge_turnover(graph,changed),.2)

    def test_noise_changes_only_observed_topology(self):
        clean=generate(self.condition,7)
        noisy=generate(replace(self.condition,eta=.4),7)
        np.testing.assert_array_equal(clean.features,noisy.features)
        np.testing.assert_array_equal(clean.target,noisy.target)
        np.testing.assert_array_equal(clean.true_graphs,noisy.true_graphs)
        self.assertFalse(np.array_equal(clean.observed_graphs,noisy.observed_graphs))
        np.testing.assert_array_equal(noisy.observed_graphs.sum(-1),noisy.true_graphs.sum(-1))
        self.assertEqual(noisy.static_graph.sum(),noisy.true_graphs[0].sum())

    def test_feature_only_is_invariant_to_graph_mechanisms(self):
        base=replace(self.condition,graph_signal=0)
        first=generate(base,19)
        second=generate(replace(base,kappa=4,delta=.4,eta=.4),19)
        np.testing.assert_array_equal(first.features,second.features)
        np.testing.assert_array_equal(first.target,second.target)

    def test_signal_calibration_uses_training_period(self):
        dataset=generate(replace(self.condition,kappa=4),7)
        self.assertAlmostEqual(dataset.oracle["standardized_u_variance"],1)
        self.assertAlmostEqual(dataset.oracle["standardized_v_variance"],1)
        expected=1+2*np.sqrt(.75*.25)*dataset.oracle["u_v_covariance"]
        self.assertAlmostEqual(dataset.oracle["latent_variance"],expected)
        self.assertTrue(dataset.oracle["self_loop_in_generator"])

    def test_consensus_preserves_edge_count_and_only_uses_supplied_snapshots(self):
        dataset=generate(replace(self.condition,delta=.4),7)
        count=int(self.condition.n_steps*.6)
        expected=consensus(dataset.observed_graphs[:count],np.random.default_rng(11))
        altered=dataset.observed_graphs.copy()
        altered[count:]=False
        actual=consensus(altered[:count],np.random.default_rng(11))
        np.testing.assert_array_equal(expected,actual)
        self.assertEqual(expected.sum(),dataset.observed_graphs[0].sum())
        self.assertFalse(np.diag(expected).any())

    def test_smoothness_resolvent_and_signal_to_noise(self):
        graph=initial_graph(30,6,"regular",np.random.default_rng(7))
        z=np.random.default_rng(11).normal(size=30)
        result=smooth(z,graph,2)
        laplacian=np.diag(graph.sum(1))-graph
        np.testing.assert_allclose((np.eye(30)+2*laplacian)@result,z,atol=1e-10)
        self.assertLess(result@laplacian@result,z@laplacian@z)
        dataset=generate(replace(self.condition,regime="smoothness",tau=2),7)
        self.assertAlmostEqual(dataset.latent[dataset.splits["train"]].var(),1,places=6)
        self.assertEqual(dataset.oracle["horizon"],0)

    def test_diagnostics_do_not_read_validation_or_test(self):
        first=generate(self.condition,7)
        second=copy.deepcopy(first)
        future=np.r_[second.splits["validation"],second.splits["test"]]
        second.features[future]=1000
        second.target[future]=-999
        second.observed_graphs[future]=False
        a=measure(first,dict(gate_epochs=3,folds=2))
        b=measure(second,dict(gate_epochs=3,folds=2))
        for key in ("S_X","S_G","H","R_hat","D_struct","D_pred"):
            self.assertAlmostEqual(a[key],b[key])
        self.assertEqual(a["training_target_times"],first.target_times[first.splits["train"]].tolist())
        for fit,holdout in chronological_folds(len(first.splits["train"])):
            self.assertLess(fit.max(),holdout.min())

    def test_sparse_gat_and_gatv2_are_permutation_equivariant(self):
        graph=initial_graph(30,6,"regular",np.random.default_rng(7))
        permutation=np.random.default_rng(11).permutation(30)
        x=torch.as_tensor(np.random.default_rng(13).normal(size=(2,30,8)),dtype=torch.float32)
        for name in ("Static-GCN","Static-GAT","Static-GATv2"):
            torch.manual_seed(7)
            model=SnapshotModel(name,width=12)
            first=model(x,graph_tensors(pack_graphs(graph))).detach().numpy()
            second=model(x[:,permutation],graph_tensors(pack_graphs(
                graph[permutation][:,permutation]))).detach().numpy()
            np.testing.assert_allclose(second,first[:,permutation],atol=1e-6)

    def test_static_and_dynamic_match_when_graph_is_constant(self):
        dataset=generate(self.condition,7)
        data=prepare(dataset)
        for architecture in ("GCN","GAT"):
            first=SnapshotModel("Static-"+architecture,width=12)
            second=SnapshotModel("Dynamic-"+architecture,width=12)
            second.load_state_dict(first.state_dict())
            a=first(data["x"],data["static"])
            b=second(data["x"],data["dynamic"])
            torch.testing.assert_close(a,b)

    def test_test_labels_do_not_affect_tuning_or_preprocessing(self):
        dataset=generate(self.condition,7)
        altered=copy.deepcopy(dataset)
        altered.target[altered.splits["test"]]+=1000
        first,second=prepare(dataset),prepare(altered)
        config=dict(epochs=3,patience=2,widths=[8],learning_rates=[.003,.01],threads=2)
        a,_,wa,_=train_model("MLP",first,config,7)
        b,_,wb,_=train_model("MLP",second,config,7)
        self.assertEqual(a["selection"],b["selection"])
        self.assertEqual(a["feature_mean"],b["feature_mean"])
        for key in wa:
            torch.testing.assert_close(wa[key],wb[key],rtol=0,atol=0)
        self.assertNotEqual(a["test"]["rmse"],b["test"]["rmse"])

    def test_classification_threshold_is_training_only(self):
        dataset=generate(replace(self.condition,task="classification"),7)
        labels=diagnostic_target(dataset)
        train=dataset.splits["train"]
        self.assertAlmostEqual(labels[train].mean(),.2,places=2)
        dataset.latent[dataset.splits["test"]]=1000
        np.testing.assert_array_equal(labels[train],diagnostic_target(dataset)[train])

    def test_design_has_one_axis_and_latin_hypercube_is_reproducible(self):
        config=dict(generator=self.condition.to_dict(),multi_axis_conditions=15,design_seed=17)
        self.assertEqual(len(plans(config,["A"])),6)
        self.assertEqual(len(plans(config,["B"])),28)
        self.assertEqual(len(plans(config,["C"])),56)
        first=plans(config,["D"])
        second=plans(config,["D"])
        self.assertEqual([item["condition"].to_dict() for item in first],
                         [item["condition"].to_dict() for item in second])
        self.assertEqual(len({item["condition"].identifier for item in first}),15)

    def test_bootstrap_and_factorial_interaction_are_paired(self):
        result=bootstrap([-.1,.2,.3],200,7)
        self.assertAlmostEqual(result["mean_gap"],.4/3)
        self.assertAlmostEqual(result["p_delta_positive"],2/3)
        scores=dict(zip(MODEL_NAMES,[.1,.2,.3,.4,.6]))
        rows=[]
        for seed in (7,19):
            for model,score in scores.items():
                rows.append(dict(stage="A",axis="example",value="null",condition_id="x",
                    task="regression",seed=seed,model=model,test_r2=score+seed/1000,
                    **{key:0. for key in ("S_X","S_G","S_G_shuffled","H","R_hat",
                         "R_true","R_estimation_error","D_struct","D_pred")}))
        output=paired(pd.DataFrame(rows),200)
        interaction=output[output.contrast=="Interaction"].iloc[0]
        self.assertAlmostEqual(interaction.mean_gap,(.6-.3)-(.4-.2))
        self.assertEqual(interaction.seeds,2)
        with tempfile.TemporaryDirectory() as directory:
            csv=Path(directory)/"metrics.csv"
            pd.DataFrame(rows).to_csv(csv,index=False)
            reloaded=pd.read_csv(csv,keep_default_na=False)
            for key in ("R_true","R_hat","test_r2"):
                reloaded[key]=reloaded[key].astype(str)
            again=paired(reloaded,200)
            self.assertAlmostEqual(again[again.contrast=="Interaction"].iloc[0].mean_gap,
                                   interaction.mean_gap)
            from gnn_mechanisms.reporting import export
            payload=export(reloaded,Path(directory),200,purpose="smoke")
            self.assertIsInstance(payload["model_results"][0]["test_r2"],float)

    def test_real_preregistration_rejects_future_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            panel=root/"panel.npz"
            metadata=dict(training_only=True,training_cutoff=10,dataset_id="example")
            np.savez(panel,features=np.zeros((12,3,8)),target=np.zeros((12,3)),
                graphs=np.zeros((12,3,3)),target_times=np.arange(12),
                metadata=np.array(json.dumps(metadata)))
            with self.assertRaisesRegex(ValueError,"after"):
                freeze(panel,root,root/"expectation.json",gate_epochs=1)

    def test_undefined_binary_scores_do_not_create_paired_inference(self):
        rows=[]
        for seed in (7,19):
            for name in MODEL_NAMES:
                rows.append(dict(stage="A",axis="one_class",value="null",condition_id="x",
                                 task="classification",seed=seed,model=name,test_ap=None))
        output=paired(pd.DataFrame(rows),200)
        self.assertTrue(output.empty)
        self.assertIn("contrast",output.columns)

    def test_real_expectation_is_frozen_before_all_five_heldout_scores(self):
        from gnn_mechanisms.real_validation import OBSERVABLE
        from gnn_mechanisms.reporting import CONTRASTS
        dataset=generate(self.condition,7)
        train=dataset.splits["train"]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            metadata=dict(training_only=True,training_cutoff=12,dataset_id="fixture",
                          target_alignment="X_t -> Y_t+1")
            np.savez(root/"training.npz",features=dataset.features[train],
                target=dataset.target[train],graphs=dataset.observed_graphs[train],
                target_times=dataset.target_times[train],metadata=np.array(json.dumps(metadata)))
            (root/"complete.json").write_text("{}")
            (root/"manifest.json").write_text(json.dumps(dict(config=dict(purpose="research"))))
            pd.DataFrame([dict(task="regression",regime="mechanism",attention="source",
                n_nodes=30,condition_id="fixture",seed=7,**{key:.1 for key in OBSERVABLE})
            ]).to_csv(root/"metrics.csv",index=False)
            pd.DataFrame([dict(condition_id="fixture",task="regression",contrast=name,mean_gap=.1)
                          for name in CONTRASTS]).to_csv(root/"paired_contrasts.csv",index=False)
            destination=root/"expectation.json"
            expected=freeze(root/"training.npz",root,destination,gate_epochs=1)
            self.assertEqual(expected["dataset_id"],"fixture")
            with self.assertRaises(FileExistsError):
                freeze(root/"training.npz",root,destination,gate_epochs=1)
            heldout=dict(dataset_id="fixture",expectation_sha256=file_hash(destination),
                evaluated_at=datetime.now(timezone.utc).isoformat(),
                models={name:dict(test_r2=.2+i*.1) for i,name in enumerate(MODEL_NAMES)})
            (root/"heldout.json").write_text(json.dumps(heldout))
            report=compare(destination,root/"heldout.json",root/"validation.json")
            self.assertEqual(len(report["comparisons"]),9)
            del heldout["models"]["MLP"]
            (root/"heldout.json").write_text(json.dumps(heldout))
            with self.assertRaisesRegex(ValueError,"five"):
                compare(destination,root/"heldout.json",root/"invalid.json")
            (root/"manifest.json").write_text(json.dumps(dict(config=dict(purpose="smoke"))))
            with self.assertRaisesRegex(ValueError,"Smoke"):
                freeze(root/"training.npz",root,root/"smoke_expectation.json",gate_epochs=1)


if __name__=="__main__":
    unittest.main()
