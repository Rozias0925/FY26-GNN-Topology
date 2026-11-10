"""These tiny synthetic checks must pass before the measured-data training queue."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from low_impact_runtime import set_thread_environment, ResourceGuard, RunLock, StopRequested
set_thread_environment()

import numpy as np
import pandas as pd
import torch
from gnn_graph_selection.protocol import Panel
from gnn_graph_selection.measured_benchmark import (
    prepare_regression, fit_candidate, predict_scaled, load_audited,
    moving_block_indices, file_sha, configure_cpu,
    validate_hourly_index,
    validate_config,
    run,
)
from gnn_mechanisms.models import SnapshotModel, MODEL_NAMES, pack_graphs, graph_tensors


CFG = dict(stations=['NW12', 'NX08', 'GP17'], target_station='NX08', history_length=4,
           graph_window=8, graph_stride=2, top_k=1, max_lag=1, ffill_limit=3,
           train_fraction=.6, validation_fraction=.2, threshold=1000,
           min_split_samples=6, horizons=[1, 3, 6], allow_partial_archive=False,
           include_reservoir_covariates=False, start='2021-01-01', end='2021-01-05 04:00')


def fixture():
    total, nodes = 100, 3
    rng = np.random.default_rng(42)
    y = 20 + np.arange(total)[:, None] / 10 + rng.normal(size=(total, nodes))
    x = np.stack([np.log1p(y), np.log1p(y + 1)], -1).astype(np.float32)
    dates = pd.date_range('2021-01-01', periods=total, freq='h', tz='UTC').astype(str).to_numpy()
    panel = Panel(x, y, np.log1p(y), CFG['stations'], [1], dates, {}, event_response=y + 1)
    static = np.ones((nodes, nodes), np.float32) - np.eye(nodes, dtype=np.float32)
    dynamic = np.broadcast_to(static, (total, nodes, nodes)).copy()
    return panel, static, dynamic


def tiny_config():
    return dict(purpose='exploratory_real_regression', analysis_directory='audit', seeds=[7], horizons=[1],
                runtime=dict(device='cpu', threads=1, workers=1, priority='Idle', logical_cpus=1),
                training=dict(epochs=1, patience=1, batch_size=32, widths=[4], learning_rates=[.001],
                              bootstrap_repetitions=3, bootstrap_block_hours=4, log_response=True))


class PreparationTests(unittest.TestCase):
    def test_hourly_check_accepts_pandas_time_resolutions(self):
        base = pd.date_range('2021-01-01', periods=10, freq='h', tz='UTC')
        for unit in ['s', 'us', 'ns']:
            self.assertEqual(len(validate_hourly_index(base.as_unit(unit), 10)), 10)
        with self.assertRaisesRegex(ValueError, 'hourly index'):
            validate_hourly_index(base.delete(2), 9)

    def test_regression_is_independent_of_event_class_coverage(self):
        panel, static, dynamic = fixture()
        panel.response[11, 1] = np.nan
        panel.event_response[11, 1] = np.nan
        data = prepare_regression(panel, CFG, 3, static, dynamic)
        ix = np.flatnonzero(data['times'] == 10)[0]
        self.assertEqual(data['target'][ix], panel.response[13, 1])
        self.assertTrue(np.isnan(data['labels'].set_index('issue_index').loc[10, 'event_target']))
        self.assertEqual(data['x'].shape[1:], (3, 16))
        self.assertEqual(data['target_node'], 1)

    def test_purged_split_and_training_scalers_ignore_test_changes(self):
        a, static, dynamic = fixture()
        b, _, _ = fixture()
        b.features[80:] *= 200
        b.response[80:] *= 200
        original = prepare_regression(a, CFG, 6, static, dynamic)
        changed = prepare_regression(b, CFG, 6, static, dynamic)
        train = original['splits']['train']
        self.assertLess(int((original['times'][train] + 6).max()), 60)
        self.assertEqual(original['target_mean'], changed['target_mean'])
        self.assertEqual(original['target_scale'], changed['target_scale'])
        np.testing.assert_array_equal(original['x'][train], changed['x'][train])

    def test_audited_input_detects_tampering_and_node_order_mismatch(self):
        panel, static, dynamic = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            np.savez(folder / 'hourly_panel.npz', features=panel.features, response=panel.response,
                     graph_series=panel.graph_series, timestamps=panel.timestamps.astype(str), event_response=panel.event_response)
            np.savez(folder / 'causal_graphs.npz', static=static, dynamic=dynamic,
                     timestamps=panel.timestamps.astype(str), stations=panel.node_names)
            (folder / 'analysis_config.json').write_text(json.dumps({'data': CFG, 'protocol': {}}))
            (folder / 'panel_metadata.json').write_text('{}')
            manifest = dict(completed=True, outputs_sha256={p.name: file_sha(p) for p in folder.iterdir()})
            marker = folder / 'analysis_manifest.json'
            marker.write_text(json.dumps(manifest))
            result = load_audited(folder)
            self.assertEqual(result[0].target_nodes, [1])
            (folder / 'panel_metadata.json').write_text('{"unexpected":true}')
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                load_audited(folder)
            manifest['outputs_sha256']['panel_metadata.json'] = file_sha(folder / 'panel_metadata.json')
            np.savez(folder / 'causal_graphs.npz', static=static, dynamic=dynamic,
                     timestamps=panel.timestamps.astype(str), stations=list(reversed(panel.node_names)))
            manifest['outputs_sha256']['causal_graphs.npz'] = file_sha(folder / 'causal_graphs.npz')
            marker.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'node order'):
                load_audited(folder)


class ModelChecks(unittest.TestCase):
    def setUp(self):
        configure_cpu()

    def test_all_five_models_fit_scalar_target_without_using_test_inputs(self):
        panel, static, dynamic = fixture()
        training = dict(epochs=2, patience=2, batch_size=32, min_delta=0)
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                data = prepare_regression(panel, CFG, 3, static, dynamic)
                data['x'][data['splits']['test']] = np.nan
                data['scaled_target'][data['splits']['test']] = np.nan
                # Poisoned test data must not be touched during fitting or selection.
                model, meta = fit_candidate(name, data, training, 4, .001, 7)
                prediction = predict_scaled(model, data, data['splits']['validation'])
                self.assertEqual(prediction.shape, (len(data['splits']['validation']),))
                self.assertTrue(np.isfinite(prediction).all())
                self.assertTrue(np.isfinite(meta['validation_mse']))
                self.assertEqual(torch.get_num_threads(), 1)
                self.assertEqual(next(model.parameters()).device.type, 'cpu')

    def test_mlp_cannot_read_other_nodes_and_gnn_can(self):
        torch.manual_seed(7)
        graph = graph_tensors(pack_graphs(np.ones((3, 3)) - np.eye(3)))
        for name in ['MLP', 'Static-GCN']:
            x = torch.randn(5, 3, 16, requires_grad=True)
            model = SnapshotModel(name, input_dim=16, width=8)
            model(x, graph)[:, 1].sum().backward()
            other = x.grad[:, [0, 2]].abs().sum().item()
            if name == 'MLP':
                self.assertEqual(other, 0.)
            else:
                self.assertGreater(other, 0.)

    def test_seed_repeats_are_deterministic(self):
        panel, static, dynamic = fixture()
        data = prepare_regression(panel, CFG, 1, static, dynamic)
        cfg = dict(epochs=2, patience=2, batch_size=32, min_delta=0)
        a, ma = fit_candidate('Dynamic-GAT', data, cfg, 4, .001, 7)
        b, mb = fit_candidate('Dynamic-GAT', data, cfg, 4, .001, 7)
        self.assertEqual(ma['validation_mse'], mb['validation_mse'])
        np.testing.assert_array_equal(predict_scaled(a, data, data['splits']['validation']),
                                      predict_scaled(b, data, data['splits']['validation']))


class SafetyChecks(unittest.TestCase):
    def test_invalid_budget_or_duplicate_horizon_cannot_start(self):
        original = tiny_config()
        validate_config(original)
        bad = json.loads(json.dumps(original))
        bad['training']['bootstrap_repetitions'] = 0
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            validate_config(bad)
        bad = json.loads(json.dumps(original))
        bad['horizons'] = [1, 1]
        with self.assertRaisesRegex(ValueError, 'distinct'):
            validate_config(bad)

    def test_bootstrap_blocks_stop_at_missing_time_gaps(self):
        fixed = Mock()
        fixed.integers.return_value = 0
        with patch('gnn_graph_selection.measured_benchmark.np.random.default_rng', return_value=fixed):
            rows = list(moving_block_indices(np.array([1, 2, 100, 101]), 1, 24))
        np.testing.assert_array_equal(rows[0], [0, 1, 0, 1])

    def test_resource_pressure_pauses_before_further_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            guard = object.__new__(ResourceGuard)
            guard.output, guard.last_poll, guard.interval = Path(temporary), 0., 1.
            guard.callback = Mock()
            guard.sample = Mock(side_effect=[dict(cpu_percent=10., available_ram_gib=.9),
                                             dict(cpu_percent=20., available_ram_gib=1.4)])
            with patch('low_impact_runtime.time.sleep'):
                guard.check(force=True)
            self.assertEqual([call.args[0] for call in guard.callback.call_args_list], ['waiting_for_resources', 'running'])
            for available in [[.7], [.9, .7]]:
                with self.subTest(critical_ram_samples=available):
                    guard.last_poll = 0.
                    guard.callback = Mock()
                    guard.sample = Mock(side_effect=[dict(cpu_percent=10., available_ram_gib=value)
                                                     for value in available])
                    with patch('low_impact_runtime.time.sleep'), self.assertRaises(StopRequested):
                        guard.check(force=True)
                    self.assertEqual(guard.sample.call_count, len(available))

    @unittest.skipUnless(sys.platform == 'win32', 'Windows process identity lock')
    def test_same_output_cannot_be_owned_twice(self):
        with tempfile.TemporaryDirectory() as temporary:
            with RunLock(Path(temporary)):
                with self.assertRaisesRegex(RuntimeError, 'active PID'):
                    with RunLock(Path(temporary)):
                        pass
            self.assertFalse((Path(temporary) / '.runner.lock').exists())


class EndToEndChecks(unittest.TestCase):
    def test_tiny_study_scores_all_models_and_resume_does_not_retrain(self):
        configure_cpu()
        panel, static, dynamic = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit, output = root / 'audit', root / 'output'
            audit.mkdir()
            output.mkdir()
            np.savez(audit / 'hourly_panel.npz', features=panel.features, response=panel.response,
                     graph_series=panel.graph_series, timestamps=panel.timestamps.astype(str), event_response=panel.event_response)
            np.savez(audit / 'causal_graphs.npz', static=static, dynamic=dynamic,
                     timestamps=panel.timestamps.astype(str), stations=panel.node_names)
            (audit / 'analysis_config.json').write_text(json.dumps({'data': CFG, 'protocol': {}}))
            (audit / 'panel_metadata.json').write_text('{}')
            manifest = dict(completed=True, raw_source_fingerprints={'turbidity': 'synthetic-test-fixture'},
                            outputs_sha256={p.name: file_sha(p) for p in audit.iterdir()})
            (audit / 'analysis_manifest.json').write_text(json.dumps(manifest))
            config = tiny_config()
            result = run(config, root, output, {}, Mock(), Mock())
            self.assertTrue(result['regression_complete'])
            self.assertEqual(result['jobs'], 5)
            table = pd.read_csv(output / 'metrics.csv')
            self.assertEqual(set(table.model), set(MODEL_NAMES))
            self.assertTrue(np.isfinite(table.mae).all())
            self.assertTrue((output / 'paired_contrasts.json').exists())
            self.assertTrue((output / 'source_snapshot/tests/test_real_benchmark.py').exists())
            with patch('gnn_graph_selection.measured_benchmark.fit_candidate', side_effect=AssertionError('must not retrain')):
                resumed = run(config, root, output, {}, Mock(), Mock())
            self.assertEqual(resumed['plan_identity'], result['plan_identity'])


if __name__ == '__main__':
    unittest.main()
