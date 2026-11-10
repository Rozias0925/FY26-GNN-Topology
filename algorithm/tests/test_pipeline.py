from __future__ import annotations
import copy
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gnn_graph_selection.config import SimulationConfig
from gnn_graph_selection.simulation import simulate
from gnn_graph_selection.protocol import Panel, causal_fill, time_splits, graph_inputs, prepare
from gnn_graph_selection.models import ForecastModel, MODEL_NAMES, GAT, GCN
from gnn_graph_selection.evaluation import classification, choose_threshold, extract_events, event_metrics
from gnn_graph_selection.real_data import hourly_station
from gnn_graph_selection.selection import fit_selection, apply_rule
from gnn_graph_selection.descriptors import FEATURE_NAMES
from gnn_graph_selection.sources import download, read_archive, archive_fingerprint
from gnn_graph_selection.csv_data import prepare_csv
from gnn_graph_selection.training import baselines
from gnn_graph_selection.estimators import lagged_partial_relevance, _ridge_residual
from gnn_graph_selection.reporting import event_figures


class LeakageTests(unittest.TestCase):
    def test_vectorized_relevance_matches_scalar_definition(self):
        y=np.random.default_rng(12).normal(size=(80,4))
        y[:,1]=.7*np.roll(y[:,0],1)+.3*y[:,1]
        expected=np.zeros((4,4))
        for i in range(4):
            controls=np.column_stack([y[3-k:80-k,i] for k in range(1,4)])
            target=_ridge_residual(y[3:,i],controls,1e-6)
            for j in range(4):
                if i==j:
                    continue
                values=[]
                for k in range(1,4):
                    residual=_ridge_residual(y[3-k:80-k,j],controls,1e-6)
                    values.append(target@residual/(np.linalg.norm(target)*np.linalg.norm(residual)))
                expected[i,j]=values[int(np.abs(values).argmax())]
        np.testing.assert_allclose(lagged_partial_relevance(y,max_lag=3),expected,atol=1e-12)

    def test_graph_estimates_do_not_see_future(self):
        rng = np.random.default_rng(10)
        y = rng.normal(size=(150, 3))
        panel = Panel(y[...,None], y, y.copy(), ['a','b','c'], [0], np.arange(150), {})
        cfg = {'graph_window':30, 'graph_stride':5, 'top_k':1, 'max_lag':2}
        a = graph_inputs(panel, cfg)
        panel.graph_series[80:] = rng.normal(100,30,size=(70,3))
        b = graph_inputs(panel, cfg)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1][:80], b[1][:80])

    def test_single_class_test_is_reported_without_changing_split(self):
        y = np.random.default_rng(99).normal(size=(200,3))
        y[160:] = -10
        panel = Panel(y[...,None], y, y, ['a','b','c'], [0], np.arange(200), {})
        cfg = {'history_length':6, 'graph_window':20, 'threshold':0}
        data = prepare(panel,cfg,3,(np.eye(3),np.tile(np.eye(3),(200,1,1))))
        self.assertTrue(data['metadata']['single_class_test'])
        self.assertEqual(data['metadata']['validation_end'],160)

    def test_purged_horizon(self):
        t = np.arange(100)
        s = time_splits(t, 6, 60, 80)
        self.assertTrue(np.all(t[s['train']] + 6 < 60))
        self.assertTrue(np.all((t[s['validation']] >= 60) & (t[s['validation']] + 6 < 80)))
        self.assertTrue(np.all(t[s['test']] >= 80))
        self.assertNotIn(59, t[s['train']])

    def test_no_backfill_or_test_fitted_scaler(self):
        x = np.array([[1., np.nan], [np.nan, 10.], [3., 20.], [4., np.nan], [5., 50.]])
        y = x.copy()
        y[4] = 100000
        a, mask, stats = causal_fill(x, 3, limit=1)
        b, _, other = causal_fill(y, 3, limit=1)
        np.testing.assert_array_equal(a[:4], b[:4])
        np.testing.assert_array_equal(stats['mean'], other['mean'])
        self.assertEqual(mask[0, 1], 0)
        self.assertEqual(stats['median'][1], 15.)

    def test_missing_future_never_becomes_negative(self):
        rng = np.random.default_rng(42)
        y = rng.normal(size=(200, 3))
        y[70, 0] = np.nan
        panel = Panel(y[..., None], y, y, ['a','b','c'], [0], np.arange(200), {})
        cfg = {'history_length':6, 'graph_window':20, 'horizons':[3], 'threshold':0., 'min_split_samples':4}
        graphs = (np.ones((3,3)) - np.eye(3), np.tile(np.ones((3,3))-np.eye(3), (200,1,1)))
        data = prepare(panel, cfg, 3, graphs)
        self.assertTrue(all(t not in data['times'] for t in [67, 68, 69]))
        self.assertIn(70, data['times'])

    def test_noise_changes_do_not_change_latent_process(self):
        cfg = SimulationConfig(n_steps=40, n_nodes=8, degree=2, seed=1, burn_in=10)
        a, b = simulate(cfg), simulate(replace(cfg, edge_noise=.8))
        np.testing.assert_array_equal(a.response, b.response)
        np.testing.assert_array_equal(a.true_adjacency, b.true_adjacency)
        c = simulate(replace(cfg, graph_drift=.8))
        np.testing.assert_array_equal(a.node_features, c.node_features)
        self.assertTrue(np.all(a.realized_edge_change <= 1))


class ModelTests(unittest.TestCase):
    def test_equivariance_and_gradients_for_all_models(self):
        torch.manual_seed(14)
        x = torch.randn(4, 5, 3, 2)
        a = torch.tensor([[0.,1,0],[1,0,1],[0,1,0]])
        perm = torch.tensor([2,0,1])
        for name in MODEL_NAMES:
            model = ForecastModel(name, 5, 2, 8)
            original = model(x, a)
            reordered = model(x[:,:,perm], a[perm][:,perm])
            torch.testing.assert_close(reordered[0], original[0][:,perm])
            loss = sum(v.square().mean() for v in original)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_mlp_cannot_read_neighbor_features(self):
        x = torch.randn(2,4,3,2)
        model = ForecastModel('MLP',4,2,8)
        a = torch.ones(3,3)
        first = model(x,a)[0][:,0]
        x[:,:,1:] = 10000
        torch.testing.assert_close(first, model(x,a)[0][:,0])

    def test_attention_mask_self_loops_and_normalization(self):
        layer = GAT(4)
        a = torch.zeros(3,3)
        _, weights = layer.attention(torch.randn(2,3,4), a)
        torch.testing.assert_close(weights, torch.eye(3).expand(2,3,3))
        a[0,1] = 1
        _, weights = layer.attention(torch.randn(2,3,4), a)
        torch.testing.assert_close(weights.sum(-1), torch.ones(2,3))
        self.assertTrue(torch.all(weights[:,0,2] == 0))

    def test_gcn_numerical_reference(self):
        layer = GCN(1)
        with torch.no_grad():
            layer.linear.weight.fill_(1)
            layer.bias.zero_()
        result = layer(torch.tensor([[[2.],[4.]]]), torch.tensor([[0.,1.],[1.,0.]]))
        torch.testing.assert_close(result, torch.tensor([[[3.],[3.]]]))


class EventTests(unittest.TestCase):
    def test_real_event_artifacts_include_actual_timestamps(self):
        y=np.zeros((35,1))
        y[12:15]=8
        dates=pd.date_range('2021-01-01',periods=35,freq='1h',tz='Asia/Taipei').astype(str).to_numpy()
        panel=Panel(y[...,None],y,y,['target'],[0],dates,{'kind':'real','step_hours':1},event_response=y)
        times=np.arange(1,31)
        p=np.zeros((30,1))
        p[times==10]=.9
        event=event_metrics(y[:,0],times,p[:,0],5,.5,3)
        event['node']='target'
        result={'events':[event],'model':'MLP','alarm_threshold':.5}
        data={'times':times,'splits':{'test':np.arange(len(times))},'target_nodes':[0],'threshold':5,'horizon':3}
        with tempfile.TemporaryDirectory() as folder:
            event_figures(panel,data,p,result,folder)
            saved=json.loads((Path(folder)/'events_node_0.json').read_text('utf-8'))
            self.assertEqual(saved['events'][0]['onset_timestamp'],str(dates[12]))
            self.assertEqual(saved['events'][0]['lead_time_steps'],2)
            self.assertTrue((Path(folder)/'event_0_1.png').exists())

    def test_missingness_breaks_events(self):
        y = [0,4,4,np.nan,4,0,4,4,0]
        self.assertEqual(extract_events(y,3,merge_gap=1), [(1,2),(4,7)])

    def test_early_hit_false_alarm_and_duplicate_are_separate(self):
        y = np.zeros(30)
        y[10:13] = 10
        t = np.arange(1,27)
        p = np.zeros_like(t, float)
        p[np.isin(t,[5,7,8,10])] = .9
        r = event_metrics(y,t,p,5,.5,3)
        self.assertEqual(r['detected'], 1)
        self.assertEqual(r['median_lead_steps'], 3.)
        self.assertEqual(r['duplicate_alarms'], 1)
        self.assertEqual(r['event_free_alarms'], 1)
        self.assertNotIn(10, r['alarm_times'])

    def test_no_events_or_single_class_not_fake_perfect(self):
        r = event_metrics(np.zeros(25), np.arange(1,20), np.zeros(19), 5,.5,3)
        self.assertIsNone(r['detection_rate'])
        self.assertIsNone(classification([0,0], [.1,.1])['auroc'])

    def test_window_metrics_fpr_and_false_alarm_ratio(self):
        r = classification([1,0,0,0],[.9,.8,.1,.1], .5)
        self.assertAlmostEqual(r['fpr'], 1/3)
        self.assertEqual(r['false_alarm_ratio'], .5)

    def test_missing_onset_history_is_censored(self):
        y = np.zeros(30)
        y[9] = np.nan
        y[10:12] = 8
        r = event_metrics(y,np.arange(1,26),np.ones(25),5,.5,3)
        self.assertEqual(r['n_events'], 0)
        self.assertEqual(r['censored_events'], 1)


class DataTests(unittest.TestCase):
    def test_empty_hours_and_long_gap_do_not_crash_or_extend_flat_run(self):
        left = pd.date_range('2021-01-01 00:05',periods=12,freq='5min',tz='UTC')
        right = left + pd.Timedelta(days=2)
        frame = pd.DataFrame({'time':left.append(right), 'value':np.ones(24)})
        result = hourly_station(frame)
        self.assertEqual(result.iloc[-1]['last'],1.)
        self.assertTrue(np.isnan(result.iloc[1]['last']))

    def test_persistence_event_uses_hourly_peak(self):
        response = np.zeros((10,1))
        peak = response.copy()
        peak[5] = 8
        panel = Panel(response[...,None],response,response,['a'],[0],np.arange(10),{},event_response=peak)
        data = {'times':np.array([1,5]),'splits':{'train':np.array([0]),'test':np.array([1])},
                'target_nodes':[0], 'event':np.array([[0.],[1.]]), 'regression':np.zeros((2,1)), 'threshold':5.}
        self.assertEqual(baselines(data,panel)['persistence']['tp'],1)

    def test_csv_release_time_prevents_early_feature_access(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'observations.csv'
            pd.DataFrame([{'timestamp':'2021-01-01 01:00','available_at':'2021-01-01 03:00',
                           'station_id':s,'variable':'turbidity','value':9,'unit':'NTU','quality':'good'} for s in ['a','b']]).to_csv(path,index=False)
            cfg = {'observations_csv':str(path),'stations':['a','b'],'target_station':'a','start':'2021-01-01','end':'2021-01-02',
                   'graph_variable_by_station':{'a':'turbidity','b':'turbidity'}}
            panel = prepare_csv(cfg)
            self.assertTrue(np.isnan(panel.features[1,0,0]))
            self.assertEqual(panel.features[3,0,0],9)
            self.assertEqual(panel.response[1,0],9)

    def test_hour_end_is_causal_and_labels_not_filled(self):
        times = pd.date_range('2021-01-01',periods=25,freq='5min',tz='UTC')
        frame = pd.DataFrame({'time':times, 'value':np.arange(25,dtype=float)})
        a = hourly_station(frame)
        frame.loc[frame.time > times[12], 'value'] = 999
        b = hourly_station(frame)
        self.assertEqual(a.loc[times[12],'last'],12)
        self.assertEqual(a.loc[times[12],'max'],12)
        self.assertEqual(a.loc[times[12],'last'],b.loc[times[12],'last'])
        self.assertTrue(np.isnan(a.iloc[0]['last']))

    def test_conflicting_duplicate_is_not_arbitrarily_selected(self):
        times = pd.date_range('2021-01-01 00:05',periods=12,freq='5min',tz='UTC')
        frame = pd.DataFrame({'time':times, 'value':np.ones(12)})
        frame = pd.concat([frame,pd.DataFrame({'time':[times[-1]], 'value':[999]})])
        result = hourly_station(frame, minimum_per_hour=12)
        self.assertTrue(np.isnan(result.iloc[0]['last']))

    def test_downloader_detects_ignored_pagination(self):
        data = json.dumps([{'x':1}]).encode()
        with tempfile.TemporaryDirectory() as folder, patch('gnn_graph_selection.sources.fetch_bytes', return_value=data), patch('gnn_graph_selection.sources.time.sleep'):
            with self.assertRaisesRegex(ValueError,'Repeated page'):
                download('turbidity',folder,max_pages=3,page_size=1)
            manifest = json.loads((Path(folder)/'turbidity/manifest.json').read_text('utf-8'))
            self.assertFalse(manifest['download_complete'])

    def test_source_archive_checksum_and_resume(self):
        with tempfile.TemporaryDirectory() as folder, patch('gnn_graph_selection.sources.fetch_bytes', side_effect=[b'[{"x":1}]',b'[]']) as fetch, patch('gnn_graph_selection.sources.time.sleep'):
            r = download('turbidity',folder,max_pages=3,page_size=1)
            self.assertTrue(r['download_complete'])
            self.assertEqual(list(read_archive(Path(folder)/'turbidity')), [{'x':1}])
            before=archive_fingerprint(Path(folder)/'turbidity')
            download('turbidity',folder,max_pages=3,page_size=1)
            self.assertEqual(fetch.call_count,2)
            self.assertEqual(before,archive_fingerprint(Path(folder)/'turbidity'))


class SelectionTests(unittest.TestCase):
    def test_noise_variants_stay_in_same_generation_holdout(self):
        rows=[]
        for regime in range(8):
            for noise in [0,1]:
                for name in MODEL_NAMES:
                    rows.append({'scenario_id':f'{regime}_{noise}','selection_group':str(regime),'seed':7,'model':name,
                                 'graph_mode':'estimated','horizon':1,'validation_ap':.3,'test_ap':.3,
                                 **{f:float(regime) for f in FEATURE_NAMES}})
        with tempfile.TemporaryDirectory() as folder:
            rule=fit_selection(rows,Path(folder)/'rule.json')[0]
        train={s.split('_')[0] for s in rule['heldout']['train_scenario_ids']}
        test={s.split('_')[0] for s in rule['heldout']['heldout_scenario_ids']}
        self.assertFalse(train & test)

    def test_seeds_of_same_scenario_cannot_cross_holdout(self):
        rows = []
        for scenario in range(8):
            for seed in [1,2]:
                for i,name in enumerate(MODEL_NAMES):
                    rows.append({'scenario_id':str(scenario),'seed':seed,'model':name,'graph_mode':'estimated','horizon':1,
                                 'validation_ap':.1+.02*i,'test_ap':.15+.02*i,
                                 **{f:scenario+seed*.01 for f in FEATURE_NAMES}})
        with tempfile.TemporaryDirectory() as folder:
            rule = fit_selection(rows, Path(folder)/'selection.json')[0]
        self.assertFalse(set(rule['heldout']['train_scenario_ids']) & set(rule['heldout']['heldout_scenario_ids']))
        result = apply_rule(rule,{f:100 for f in FEATURE_NAMES})
        self.assertIsNone(result['selected_model'])
        self.assertTrue(result['outside_simulation_domain'])


if __name__ == '__main__':
    unittest.main()
