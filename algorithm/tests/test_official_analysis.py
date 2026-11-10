"""Critical measured-data invariants, with no internet requests in tests."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gnn_graph_selection.protocol import Panel, graph_inputs
from gnn_graph_selection.official_analysis import (
    forecast_labels, label_readiness, inner_folds, training_diagnostics,
    event_catalog, block_paired_intervals,
    control_permutations,
)
from gnn_graph_selection.sources import archive_records, decode_records, download, read_archive, SOURCES
from gnn_graph_selection.real_data import hourly_station
from gnn_graph_selection.official_history import common_period


def make_panel(total=100):
    rng = np.random.default_rng(123)
    y = rng.uniform(0, 2000, (total, 3))
    dates = pd.date_range('2021-01-01', periods=total, freq='h', tz='Asia/Taipei').tz_convert('UTC')
    return Panel(np.stack([np.log1p(y), np.log1p(y)], -1).astype('float32'), y, np.log1p(y),
                 ['NW12', 'NX08', 'GP17'], [1], dates.astype(str).to_numpy(), {}, event_response=y.copy())


CFG = dict(history_length=4, graph_window=8, graph_stride=2, top_k=1,
           max_lag=1, ffill_limit=3, train_fraction=.6, validation_fraction=.2,
           threshold=1000, min_split_samples=4)


class ForecastTests(unittest.TestCase):
    def test_negative_controls_exclude_identity_even_for_three_nodes(self):
        # seed 47's first draw for N=3 is identity; the old single draw was no control.
        self.assertEqual(np.random.default_rng(47).permutation(3).tolist(), [0, 1, 2])
        permutations = control_permutations(3, 5, 47)
        self.assertEqual(len(permutations), 5)
        self.assertEqual(len({tuple(p) for p in permutations}), 5)
        self.assertTrue(all(not np.array_equal(p, np.arange(3)) for p in permutations))
        self.assertEqual(len(control_permutations(2, 5)), 1)

    def test_history_period_uses_common_timestamps_without_event_counts(self):
        inventory = pd.DataFrame({'station': ['a', 'b', 'c', 'ignored'],
                                  'start': ['2019-01-01', '2021-01-01', '2020-01-01', '2026-01-01'],
                                  'end': ['2026-10-11 09:00Z', '2026-10-12 09:00Z',
                                          '2026-10-13 09:00Z', '2026-10-14 09:00Z'],
                                  'exceedances': [0, 99, 3000, 1]})
        result = common_period(inventory, ['a', 'b', 'c'])
        self.assertEqual(result, ('2021-01-01 08:00:00', '2026-10-11 18:00:00'))
        inventory['exceedances'] = 9000
        self.assertEqual(common_period(inventory, ['a', 'b', 'c']), result)

    def test_missing_hour_is_not_negative_and_point_task_is_independent(self):
        panel = make_panel()
        panel.event_response[11, 1] = np.nan
        panel.response[11, 1] = np.nan
        panel.event_response[12, 1] = 1800
        panel.response[13, 1] = 25
        row = forecast_labels(panel, CFG, 3).set_index('issue_index').loc[10]
        self.assertTrue(np.isnan(row.event_target))
        self.assertEqual(row.point_target_ntu, 25)
        self.assertFalse(row.complete_future_event_window)

    def test_threshold_is_strict_and_current_hour_is_excluded(self):
        panel = make_panel()
        panel.event_response[10:14, 1] = [9999, 1000, 1000, 1000]
        labels = forecast_labels(panel, CFG, 3).set_index('issue_index')
        self.assertEqual(labels.loc[10, 'event_target'], 0)
        panel.event_response[13, 1] = 1000.01
        self.assertEqual(forecast_labels(panel, CFG, 3).set_index('issue_index').loc[10, 'event_target'], 1)

    def test_targets_do_not_cross_chronological_boundaries(self):
        labels = forecast_labels(make_panel(), CFG, 6)
        self.assertTrue((labels[labels.split == 'train'].issue_index + 6 < 60).all())
        self.assertTrue((labels[labels.split == 'validation'].issue_index >= 60).all())
        self.assertTrue((labels[labels.split == 'validation'].issue_index + 6 < 80).all())
        self.assertEqual(set(labels[labels.split == 'purged'].issue_index), set(range(54, 60)) | set(range(74, 80)))

    def test_regression_remains_ready_when_classification_lacks_events(self):
        panel = make_panel()
        panel.event_response[:] = 10
        ready = label_readiness(forecast_labels(panel, CFG, 3), CFG)
        self.assertTrue(ready.regression_ready.all())
        self.assertFalse(ready.classification_ready.any())

    def test_inner_folds_purge_actual_times_including_missing_origins(self):
        origins = np.r_[np.arange(8, 25), np.arange(30, 60)]
        for fit, hold in inner_folds(origins, 6):
            self.assertLess((origins[fit] + 6).max(), origins[hold].min())

    def test_graphs_and_diagnostics_ignore_future_test_values(self):
        a, b = make_panel(), make_panel()
        b.response[60:] *= 100
        b.event_response[60:] *= 100
        b.features[60:] *= 100
        b.graph_series[60:] *= 100
        graphs_a, graphs_b = graph_inputs(a, CFG), graph_inputs(b, CFG)
        np.testing.assert_array_equal(graphs_a[0], graphs_b[0])
        np.testing.assert_array_equal(graphs_a[1][:60], graphs_b[1][:60])
        config = dict(gate_max_snapshots=0)
        da = training_diagnostics(a, CFG, config, forecast_labels(a, CFG, 3), *graphs_a[:2])
        db = training_diagnostics(b, CFG, config, forecast_labels(b, CFG, 3), *graphs_b[:2])
        self.assertEqual(da, db)

    def test_events_are_censored_at_missing_boundary(self):
        panel = make_panel()
        panel.event_response[:] = 10
        panel.event_response[11, 1] = np.nan
        panel.event_response[12:14, 1] = 1500
        rows = event_catalog(panel, CFG)
        event = rows[rows.station == 'NX08'].iloc[0]
        self.assertTrue(event.censored)
        self.assertFalse(event.onset_observed)
        self.assertTrue(event.end_observed)

    def test_paired_interval_direction_and_gap_handling(self):
        labels = pd.DataFrame({'issue_index': [1, 2, 50, 51], 'point_target_ntu': [10., 20., 30., 40.]})
        result = block_paired_intervals(labels, labels.point_target_ntu.to_numpy(), np.zeros(4), 50, 24)
        self.assertGreater(result['ci95'][0], 0)
        self.assertEqual(result['mean'], 25)


class DownloadTests(unittest.TestCase):
    @staticmethod
    def fews():
        return {'timeZone': '8.0', 'timeSeries': [{'header': {
            'locationId': 'R10201', 'stationName': '石門水庫', 'units': 'mm',
            'type': 'accumulative', 'timeStep': {'unit': 'second', 'multiplier': '3600'},
            'missVal': '-999.0'}, 'events': [{'date': '2021-01-01', 'time': '01:00:00',
                                           'value': '-999.0', 'flag': '1'}]}]}

    def test_fews_big5_round_trip_and_missing_sentinel_is_preserved(self):
        raw = json.dumps(self.fews(), ensure_ascii=False).encode('cp950')
        spec = SOURCES['catchment_rain']
        row = archive_records(decode_records(raw), spec)[0]
        self.assertEqual(row['observationtime'], '2021-01-01T01:00:00+08:00')
        self.assertEqual(row['rainfall_raw'], '-999.0')
        self.assertEqual(row['value_type'], 'accumulative')
        with tempfile.TemporaryDirectory() as root, patch('gnn_graph_selection.sources.fetch_bytes', return_value=raw) as fetch:
            manifest = download('catchment_rain', root)
            self.assertTrue(manifest['download_complete'])
            self.assertEqual(manifest['rows'], 1)
            self.assertEqual(list(read_archive(Path(root) / 'catchment_rain')), [row])
            self.assertEqual(fetch.call_count, 1)

    def test_error_dict_is_not_a_successful_empty_dataset(self):
        with self.assertRaises(ValueError):
            archive_records({'error': 'not found'}, SOURCES['catchment_rain'])
        with self.assertRaises(ValueError):
            archive_records({'error': 'not found'}, SOURCES['turbidity'])

    def test_repeated_page_does_not_claim_download_completion(self):
        raw = b'[{"monitordate":"2021-01-01","turbidity":12}]'
        with tempfile.TemporaryDirectory() as root, patch('gnn_graph_selection.sources.fetch_bytes', return_value=raw):
            with self.assertRaisesRegex(ValueError, 'Repeated page'):
                download('turbidity', root, max_pages=2, page_size=1)
            manifest = json.loads((Path(root) / 'turbidity/manifest.json').read_text('utf-8'))
            self.assertFalse(manifest['download_complete'])

    def test_http_200_html_is_retried_and_not_saved_as_observations(self):
        valid = b'[{"monitordate":"2021-01-01","turbidity":12}]'
        with tempfile.TemporaryDirectory() as root, patch('gnn_graph_selection.sources.fetch_bytes',
                                                          side_effect=[b'<html>maintenance</html>', valid]), \
                patch('gnn_graph_selection.sources.time.sleep'):
            manifest = download('turbidity', root)
            self.assertTrue(manifest['download_complete'])
            self.assertEqual(list(read_archive(Path(root) / 'turbidity'))[0]['turbidity'], 12)

    def test_parallel_fetch_preserves_order_and_stops_at_first_empty_page(self):
        from urllib.parse import urlparse, parse_qs
        def response(url):
            page = int(parse_qs(urlparse(url).query)['page'][0])
            return json.dumps([{'page': page}] if page < 4 else []).encode()
        with tempfile.TemporaryDirectory() as root, patch('gnn_graph_selection.sources.fetch_bytes', side_effect=response):
            manifest = download('turbidity', root, max_pages=6, page_size=1, workers=3)
            self.assertTrue(manifest['download_complete'])
            self.assertEqual(manifest['rows'], 3)
            self.assertEqual([r['page'] for r in read_archive(Path(root) / 'turbidity')], [1, 2, 3])
            self.assertEqual([p['page'] for p in manifest['pages']], [1, 2, 3, 4])

    def test_conflicting_duplicates_are_not_valid_targets(self):
        dates = pd.date_range('2021-01-01 00:05', periods=12, freq='5min', tz='UTC')
        frame = pd.DataFrame({'time': list(dates) + [dates[-1]], 'value': [10.] * 12 + [2000.]})
        result = hourly_station(frame, minimum_per_hour=12)
        self.assertTrue(np.isnan(result.loc[dates[-1], 'max']))


if __name__ == '__main__':
    unittest.main()
