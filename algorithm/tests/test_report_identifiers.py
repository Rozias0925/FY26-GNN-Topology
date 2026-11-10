"""Numeric-looking hashes must survive report aggregation without coercion."""
from pathlib import Path
import sys
import tempfile
import unittest
import json
import numpy as np
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gnn_mechanisms.suite import read_metrics_csv
from gnn_mechanisms.real_validation import freeze, OBSERVABLE


class IdentifierTests(unittest.TestCase):
    def test_hashes_remain_exact_strings(self):
        for identifier in ['036689513890', '50028716e038', '12345678e999', '765e0fce6b79']:
            with self.subTest(identifier=identifier), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'metrics.csv'
                path.write_text('condition_id,seed,test_r2\n' + identifier + ',7,0.8\n')
                row = read_metrics_csv(path).iloc[0]
                self.assertEqual(row.condition_id, identifier)
                self.assertEqual(row.test_r2, .8)

    def test_external_expectation_retains_ids_and_matching_contrasts(self):
        identifiers = ['036689513890', '50028716e038']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            panel = root / 'training.npz'
            metadata = dict(training_only=True, training_cutoff=12,
                            target_alignment='features at t, target at t+1', dataset_id='test')
            np.savez(panel, features=np.ones((12, 3, 2)), target=np.ones((12, 3)),
                     graphs=np.tile(np.ones((3, 3)) - np.eye(3), (12, 1, 1)),
                     target_times=np.arange(1, 13), metadata=json.dumps(metadata))
            (root / 'complete.json').write_text('{}')
            (root / 'manifest.json').write_text(json.dumps({'config': {'purpose': 'research'}}))
            (root / 'metrics.csv').write_text(
                'condition_id,seed,task,regime,attention,n_nodes,' + ','.join(OBSERVABLE) + '\n' +
                ''.join(identifier + ',7,regression,mechanism,source,100,' + ','.join(['0.2'] * 6) + '\n'
                        for identifier in identifiers))
            (root / 'paired_contrasts.csv').write_text('condition_id,task,contrast,mean_gap\n' +
                ''.join(identifier + ',regression,GAT_vs_GCN,0.3\n' for identifier in identifiers))
            with patch('gnn_mechanisms.real_validation.probe_diagnostics',
                       return_value={key: .2 for key in OBSERVABLE}), \
                    patch('gnn_mechanisms.real_validation.estimate_relevance', return_value=.2):
                result = freeze(panel, root, root / 'expectation.json')
            self.assertEqual(set(result['nearest_condition_ids']), set(identifiers))
            self.assertAlmostEqual(result['expectation']['GAT_vs_GCN']['mean_local_gap'], .3)


if __name__ == '__main__':
    unittest.main()
