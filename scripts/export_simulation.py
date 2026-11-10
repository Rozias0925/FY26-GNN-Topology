"""Publish verified, aggregate simulation artifacts into the offline website.

Run from the project root. No model retraining, private paths or raw observations.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['MLP', 'Static GCN', 'Static GAT', 'Dynamic GCN', 'Dynamic GAT']
METRICS = ['test_ap', 'test_auroc', 'test_brier', 'test_f1', 'test_mae', 'test_rmse']
FEATURES = ['graph_utility', 'sdnr', 'nrc', 'dirichlet_energy', 'density']


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    text_fields = {'model','scenario_id','selection_group','graph_mode','reference','metric'}
    for row in rows:
        for key, value in row.items():
            if key not in text_fields:
                row[key] = float(value) if value else None
                if row[key] is not None and not math.isfinite(row[key]):
                    raise ValueError(f'Nonfinite {key} in {path}')
    return rows


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def export(batch, destination):
    sim = batch / 'simulation'
    status = json.loads((batch / 'status.json').read_text('utf-8'))
    manifest = json.loads((sim / 'manifest.json').read_text('utf-8'))
    if manifest['status'] != 'complete' or status['completed_simulation_models'] != status['expected_simulation_models']:
        raise ValueError('Export requires a completed simulation batch')
    rows = read_csv(sim / 'metrics.csv')
    summary = read_csv(sim / 'summary.csv')
    paired = read_csv(sim / 'paired_differences.csv')
    keys = {(r['scenario_id'],r['seed'],r['horizon'],r['model']) for r in rows}
    if len(keys) != len(rows) or len(rows) != status['expected_simulation_models']:
        raise ValueError('Missing or duplicate model metrics')
    horizons = manifest['config']['protocol']['horizons']
    seeds = manifest['config']['seeds']
    scenarios = []
    baselines = []
    for sid in sorted({r['scenario_id'] for r in rows}):
        subset = [r for r in rows if r['scenario_id'] == sid]
        first = subset[0]
        expected = {(sid,float(s),float(h),m) for s in seeds for h in horizons for m in MODELS}
        if not expected <= keys:
            raise ValueError(f'Incomplete scenario: {sid}')
        observed = [r for r in subset if r['horizon'] == horizons[0] and r['model'] == 'MLP']
        scenario = {k:first[k] for k in ['scenario_id','graph_mode','selection_group','graph_signal','graph_drift','edge_noise','neighbor_heterogeneity']}
        scenario['descriptors'] = {k:{'mean':mean([r[k] for r in observed]),
            'min':min(r[k] for r in observed if r[k] is not None) if any(r[k] is not None for r in observed) else None,
            'max':max(r[k] for r in observed if r[k] is not None) if any(r[k] is not None for r in observed) else None} for k in FEATURES}
        scenarios.append(scenario)
        for h in horizons:
            source = [json.loads((sim/sid/f'seed_{seed}'/f'h{h}'/'baselines.json').read_text('utf-8')) for seed in seeds]
            baselines.append({'graph_mode':first['graph_mode'],'scenario_id':sid,'horizon':h,
                'climatology_ap':mean([r['climatology']['average_precision'] for r in source]),
                'persistence_ap':mean([r['persistence']['average_precision'] for r in source]),
                'persistence_rmse':mean([r['persistence']['rmse'] for r in source]),
                'prevalence':mean([r['climatology']['prevalence'] for r in source])})
    for item in summary:
        subset = [r for r in rows if all(r[k]==item[k] for k in ['scenario_id','horizon','model'])]
        for metric in METRICS:
            if not math.isclose(item[metric],mean([r[metric] for r in subset]),rel_tol=1e-10,abs_tol=1e-12):
                raise ValueError(f'Summary mismatch: {metric}')
        item['seconds'] = mean([r['seconds'] for r in subset])
        item['parameters'] = subset[0]['parameters']
    aggregate = []
    for mode in manifest['config']['graph_modes']:
        mode_scenarios = [s for s in scenarios if s['graph_mode']==mode]
        for h in horizons:
            for model in MODELS:
                subset = [r for r in summary if r['graph_mode']==mode and r['horizon']==h and r['model']==model]
                aggregate.append({'graph_mode':mode,'scenario_id':'all','horizon':h,'model':model,
                    'scenario_count':len(mode_scenarios), 'replicates':len(seeds),
                    **{m:mean([r[m] for r in subset]) for m in [*METRICS,'seconds','parameters']}})
            base = [r for r in baselines if r['graph_mode']==mode and r['horizon']==h]
            baselines.append({'graph_mode':mode,'scenario_id':'all','horizon':h,
                **{m:mean([r[m] for r in base]) for m in ['climatology_ap','persistence_ap','persistence_rmse','prevalence']}})
    source_names = ['metrics.csv','summary.csv','paired_differences.csv','selection_rule.json']
    data = {'schemaVersion':1,'batch':batch.name,'purpose':manifest['purpose'],
        'completedAt':status['stages']['simulation']['finished_at'], 'modelRuns':len(rows),
        'candidateFits':len(rows)*len(manifest['config']['training']['widths'])*len(manifest['config']['training']['learning_rates']),
        'models':MODELS,'horizons':horizons,'config':manifest['config'],
        'sourceSha256':manifest['source_sha256'], 'fingerprint':manifest['fingerprint'],
        'sourceFiles':{name:hashlib.sha256((sim/name).read_bytes()).hexdigest() for name in source_names},
        'scenarios':scenarios,'summary':summary,'aggregate':aggregate,'paired':paired,'baselines':baselines,
        'rules':json.loads((sim/'selection_rule.json').read_text('utf-8'))['rules']}
    destination.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    (destination/'simulation-results.json').write_text(payload+'\n',encoding='utf-8')
    (destination/'simulation-results.js').write_text('/* Generated by scripts/export_simulation.py; completed simulation artifacts only. */\nwindow.SIMULATION_RESULTS = '+payload+';\n',encoding='utf-8')
    downloads=destination/'simulation'
    downloads.mkdir(exist_ok=True)
    for name in source_names:
        shutil.copy2(sim/name,downloads/name)
    (downloads/'config.json').write_text(json.dumps(manifest['config'],ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    provenance={k:data[k] for k in ['batch','purpose','completedAt','modelRuns','candidateFits','sourceSha256','fingerprint','sourceFiles']}
    (downloads/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print(f'Exported {len(rows)} verified models, {len(scenarios)} scenarios and {len(data["rules"])} rules to {destination}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,default=ROOT/'algorithm/results/overnight_20260930')
    parser.add_argument('--destination',type=Path,default=ROOT/'dist/data')
    args=parser.parse_args()
    export(args.batch,args.destination)
