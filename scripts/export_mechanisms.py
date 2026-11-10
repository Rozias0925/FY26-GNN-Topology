"""Export a COMPLETE V2 study to the offline website without training or rewriting it."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MODELS = {'MLP', 'Static-GCN', 'Static-GAT', 'Dynamic-GCN', 'Dynamic-GAT'}
EXTRA_MODELS = {'Static-GATv2', 'Dynamic-GATv2'}
SEEDS = {7, 19, 31, 43, 59}
A_AXES = {'feature_only', 'uniform_neighbors', 'selective_neighbors', 'smooth_signal',
          'predictive_dynamics', 'noisy_dynamics'}
CSV_NAMES = ['metrics.csv', 'paired_contrasts.csv', 'response_curves.csv', 'phase_points.csv']
MAIN_FIGURES = {f'response_{axis}_regression.png' for axis in ['graph_signal','kappa','tau','delta','eta']} | {
    'phase_S_G_R_hat_Architecture_Static.png','phase_D_struct_D_pred_Dynamics_GCN.png',
    'phase_S_G_H_Static-GCN_minus_MLP.png','sample_size_n_nodes_regression.png'}
ROBUST_FIGURES = {'robustness_architecture_control_regression.png','robustness_graph_family_regression.png',
                  'robustness_noise_regression.png','robustness_classification_classification.png',
                  'robustness_query_dependent_regression.png'}
SHA = re.compile(r'^[0-9a-f]{64}$')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    require(path.is_file(), f'Missing required artifact: {path.name}')
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digest(folder):
    """Match the frozen package + runner hash, excluding separately copied runners."""
    require(folder.is_dir(), 'Missing frozen source snapshot')
    h = hashlib.sha256()
    files = sorted(p for p in folder.glob('*.py') if not p.name.startswith('run_'))
    require(bool(files), 'Empty frozen package snapshot')
    for p in files:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    h.update((folder / 'run_mechanisms.py').read_bytes())
    return h.hexdigest()


def check_source(folder, manifest):
    value = manifest.get('source_sha256', '')
    require(bool(SHA.fullmatch(value)), 'Invalid source SHA256')
    require(source_digest(folder / 'source') == value, 'Frozen source checksum mismatch')
    if 'runner_sha256' in manifest:
        require(digest(folder / 'source' / 'run_simulation_suite.py') == manifest['runner_sha256'],
                'Frozen suite runner checksum mismatch')
    return value


def value_key(value):
    if isinstance(value, str):
        if value in ('', 'null', 'None'):
            value = None
        else:
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    pass
    # Treat 1 and 1.0 as the same design point, including dictionary values.
    def normalize(v):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            require(math.isfinite(v), 'Nonfinite design value')
            return float(v)
        if isinstance(v, dict):
            return {k: normalize(x) for k, x in v.items()}
        return v
    return json.dumps(normalize(value), sort_keys=True, ensure_ascii=False, allow_nan=False)


def ref_key(row):
    return row['stage'], row['axis'], value_key(row.get('value'))


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_rows(payload, expected_conditions, expected_refs, models, expected_count):
    require(payload.get('schema_version') == 2 and payload.get('purpose') == 'research',
            'Only research V2 artifacts can be exported; smoke is not evidence')
    rows = payload.get('model_results', [])
    require(bool(rows), 'No model results')
    unique = {}
    references = {}
    for row in rows:
        key = row['condition_id'], int(row['seed']), row['model']
        require(key[0] in expected_conditions, 'Unplanned condition in report')
        require(key[1] in SEEDS and key[2] in models, 'Unplanned seed or model')
        ref = ref_key(row)
        require(ref in expected_refs[key[0]], 'Unplanned experiment reference')
        require((key, ref) not in references, 'Duplicate experiment reference')
        references[key, ref] = True
        for metric in (['test_r2', 'test_rmse', 'test_mae'] if row.get('task') == 'regression'
                       else ['test_ap', 'test_auroc', 'test_brier', 'test_log_loss']):
            require(finite(row.get(metric)), f'Missing/nonfinite primary or auxiliary score: {metric}')
        for metric in ['S_X', 'S_G', 'S_G_shuffled', 'H', 'R_hat', 'D_struct', 'D_pred']:
            require(finite(row.get(metric)), f'Missing/nonfinite diagnostic: {metric}')
        require(finite(row.get('R_true')) or (row.get('regime')=='smoothness' and row.get('R_true') is None),
                'Missing oracle relevance outside the separate denoising task')
        if key in unique:
            # Cross-stage references must be copies of the same evaluation.
            left = {k:v for k,v in unique[key].items() if k not in {'stage','axis','value'}}
            right = {k:v for k,v in row.items() if k not in {'stage','axis','value'}}
            require(left == right, 'Conflicting scores for a reused evaluation')
        unique[key] = row
    expected_keys = {(cid, seed, model) for cid in expected_conditions for seed in SEEDS for model in models}
    require(set(unique) == expected_keys and len(unique) == expected_count,
            'Incomplete paired datasets or model evaluation count')
    require(set(references) == {(key, ref) for key in expected_keys for ref in expected_refs[key[0]]},
            'Missing cross-stage experiment references')
    contrasts = payload.get('contrasts', [])
    require(bool(contrasts), 'Missing paired contrasts')
    seen_contrasts = set()
    for row in contrasts:
        require(row['condition_id'] in expected_conditions, 'Unplanned contrast condition')
        require(ref_key(row) in expected_refs[row['condition_id']], 'Unplanned contrast reference')
        key = row['condition_id'], ref_key(row), row['contrast']
        require(key not in seen_contrasts, 'Duplicate paired contrast')
        seen_contrasts.add(key)
        require(row.get('seeds') == len(SEEDS) and row.get('inference_ready') is True,
                'Paired inference requires all five dataset seeds')
        require(all(finite(row.get(k)) for k in ['mean_gap','ci_low','ci_high','p_delta_positive']),
                'Nonfinite paired contrast')
        require(row['ci_low'] <= row['mean_gap'] <= row['ci_high'], 'Invalid paired confidence interval')
        require(0 <= row['p_delta_positive'] <= 1, 'Invalid positive-seed fraction')
    base_contrasts = {'Static-GCN_minus_MLP','Static-GAT_minus_MLP','Dynamic-GCN_minus_MLP',
                      'Dynamic-GAT_minus_MLP','Architecture_Static','Architecture_Dynamic',
                      'Dynamics_GCN','Dynamics_GAT','Interaction'}
    require(all((cid, ref, contrast) in seen_contrasts for cid in expected_conditions
                for ref in expected_refs[cid] for contrast in base_contrasts), 'Missing factorial contrasts')
    return unique


def check_csv(folder, payload):
    """The downloaded scores must agree with the displayed rows, not only counts."""
    for name, rows in [('metrics.csv', payload['model_results']),
                       ('paired_contrasts.csv', payload['contrasts'])]:
        p = folder / name
        with p.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)
        require(len(csv_rows) == len(rows), f'CSV/JSON row count mismatch: {name}')
        key_fields = ['stage','axis','condition_id'] + (['seed','model'] if name=='metrics.csv' else ['contrast'])
        key = lambda r: tuple(str(r[k]) for k in key_fields) + (value_key(r.get('value')),)
        lookup = {key(r):r for r in csv_rows}
        require(len(lookup) == len(rows), f'Duplicate CSV keys: {name}')
        for row in rows:
            require(key(row) in lookup, f'CSV/JSON reference mismatch: {name}')
            match = lookup[key(row)]
            metrics = ['test_r2','test_ap','test_rmse','test_mae'] if name=='metrics.csv' else ['mean_gap','ci_low','ci_high']
            for metric in metrics:
                if finite(row.get(metric)):
                    require(metric in match and math.isclose(float(match[metric]),row[metric],rel_tol=1e-7,abs_tol=1e-9),
                            f'CSV/JSON score mismatch: {metric}')


def reject_paths(value):
    if isinstance(value, dict):
        for item in value.values(): reject_paths(item)
    elif isinstance(value, list):
        for item in value: reject_paths(item)
    elif isinstance(value, str):
        require(not re.search(r'(?:[A-Za-z]:[\\/]|\\\\|/(?:Users|home)/)', value),
                'Personal/absolute filesystem path in public artifact')


def collect(batch, stage_a):
    manifest = read_json(batch/'manifest.json')
    status = read_json(batch/'status.json')
    require(status.get('status') == 'complete' and not status.get('failures'),
            'V2 suite is incomplete; leave the pending website unchanged')
    marker = read_json(batch/'complete.json')
    cfg = manifest['config']
    require(cfg.get('purpose') == 'research', 'Smoke exports are forbidden')
    require(set(cfg.get('seeds', [])) == SEEDS, 'Website requires the formal five-seed design')
    gen = cfg['generator']
    require(all(gen.get(k)==v for k,v in {'n_nodes':500,'n_steps':40,'degree':10,'phi':.6,
                'noise':.3,'graph_family':'regular','task':'regression'}.items()), 'Website/generator setting mismatch')
    require(cfg.get('multi_axis_conditions') == 200 and cfg['training']['epochs'] == 200,
            'Website/main study size or training budget mismatch')
    require(cfg['training']['widths']==[32,64] and cfg['training']['learning_rates']==[.001,.003],
            'Website/tuning protocol mismatch')
    source = check_source(batch, manifest)
    require(marker.get('fingerprint') == manifest['fingerprint'] and marker.get('source_sha256') == source,
            'Suite completion provenance mismatch')
    require(status.get('completed_model_evaluations') == status.get('planned_model_evaluations') == 7240
            and marker.get('model_evaluations') == 7240, 'Suite model counts incomplete/mismatched')
    require(status.get('completed_jobs') == status.get('planned_jobs') == 284, 'Incomplete suite jobs')
    plan = read_json(batch/'queue_plan.json')
    require(len(plan) == 284 and len({j['id'] for j in plan}) == 284, 'Queue plan count mismatch')
    coverage = Counter(r['stage'] for j in plan if j['group']=='main' for r in j['references'])
    require(coverage == {'B':28,'C':56,'D':200,'sample_size':4}, 'Formal stage coverage mismatch')
    groups, copies, source_files = {}, [], {}
    group_sizes = {'main':(264,6600), 'convergence':(6,150), 'robustness':(14,490)}
    for name, (count, evaluations) in group_sizes.items():
        folder = batch/'reports'/name
        payload = read_json(folder/'website_results_v2.json')
        report_status = read_json(folder/'status.json')
        complete = read_json(folder/'complete.json')
        report_manifest = read_json(folder/'manifest.json')
        require(report_manifest['config']['training']['epochs'] == (400 if name=='convergence' else 200),
                f'Training-budget mismatch: {name}')
        require(payload.get('complete') is True and report_status.get('status') == 'complete', f'Partial group: {name}')
        require(payload.get('completed_conditions') == payload.get('planned_conditions') == count,
                f'Incomplete group condition count: {name}')
        require(payload.get('suite_source_sha256') == complete.get('source_sha256') == source
                and report_manifest.get('source_sha256') == source, f'Group provenance mismatch: {name}')
        require(complete.get('model_evaluations') == evaluations, f'Group evaluation count mismatch: {name}')
        jobs = [j for j in plan if j['group']==name]
        require(len(jobs) == count, f'Group queue count mismatch: {name}')
        expected = {j['id'].split('/')[-1]:{ref_key(r) for r in j['references']} for j in jobs}
        for job in jobs:
            job_marker = read_json(batch/'jobs'/job['id']/'complete.json')
            require(job_marker.get('source_sha256') == source, 'Incomplete/mismatched condition checkpoint')
        models = MODELS | EXTRA_MODELS if name=='robustness' else MODELS
        unique = check_rows(payload, set(expected), expected, models, evaluations)
        allowed = set(report_manifest.get('evaluation_source_sha256s',[]))
        require(allowed and all(SHA.fullmatch(x) for x in allowed), 'Missing evaluation-source provenance')
        require({r.get('evaluation_source_sha256') for r in unique.values()} == allowed,
                'Evaluation-source provenance mismatch')
        check_csv(folder, payload)
        wanted_figures = MAIN_FIGURES if name=='main' else ROBUST_FIGURES if name=='robustness' else set()
        require(set(payload.get('figures',[])) == wanted_figures, f'Incomplete figure coverage: {name}')
        download_names = CSV_NAMES + ['convergence_flags.csv']
        if name=='convergence': download_names += ['budget_sensitivity.csv','budget_sensitivity_paired.csv']
        for filename in download_names:
            require((folder/filename).is_file(), f'Missing download: {name}/{filename}')
        for filename in wanted_figures:
            require((folder/'figures'/filename).is_file(), f'Missing figure: {filename}')
        groups[name] = payload
        source_files[name] = {'website_json_sha256':digest(folder/'website_results_v2.json'),
                              'evaluation_source_sha256s':sorted(allowed)}
        copies.append((name, folder))
    # A uses the earlier frozen 200-epoch source and is never relabelled as convergence.
    a_manifest, a_status = read_json(stage_a/'manifest.json'), read_json(stage_a/'status.json')
    a_marker = read_json(stage_a/'complete.json')
    a_source = check_source(stage_a, a_manifest)
    require(a_status.get('status') == 'complete' and a_marker.get('model_evaluations') == 150,
            'Stage A incomplete')
    require(a_marker.get('fingerprint') == a_manifest['fingerprint'] and a_marker.get('source_sha256') == a_source,
            'Stage A completion provenance mismatch')
    require(a_manifest['config']['training']['epochs']==200 and set(a_manifest['config']['seeds'])==SEEDS,
            'Stage A protocol mismatch')
    payload = read_json(stage_a/'website_results_v2.json')
    expected = {}
    for row in payload.get('model_results',[]):
        require(row['stage']=='A' and row['axis'] in A_AXES, 'Unexpected Stage A regime')
        expected.setdefault(row['condition_id'],set()).add(ref_key(row))
    require(len(expected)==6 and {r['axis'] for r in payload['model_results']}==A_AXES, 'Stage A coverage mismatch')
    check_rows(payload,set(expected),expected,MODELS,150)
    check_csv(stage_a,payload)
    require(set(expected) == {r['condition_id'] for r in groups['convergence']['model_results']},
            '200/400-epoch conditions do not pair')
    for filename in CSV_NAMES:
        require((stage_a/filename).is_file(), f'Missing Stage A download: {filename}')
    payload.update(complete=True,completed_conditions=6,planned_conditions=6)
    groups['stage_a']=payload
    source_files['stage_a']={'website_json_sha256':digest(stage_a/'website_results_v2.json'),
                             'evaluation_source_sha256s':[a_source]}
    copies.append(('stage_a',stage_a))
    provenance={'suite_source_sha256':source,'stage_a_source_sha256':a_source,
                'completed_at':status.get('updated_at'),'suite_model_evaluations':7240,
                'separate_stage_a_evaluations':150,'groups':source_files}
    if (batch/'report_repair.json').exists():
        repair = read_json(batch/'report_repair.json')
        require(repair.get('status') == 'complete' and repair.get('training_source_sha256') == source
                and repair.get('verified_original_evaluations') == 7240
                and repair.get('checkpoint_scores_unchanged') is True,
                'Unverified report repair')
        fields = ['reason','completed_at','report_generation_source_sha256','repair_script_sha256',
                  'verified_original_evaluations','checkpoint_scores_unchanged']
        require(all(SHA.fullmatch(repair.get(k,'')) for k in
                    ['report_generation_source_sha256','repair_script_sha256']),
                'Invalid report-repair source checksum')
        summary = {k:repair[k] for k in fields}
        require(all(read_json(batch/'reports'/name/'manifest.json').get('report_repair') == summary
                    for name in group_sizes), 'Report-repair provenance mismatch')
        provenance['report_repair'] = summary
    public={'schema_version':2,'status':'complete','groups':groups,'provenance':provenance,'downloads':[]}
    reject_paths(public)
    return public, copies


def export(batch,stage_a,destination,check_only=False):
    public,copies=collect(Path(batch),Path(stage_a))  # All checks happen before destination writes.
    if check_only:
        return public
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='v2-export-',dir=destination.parent) as temp:
        staging=Path(temp)
        artifact_hashes={}
        for name,folder in copies:
            target=staging/'mechanisms-v2'/name
            target.mkdir(parents=True)
            figure_paths=[]
            for filename in public['groups'][name].get('figures',[]):
                require(Path(filename).name==filename and filename.endswith('.png'),'Invalid figure filename')
                src=folder/'figures'/filename
                require(src.is_file(),f'Missing scientific figure: {filename}')
                (target/'figures').mkdir(exist_ok=True)
                shutil.copyfile(src,target/'figures'/filename)
                figure_paths.append(f'data/mechanisms-v2/{name}/figures/{filename}')
            public['groups'][name]['figures']=figure_paths
            names=CSV_NAMES+(['convergence_flags.csv'] if name!='stage_a' else [])
            if name=='convergence':names+=['budget_sensitivity.csv','budget_sensitivity_paired.csv']
            for filename in names:
                src=folder/filename
                require(src.is_file(),f'Missing download: {filename}')
                reject_paths(src.read_text(encoding='utf-8-sig'))
                shutil.copyfile(src,target/filename)
                relative=f'data/mechanisms-v2/{name}/{filename}'
                public['downloads'].append({'group':name,'label':filename,'path':relative})
                artifact_hashes[relative]=digest(src)
            for filename in figure_paths:
                artifact_hashes[filename]=digest(staging/filename.removeprefix('data/'))
        public['provenance']['download_sha256']=artifact_hashes
        provenance_path=staging/'mechanisms-v2'/'provenance.json'
        provenance_path.write_text(json.dumps(public['provenance'],ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        public['downloads'].append({'group':'provenance','label':'SHA256','path':'data/mechanisms-v2/provenance.json'})
        reject_paths(public)
        payload=json.dumps(public,ensure_ascii=False,allow_nan=False,separators=(',',':'))
        (staging/'mechanism-results-v2.json').write_text(payload+'\n',encoding='utf-8')
        (staging/'mechanism-results-v2.js').write_text('/* Verified complete V2 study. */\nwindow.MECHANISM_RESULTS = '+payload+';\n',encoding='utf-8')
        destination.mkdir(parents=True,exist_ok=True)
        # Publish assets first; the JS data switch is last, so a failed validation cannot replace the stub.
        for p in staging.rglob('*'):
            if p.is_file() and p.name!='mechanism-results-v2.js':
                dest=destination/p.relative_to(staging)
                dest.parent.mkdir(parents=True,exist_ok=True)
                temp_dest=dest.with_name(dest.name+'.export-tmp')
                shutil.copyfile(p,temp_dest)
                os.replace(temp_dest,dest)
        os.replace(staging/'mechanism-results-v2.js',destination/'mechanism-results-v2.js')
    return public


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,default=ROOT/'algorithm/results/v2_full_suite_recovered_20261003')
    parser.add_argument('--stage-a',type=Path,default=ROOT/'algorithm/results/v2_stage_A')
    parser.add_argument('--destination',type=Path,default=ROOT/'dist/data')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    try:
        public=export(args.batch,args.stage_a,args.destination,args.check_only)
    except (ValueError,KeyError,TypeError,OSError) as error:
        parser.exit(1,f'Export refused: {error}\n')
    print(json.dumps({'status':'verified' if args.check_only else 'exported',
                      'suite_evaluations':public['provenance']['suite_model_evaluations']},ensure_ascii=False))


if __name__=='__main__':
    main()
