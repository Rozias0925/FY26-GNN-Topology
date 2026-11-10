"""Reaggregate completed checkpoints with string IDs; never run training.

The original reports and frozen training source are retained. The repair's
separate provenance records the code that generated the corrected reports.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'algorithm/src'))
from gnn_mechanisms.experiment import read_json, source_hash, write_json
from gnn_mechanisms.suite import build_jobs, read_metrics_csv, refresh_reports
from export_mechanisms import check_source, digest, require


def repair(batch):
    batch = Path(batch).resolve()
    status, manifest = read_json(batch/'status.json'), read_json(batch/'manifest.json')
    require(status['status'] == 'complete' and not status['failures'], 'Suite must be complete')
    training_source = check_source(batch, manifest)
    jobs = build_jobs(manifest['config'])
    plan = read_json(batch/'queue_plan.json')
    require({j['id'] for j in jobs} == {j['id'] for j in plan}, 'Queue reconstruction mismatch')
    require(len(jobs) == 284, 'Formal queue must have 284 jobs')
    checkpoint_hash = hashlib.sha256()
    score_keys = {}
    affected = []
    count = 0
    for job in jobs:
        folder = batch/'jobs'/job['id']
        cid = job['id'].split('/')[-1]
        require(read_json(folder/'complete.json')['source_sha256'] == training_source,
                'Incomplete checkpoint provenance')
        table = read_metrics_csv(folder/'metrics.csv')
        require(set(table.condition_id) == {cid}, 'Raw job CSV condition mismatch')
        block = table.drop_duplicates(['condition_id','seed','model'])
        require(len(block) == job['workload']['model_evaluations'], 'Missing checkpoint rows')
        original = read_json(folder/'website_results_v2.json')
        require(all(r['condition_id'] == cid for r in original['model_results']),
                'Original per-job JSON lost its identifier')
        try:
            float(cid)
            affected.append(job['id'])
        except ValueError:
            pass
        for row in block.to_dict('records'):
            p = folder/'datasets'/cid/f"seed_{row['seed']}"/row['model']/'result.json'
            result = read_json(p)
            require(result['model'] == row['model'], 'Checkpoint model mismatch')
            key = (job['group'],cid,int(row['seed']),row['model'])
            scores = {}
            for metric, value in result['test'].items():
                if isinstance(value, (int,float)) and not isinstance(value, bool) and math.isfinite(value):
                    field = 'test_' + metric
                    require(math.isclose(float(row[field]),value,rel_tol=1e-7,abs_tol=1e-9),
                            f'Raw checkpoint score mismatch: {job["id"]}/{field}')
                    scores[field] = value
            score_keys[key] = scores
            checkpoint_hash.update(str(p.relative_to(batch)).encode())
            checkpoint_hash.update(digest(p).encode())
            count += 1
    require(count == 7240, 'Expected 7240 original result checkpoints')
    backup = batch/'reports_before_id_repair_20261006'
    require(not backup.exists(), 'Original-report backup already exists; inspect before another repair')
    shutil.copytree(batch/'reports',backup)
    before = {str(p.relative_to(backup)):digest(p) for p in backup.rglob('*') if p.is_file()}
    refresh_reports(jobs,batch,{j['id'] for j in jobs},manifest['config'],
                    evaluation_source=training_source)
    checked = set()
    for name in ['main','robustness','convergence']:
        for row in read_json(batch/'reports'/name/'website_results_v2.json')['model_results']:
            key = (name,row['condition_id'],int(row['seed']),row['model'])
            require(key in score_keys, 'Repaired report has an unplanned checkpoint')
            for field, value in score_keys[key].items():
                require(math.isclose(row[field],value,rel_tol=1e-7,abs_tol=1e-9),
                        'Repaired report changed a test score')
            checked.add(key)
    require(len(checked) == 7240, 'Repaired report is missing model checkpoints')
    record = dict(status='complete',completed_at=datetime.now(timezone.utc).isoformat(),
                  reason='CSV type inference coerced numeric-looking condition hashes',
                  training_source_sha256=training_source,
                  report_generation_source_sha256=source_hash(),
                  repair_script_sha256=digest(Path(__file__)),
                  verified_original_evaluations=count,checkpoint_scores_unchanged=True,
                  checkpoint_result_manifest_sha256=checkpoint_hash.hexdigest(),
                  numeric_looking_job_ids=affected,original_report_sha256=before)
    write_json(batch/'report_repair.json',record)
    for name in ['main','robustness','convergence']:
        p = batch/'reports'/name/'manifest.json'
        value = read_json(p)
        value['report_repair'] = {k:record[k] for k in ['reason','completed_at',
            'report_generation_source_sha256','repair_script_sha256',
            'verified_original_evaluations','checkpoint_scores_unchanged']}
        write_json(p,value)
    return {k:record[k] for k in ['status','verified_original_evaluations',
                                'checkpoint_scores_unchanged','numeric_looking_job_ids']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,default=ROOT/'algorithm/results/v2_full_suite_recovered_20261003')
    print(json.dumps(repair(parser.parse_args().batch),ensure_ascii=False))
