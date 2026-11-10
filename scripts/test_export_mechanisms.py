"""Integrity tests for the public V2 export; fixtures never train or enter dist/."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

SPEC=importlib.util.spec_from_file_location('export_mechanisms',Path(__file__).with_name('export_mechanisms.py'))
exporter=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)
CONTRASTS=['Static-GCN_minus_MLP','Static-GAT_minus_MLP','Dynamic-GCN_minus_MLP','Dynamic-GAT_minus_MLP',
           'Architecture_Static','Architecture_Dynamic','Dynamics_GCN','Dynamics_GAT','Interaction']


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')


def source(folder):
    p=folder/'source';p.mkdir(parents=True)
    (p/'__init__.py').write_text('# fixture only\n',encoding='utf-8')
    (p/'run_mechanisms.py').write_text('# fixture only\n',encoding='utf-8')
    return exporter.source_digest(p)


def report(folder,jobs,models,source_hash,epochs):
    rows,contrasts=[],[]
    for job in jobs:
        cid=job['id'].split('/')[-1]
        for ref in job['references']:
            for seed in sorted(exporter.SEEDS):
                for model in sorted(models):
                    row=dict(ref,condition_id=cid,seed=seed,model=model,task='regression',
                             test_r2=.7,test_rmse=.3,test_mae=.2,evaluation_source_sha256=source_hash)
                    row.update({k:.1 for k in ['S_X','S_G','S_G_shuffled','H','R_hat','R_true','D_struct','D_pred']})
                    rows.append(row)
            for contrast in CONTRASTS:
                contrasts.append(dict(ref,condition_id=cid,contrast=contrast,seeds=5,inference_ready=True,
                                      mean_gap=.1,ci_low=.05,ci_high=.15,p_delta_positive=1.))
    name=jobs[0]['group']
    figures=sorted(exporter.MAIN_FIGURES if name=='main' else exporter.ROBUST_FIGURES if name=='robustness' else [])
    payload=dict(schema_version=2,purpose='research',model_results=rows,contrasts=contrasts,figures=figures,
                 complete=True,completed_conditions=len(jobs),planned_conditions=len(jobs),suite_source_sha256=source_hash)
    write(folder/'website_results_v2.json',payload)
    write(folder/'status.json',{'status':'complete'})
    write(folder/'complete.json',{'source_sha256':source_hash,'model_evaluations':len(jobs)*5*len(models)})
    write(folder/'manifest.json',{'source_sha256':source_hash,'evaluation_source_sha256s':[source_hash],
                                 'config':{'training':{'epochs':epochs}}})
    for filename,data in [('metrics.csv',rows),('paired_contrasts.csv',contrasts)]:
        with (folder/filename).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    names=['response_curves.csv','phase_points.csv','convergence_flags.csv']
    if name=='convergence':names+=['budget_sensitivity.csv','budget_sensitivity_paired.csv']
    for filename in names:(folder/filename).write_text('fixture\n',encoding='utf-8')
    for filename in figures:
        (folder/'figures').mkdir(exist_ok=True)
        (folder/'figures'/filename).write_bytes(b'fixture science figure')


def fixture(parent):
    batch,stage=parent/'batch',parent/'stage-a'
    sha=source(batch);a_sha=source(stage)
    cfg={'purpose':'research','seeds':sorted(exporter.SEEDS),'multi_axis_conditions':200,
         'generator':{'n_nodes':500,'n_steps':40,'degree':10,'phi':.6,'noise':.3,
                      'graph_family':'regular','task':'regression'},
         'training':{'epochs':200,'widths':[32,64],'learning_rates':[.001,.003]}}
    write(batch/'manifest.json',{'config':cfg,'source_sha256':sha,'fingerprint':'fixture'})
    write(batch/'status.json',{'status':'complete','failures':[],'updated_at':'fixture only',
           'completed_model_evaluations':7240,'planned_model_evaluations':7240,'completed_jobs':284,'planned_jobs':284})
    write(batch/'complete.json',{'source_sha256':sha,'fingerprint':'fixture','model_evaluations':7240})
    plan=[]
    for i in range(264):
        stage_name='B' if i<28 else 'C' if i<84 else 'D'
        plan.append({'id':f'main/m{i}','group':'main','references':[{'stage':stage_name,'axis':stage_name,'value':i}]})
    for i in range(20):plan[i]['references'].append({'stage':'D','axis':'D','value':180+i})
    for i in range(4):plan[i+20]['references'].append({'stage':'sample_size','axis':'N','value':i})
    a_jobs=[]
    for i,axis in enumerate(sorted(exporter.A_AXES)):
        a_jobs.append({'id':f'convergence/a{i}','group':'convergence',
                       'references':[{'stage':'A','axis':axis,'value':None}]})
    plan+=a_jobs
    plan+=[{'id':f'robustness/r{i}','group':'robustness','references':[{'stage':'robustness','axis':'test','value':i}]} for i in range(14)]
    write(batch/'queue_plan.json',plan)
    for job in plan:write(batch/'jobs'/job['id']/'complete.json',{'source_sha256':sha})
    for name in ['main','convergence','robustness']:
        report(batch/'reports'/name,[j for j in plan if j['group']==name],
               exporter.MODELS|exporter.EXTRA_MODELS if name=='robustness' else exporter.MODELS,
               sha,400 if name=='convergence' else 200)
    report(stage,a_jobs,exporter.MODELS,a_sha,200)
    write(stage/'manifest.json',{'config':cfg,'source_sha256':a_sha,'fingerprint':'fixture-a'})
    write(stage/'complete.json',{'source_sha256':a_sha,'fingerprint':'fixture-a','model_evaluations':150})
    return batch,stage


class ExportChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp_root=exporter.ROOT/'tmp';temp_root.mkdir(exist_ok=True)
        cls.temp=tempfile.TemporaryDirectory(prefix='export-tests-',dir=temp_root)
        cls.template=Path(cls.temp.name)/'template'
        fixture(cls.template)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix='case-',dir=self.temp.name))
        shutil.copytree(self.template,self.root/'input')
        self.batch,self.stage=self.root/'input/batch',self.root/'input/stage-a'
        self.destination=self.root/'public'
        self.destination.mkdir()
        (self.destination/'mechanism-results-v2.js').write_text('pending',encoding='utf-8')

    def refused(self):
        with self.assertRaises(ValueError):exporter.export(self.batch,self.stage,self.destination)
        self.assertEqual((self.destination/'mechanism-results-v2.js').read_text('utf-8'),'pending')

    def test_complete_export_and_immutable_source(self):
        before=exporter.digest(self.batch/'reports/main/metrics.csv')
        public=exporter.export(self.batch,self.stage,self.destination)
        self.assertEqual(public['status'],'complete')
        self.assertEqual(len(public['groups']['main']['model_results']),7200)
        self.assertEqual(len(public['groups']['main']['figures']),9)
        self.assertEqual(exporter.digest(self.batch/'reports/main/metrics.csv'),before)
        self.assertNotIn(str(self.root),json.dumps(public))
        self.assertTrue((self.destination/'mechanisms-v2/provenance.json').is_file())

    def test_partial_refused_without_stub_write(self):
        p=self.batch/'status.json';d=exporter.read_json(p);d['status']='running';write(p,d);self.refused()

    def test_missing_paired_seed_refused(self):
        p=self.batch/'reports/main/website_results_v2.json';d=exporter.read_json(p)
        d['model_results']=[r for r in d['model_results'] if not(r['condition_id']=='m0' and r['seed']==59)]
        write(p,d);self.refused()

    def test_nonfinite_score_refused(self):
        p=self.batch/'reports/main/website_results_v2.json';d=exporter.read_json(p);d['model_results'][0]['test_r2']=float('nan');write(p,d);self.refused()

    def test_tampered_source_refused(self):
        (self.batch/'source/__init__.py').write_text('changed',encoding='utf-8');self.refused()

    def test_smoke_refused(self):
        p=self.batch/'manifest.json';d=exporter.read_json(p);d['config']['purpose']='smoke';write(p,d);self.refused()

    def test_missing_figure_refused(self):
        (self.batch/'reports/main/figures/response_kappa_regression.png').unlink();self.refused()

    def test_conflicting_reuse_refused(self):
        p=self.batch/'reports/main/website_results_v2.json';d=exporter.read_json(p)
        row=next(r for r in d['model_results'] if r['condition_id']=='m0' and r['stage']=='D')
        row['test_r2']=.6;write(p,d);self.refused()

    def test_csv_disagrees_refused(self):
        p=self.batch/'reports/main/metrics.csv';s=p.read_text('utf-8');p.write_text(s.replace('0.7','0.8',1),encoding='utf-8');self.refused()

    def test_smoothness_has_no_oracle_attention(self):
        d=exporter.read_json(self.stage/'website_results_v2.json')
        d['model_results']=[r for r in d['model_results'] if r['axis']=='smooth_signal']
        d['contrasts']=[r for r in d['contrasts'] if r['axis']=='smooth_signal']
        for row in d['model_results']:row.update(regime='smoothness',R_true=None)
        cid=d['model_results'][0]['condition_id']
        refs={cid:{exporter.ref_key(d['model_results'][0])}}
        self.assertEqual(len(exporter.check_rows(d,{cid},refs,exporter.MODELS,25)),25)

    def test_verified_report_repair_provenance_exported(self):
        record=dict(status='complete',training_source_sha256=exporter.read_json(self.batch/'manifest.json')['source_sha256'],
                    reason='CSV identifier repair',completed_at='fixture only',
                    report_generation_source_sha256='1'*64,repair_script_sha256='2'*64,
                    verified_original_evaluations=7240,checkpoint_scores_unchanged=True)
        write(self.batch/'report_repair.json',record)
        fields=['reason','completed_at','report_generation_source_sha256','repair_script_sha256',
                'verified_original_evaluations','checkpoint_scores_unchanged']
        for name in ['main','robustness','convergence']:
            p=self.batch/'reports'/name/'manifest.json';d=exporter.read_json(p)
            d['report_repair']={k:record[k] for k in fields};write(p,d)
        public=exporter.export(self.batch,self.stage,self.destination)
        self.assertEqual(public['provenance']['report_repair']['verified_original_evaluations'],7240)

    def test_unverified_report_repair_refused(self):
        write(self.batch/'report_repair.json',{'status':'incomplete'})
        self.refused()


if __name__=='__main__':unittest.main()
