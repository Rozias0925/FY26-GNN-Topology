"""Frozen-source, resumable sequential batch suitable for hidden background runs."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from gnn_graph_selection.io import write_json, source_digest, digest


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--simulation-config',type=Path,default=ROOT/'configs/overnight.json')
    parser.add_argument('--real-config',type=Path,default=ROOT/'configs/real_research.json')
    parser.add_argument('--raw',type=Path,default=ROOT/'data/raw')
    a=parser.parse_args()
    output=a.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    lock=output/'batch.lock'
    if lock.exists():
        # Never start a concurrent writer to the same experiment folder.
        pid=int(lock.read_text('ascii'))
        try:
            if sys.platform=='win32':
                import ctypes
                ctypes.windll.kernel32.OpenProcess.restype=ctypes.c_void_p
                ctypes.windll.kernel32.CloseHandle.argtypes=[ctypes.c_void_p]
                handle=ctypes.windll.kernel32.OpenProcess(0x1000,False,pid)
                if handle:
                    ctypes.windll.kernel32.CloseHandle(handle)
                    raise RuntimeError(f'Batch PID {pid} still exists; do not run two writers')
            else:
                os.kill(pid,0)
                raise RuntimeError(f'Batch PID {pid} still exists')
        except ProcessLookupError:
            pass
        lock.unlink()
    with lock.open('x',encoding='ascii') as f:
        f.write(str(os.getpid()))
    state={'pid':os.getpid(),'started_at':now(),'updated_at':now(),'status':'initializing','stages':{}}
    child=None
    try:
        frozen=output/'code'
        snapshot=output/'snapshot.json'
        if not snapshot.exists():
            frozen.mkdir(exist_ok=True)
            shutil.copytree(ROOT/'src',frozen/'src',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copytree(ROOT/'tests',frozen/'tests',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
            for p in ROOT.glob('*.py'):
                shutil.copy2(p,frozen/p.name)
            for name in ['requirements-lock.txt','pyproject.toml','README.md','RESEARCH_PROTOCOL.md']:
                if (ROOT/name).exists():
                    shutil.copy2(ROOT/name,frozen/name)
            (frozen/'configs').mkdir(exist_ok=True)
            shutil.copy2(a.simulation_config,frozen/'configs/simulation.json')
            shutil.copy2(a.real_config,frozen/'configs/real.json')
            write_json(snapshot,{'created_at':now(),'source_sha256':source_digest(),
                                'simulation_config':json.loads(a.simulation_config.read_text('utf-8')),
                                'real_config':json.loads(a.real_config.read_text('utf-8')),
                                'raw_path':str(a.raw.resolve()),'python':sys.executable})
        saved=json.loads(snapshot.read_text('utf-8'))
        raw=Path(saved['raw_path'])
        from gnn_graph_selection.experiment import scenario_specs
        config=saved['simulation_config']
        expected=len(list(scenario_specs(config)))*len(config['seeds'])*len(config['protocol']['horizons'])*5
        state['expected_simulation_models']=expected
        stages=[('tests',[sys.executable,'-m','unittest','discover','-s',str(frozen/'tests'),'-v']),
                ('simulation',[sys.executable,'-u',str(frozen/'run_research.py'),'simulate','--config',str(frozen/'configs/simulation.json'),'--output',str(output/'simulation')]),
                ('real',[sys.executable,'-u',str(frozen/'run_research.py'),'real','--config',str(frozen/'configs/real.json'),'--raw',str(raw),
                         '--selection',str(output/'simulation/selection_rule.json'),'--output',str(output/'real')])]
        for stage,command in stages:
            state.update(status='running',stage=stage,updated_at=now())
            state['stages'][stage]={'status':'running','started_at':now()}
            write_json(output/'status.json',state)
            print(f'{now()} Start {stage}',flush=True)
            env=os.environ.copy()
            env.update(PYTHONUTF8='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            with (output/f'{stage}.log').open('a',encoding='utf-8') as log:
                child=subprocess.Popen(command,cwd=frozen,stdout=log,stderr=subprocess.STDOUT,env=env)
                state['child_pid']=child.pid
                while child.poll() is None:
                    state['updated_at']=now()
                    state['completed_simulation_models']=sum(1 for _ in (output/'simulation').glob('*/seed_*/h*/*/result.json'))
                    write_json(output/'status.json',state)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
            if child.returncode:
                raise RuntimeError(f'{stage} exited {child.returncode}; see {stage}.log')
            manifest=output/stage/'manifest.json'
            final=json.loads(manifest.read_text('utf-8'))['status'] if manifest.exists() else 'complete'
            state['stages'][stage].update(status=final,finished_at=now())
            print(f'{now()} {stage}: {final}',flush=True)
        state['status']='complete_with_data_gap' if state['stages']['real']['status']=='insufficient_data' else 'complete'
        state['stage']='finished'
        state['completed_simulation_models']=sum(1 for _ in (output/'simulation').glob('*/seed_*/h*/*/result.json'))
        write_json(output/'website_manifest.json',{'simulation':'simulation/website_results.json','simulation_report':'simulation/report.html',
                   'real_status':state['stages']['real']['status'],'real_report':'real/report.html','snapshot':'snapshot.json'})
        (output/'index.html').write_text('<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>研究批次結果</title><body style="background:#12372e;color:#f7f6ed;font:18px/1.8 system-ui;padding:40px"><h1>研究批次結果</h1><p><a style="color:#f7f6ed" href="simulation/report.html">模擬比較與模型選擇規則</a></p><p><a style="color:#f7f6ed" href="real/report.html">真實資料結果或資料不足檢查</a></p><p>完成執行不代表假設獲得支持，請依報告與資料限制撰寫結論。</p></body></html>',encoding='utf-8')
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=30)
        state.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        state['updated_at']=now()
        write_json(output/'status.json',state)
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    main()
