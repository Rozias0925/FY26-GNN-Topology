"""Read-only label readiness; preserves the configured split and threshold."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from gnn_graph_selection.real_data import prepare_real
from gnn_graph_selection.protocol import time_splits
from gnn_graph_selection.evaluation import extract_events
from gnn_graph_selection.io import write_json


def inspect(panel,cfg):
    total=len(panel.features)
    train_end=int(total*cfg.get('train_fraction',.6))
    val_end=int(total*(cfg.get('train_fraction',.6)+cfg.get('validation_fraction',.2)))
    y=panel.response[:,panel.target_nodes]
    ye=(panel.event_response if panel.event_response is not None else panel.response)[:,panel.target_nodes]
    tau=cfg.get('threshold',float(np.nanquantile(ye[:train_end],cfg.get('threshold_quantile',.9))))
    horizon_rows=[]
    issues=[]
    for h in cfg['horizons']:
        times=np.arange(max(cfg['history_length']-1,cfg['graph_window']-1),total-h)
        keep=np.array([np.isfinite(y[t+1:t+h+1]).all() and np.isfinite(ye[t+1:t+h+1]).all() for t in times])
        times=times[keep]
        event=np.array([(ye[t+1:t+h+1]>tau).any(axis=0) for t in times])
        record={'horizon':h,'splits':{}}
        for name,ix in time_splits(times,h,train_end,val_end).items():
            label=event[ix]
            row={'windows':len(ix),'positives':int(label.sum()),'negatives':int(label.size-label.sum())}
            record['splits'][name]=row
            if len(ix)<cfg.get('min_split_samples',12) or (name!='test' and (row['positives']==0 or row['negatives']==0)):
                issues.append(f'h={h}: {name} has insufficient windows or only one class')
        horizon_rows.append(record)
    dates=pd.to_datetime(panel.timestamps,utc=True).tz_convert('Asia/Taipei')
    monthly=[]
    for name in sorted(set(dates.strftime('%Y-%m'))):
        a=ye[dates.strftime('%Y-%m')==name]
        monthly.append({'month':name,'observed_node_hours':int(np.isfinite(a).sum()),'exceeding_node_hours':int((a>tau).sum())})
    return {'ready_for_comparison':not issues,'issues':issues,'threshold':tau,'horizons':horizon_rows,
            'monthly_targets':monthly,'train_end_timestamp':str(dates[train_end]),'validation_end_timestamp':str(dates[val_end]),
            'warning':'Do not change the split or threshold using test outcomes to force an event comparison.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/real_research.json')
    p.add_argument('--raw',type=Path,default=ROOT/'data/raw')
    p.add_argument('--output',type=Path,default=ROOT/'data/audit/label_readiness.json')
    a=p.parse_args()
    c=json.loads(a.config.read_text('utf-8'))
    cfg={**c['protocol'],**c['data']}
    panel=prepare_real(a.raw,cfg)
    result=inspect(panel,cfg)
    write_json(a.output,result)
    print(json.dumps(result,ensure_ascii=True,indent=2))


if __name__=='__main__':
    main()
