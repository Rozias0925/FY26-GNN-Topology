"""Create reviewable reports and chronological event panels from saved outputs."""
from pathlib import Path
import html
import json
import numpy as np
import pandas as pd
from .io import write_json


def event_figures(panel, data, prediction, result, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = Path(folder)
    series = panel.event_response if panel.event_response is not None else panel.response
    issue = data['times'][data['splits']['test']]
    for j, record in enumerate(result['events']):
        node = data['target_nodes'][j]
        event_rows = []
        for event in record['events']:
            event_rows.append({**event, 'onset_timestamp':str(panel.timestamps[event['onset']]),
                               'end_timestamp':str(panel.timestamps[event['end']]),
                               'alarm_timestamp':str(panel.timestamps[event['alarm']]) if event['alarm'] is not None else None})
        write_json(folder / f'events_node_{node}.json', {'node':record['node'], 'events':event_rows})
        # Prespecified first three chronological eligible events, not best hits.
        for i, event in enumerate(event_rows[:3]):
            start, end = max(int(issue[0]), event['onset'] - 24), min(len(series), event['end'] + 25)
            times = np.arange(start, end)
            keep = (issue >= start) & (issue < end)
            fig, axes = plt.subplots(2,1,figsize=(10,5),sharex=True,constrained_layout=True)
            axes[0].plot(times,series[times,node],label='Observed hourly peak' if panel.metadata.get('step_hours') else 'Observed response')
            axes[0].axhline(data['threshold'],color='gray',linestyle='--',label='Event threshold')
            axes[0].axvline(event['onset'],color='black',linestyle=':')
            axes[0].set_ylabel('NTU' if panel.metadata.get('step_hours') else 'Response')
            axes[1].plot(issue[keep],prediction[keep,j],label='Forecast probability')
            axes[1].axhline(result['alarm_threshold'],color='gray',linestyle='--',label='Alarm threshold')
            for alarm in record['alarm_times']:
                if start <= alarm < end:
                    axes[1].axvline(alarm,color='orange',alpha=.3)
            axes[1].set(xlabel='Time index (hours)' if panel.metadata.get('step_hours') else 'Time index (steps)',ylabel='Probability',ylim=(0,1))
            for ax in axes:
                ax.legend(fontsize=8)
            fig.suptitle(f'{result["model"]} / {record["node"]} / h={data["horizon"]} / onset {event["onset_timestamp"]} / lead={event["lead_time_steps"]}')
            fig.savefig(folder/f'event_{node}_{i+1}.png',dpi=150)
            plt.close(fig)


def write_report(output,cfg,rows):
    output=Path(output)
    frame=pd.DataFrame(rows)
    cols=['graph_mode','scenario_id','seed','horizon','model','test_ap','test_auroc','test_brier','test_f1','test_rmse']
    table=frame[cols].to_html(index=False,float_format=lambda x:f'{x:.4f}',escape=True)
    figures=''.join(f'<figure><img src="{html.escape(p.name)}" alt="Model comparison"></figure>' for p in sorted(output.glob('comparison_*.png')))
    purpose=html.escape(cfg.get('purpose','research'))
    cases=''.join(f'<li><a href="{html.escape(p.relative_to(output).as_posix())}">{html.escape(p.relative_to(output).as_posix())}</a></li>' for p in sorted(output.glob('seed_*/h*/*/events_node_*.json')))
    transfer='<a href="transfer_evaluation.json">外部規則套用與拒絕選擇紀錄</a>' if (output/'transfer_evaluation.json').exists() else ''
    uncertainty='實測資料跨種子區間只反映訓練隨機性；同一段實測資料不算多個獨立水文樣本。' if 'data' in cfg else '跨種子區間反映本生成器下的 Monte Carlo 變異；不涵蓋生成器錯設或真實世界差異。'
    body=f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Graph model experiment report</title><style>body{{margin:0;background:#12372e;color:#243a30;font:16px/1.7 system-ui}}main{{max-width:1200px;margin:32px auto;padding:32px;background:#f7f6ed}}h1{{font-size:26px}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:8px;border:1px solid #d0d5c3}}.table{{overflow:auto}}img{{max-width:100%}}a{{color:#315b43}}</style>
<main><h1>Static and Dynamic Graph Construction for GNNs</h1><p>執行用途：{purpose}。Smoke 結果僅用來驗證程式，不構成研究結論。</p>
<p>每組結果使用相同時間切分；模型與警報門檻只用驗證集決定。欄位 test_ap 為 Average Precision（非梯形積分 PR-AUC）。</p>
<p>{uncertainty} 模型間請採同一情境、種子與 horizon 的配對比較；跨情境平均不是選擇規則。</p>
<p><a href="metrics.csv">完整指標 CSV</a> · <a href="summary.csv">跨種子彙整與區間</a> · <a href="selection_rule.json">模型選擇規則</a> · <a href="website_results.json">網頁資料匯出</a> · <a href="manifest.json">版本與設定</a></p>
{figures}<p>{transfer}</p><ul>{cases}</ul><div class="table">{table}</div></main></html>'''
    (output/'report.html').write_text(body,encoding='utf-8')


def write_readiness_report(output, readiness):
    output=Path(output)
    records=[{'horizon':r['horizon'],'split':name,**values} for r in readiness['horizons'] for name,values in r['splits'].items()]
    table=pd.DataFrame(records).to_html(index=False,escape=True)
    body=f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>資料可用性檢查</title>
<style>body{{background:#12372e;color:#243a30;font:16px/1.7 system-ui}}main{{background:#f7f6ed;max-width:1000px;margin:32px auto;padding:32px}}td,th{{padding:8px;border:1px solid #aaa}}table{{border-collapse:collapse}}</style>
<main><h1>真實案例：目前資料不足以驗證高濁度預警</h1><p>固定門檻 {readiness['threshold']} NTU，固定時間切分；訓練或驗證資料缺少類別，因此未訓練模型或產生假結果。</p>
<p>應補充足夠歷史事件與同期上游資料，再凍結新的研究版本。尚未證明模型可提前預警。</p>{table}
<p><a href="label_readiness.json">完整資料檢查</a> · <a href="manifest.json">執行設定</a></p></main></html>'''
    (output/'report.html').write_text(body,encoding='utf-8')
