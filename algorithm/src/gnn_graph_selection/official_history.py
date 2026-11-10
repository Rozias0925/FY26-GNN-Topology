"""Full-history event/coverage inventory, explicitly separate from the fixed study."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .io import write_json, digest, source_digest
from .sources import archive_fingerprint
from .real_data import prepare_real
from .official_analysis import csv, forecast_labels, label_readiness, event_catalog, boundaries


def common_period(inventory, stations):
    """Use timestamps only. Neither target values nor event counts choose the period."""
    group = inventory[inventory.station.isin(stations)]
    if len(group) != len(stations) or group.station.duplicated().any():
        raise ValueError('Exactly one inventory row per requested station is required')
    start = pd.to_datetime(group.start, utc=True).max().ceil('h').tz_convert('Asia/Taipei')
    end = (pd.to_datetime(group.end, utc=True).min().floor('h') + pd.Timedelta(hours=1)).tz_convert('Asia/Taipei')
    if pd.isna(start) or pd.isna(end) or end <= start:
        raise ValueError('No common chronological coverage')
    return start.strftime('%Y-%m-%d %H:%M:%S'), end.strftime('%Y-%m-%d %H:%M:%S')


def profile(raw, audit, config, output):
    raw, audit, output = Path(raw), Path(audit), Path(output)
    audit_manifest = json.loads((audit / 'analysis_manifest.json').read_text('utf-8'))
    if not audit_manifest.get('completed'):
        raise ValueError('Complete the fixed-setting analysis before profiling its station inventory')
    fingerprint = archive_fingerprint(raw / 'turbidity')
    if fingerprint != audit_manifest['raw_source_fingerprints']['turbidity']:
        raise ValueError('Station audit and raw archive identities differ')
    inventory_path = audit / 'station_inventory.csv'
    if hashlib.sha256(inventory_path.read_bytes()).hexdigest() != audit_manifest['outputs_sha256']['station_inventory.csv']:
        raise ValueError('Station inventory checksum mismatch')
    inventory = pd.read_csv(inventory_path, dtype={'station': str})
    start, end = common_period(inventory, config['data']['stations'])
    working = deepcopy(config)
    working['data'].update(start=start, end=end, allow_partial_archive=False)
    cfg = {**working['data'], **working['protocol']}
    output.mkdir(parents=True, exist_ok=True)
    marker = output / 'profile_manifest.json'
    write_json(marker, dict(completed=False, config_sha256=digest(working), raw_fingerprint=fingerprint))
    print(f'history profile: common raw coverage {start} to {end} (exclusive)', flush=True)
    panel = prepare_real(raw, cfg, output)
    dates = pd.to_datetime(panel.timestamps, utc=True).tz_convert('Asia/Taipei')
    labels = pd.concat([forecast_labels(panel, cfg, h) for h in cfg['horizons']], ignore_index=True)
    readiness = label_readiness(labels, cfg)
    csv(labels, output / 'forecast_labels.csv')
    csv(readiness.rename(columns={'classification_ready': 'minimum_class_count_check_passed',
                                  'regression_ready': 'minimum_regression_count_check_passed'}),
        output / 'minimum_sample_checks.csv')
    events = event_catalog(panel, cfg)
    csv(events, output / 'events.csv')
    target = panel.target_nodes[0]
    yearly = []
    for year in sorted(set(dates.year)):
        ix = dates.year == year
        values = panel.event_response[ix, target]
        year_events = events[(events.station == cfg['target_station']) & events.start.str.startswith(str(year))]
        yearly.append(dict(year=int(year), calendar_hours=int(ix.sum()), observed_hours=int(np.isfinite(values).sum()),
                           exceeding_hours=int((values > cfg['threshold']).sum()),
                           exceedance_segments=len(year_events), uncensored_segments=int((~year_events.censored).sum())))
    csv(pd.DataFrame(yearly), output / 'target_yearly_coverage_events.csv')
    train, valid = boundaries(len(dates), cfg)
    target_events = events[events.station == cfg['target_station']]
    result = dict(scope='descriptive full-history readiness audit; no models evaluated, no period selected by scores/events',
                  period_selection_rule='latest station start to earliest station end; timestamps only',
                  period=dict(start=start, end_exclusive=end), hourly_steps=len(dates),
                  target_station=cfg['target_station'], threshold_ntu=cfg['threshold'],
                  train_end_exclusive=dates[train].isoformat(), validation_end_exclusive=dates[valid].isoformat(),
                  minimum_class_count_check_passed=bool(readiness.classification_ready.all()),
                  observed_exceedance_segments_by_split=target_events.groupby('split').size().to_dict(),
                  uncensored_exceedance_segments_by_split=target_events[~target_events.censored].groupby('split').size().to_dict(),
                  fixed_2021_analysis_unchanged=True, formal_external_validation_ready=False,
                  limitations=[
                      'Positive forecast windows overlap; they do not equal independent events.',
                      'The newly audited interval is a candidate data inventory, not the original fixed 2021 study.',
                      'Recent rain/flow/water-level snapshots cannot supply the preceding historical years.',
                      'Three stations remain outside the synthetic N=100..1000 support.',
                      'Publication latency, physical station connectivity and predictor coverage remain unverified.',
                      'Do not pick a threshold or final test interval by looking for favorable model outcomes.'])
    write_json(output / 'history_summary.json', result)
    write_json(output / 'candidate_period_config.json', working)
    lines = ['# 完整官方歷史資料：事件與可用性盤點', '',
             f'共同資料期間：{start} 至 {end}（不含右端），{len(dates):,} 個小時。', '',
             '期間只根據三個測站的開始／結束時間取交集；1000 NTU、1／3／6h、60%／20%／20% 規則沿用。這是新的資料盤點，不替換原本 2021 年設定，也未跑模型或宣稱預警已成功。', '',
             '| 年 | 有觀測小時 | 超標小時 | 超標片段 | 邊界完整片段 |', '|---:|---:|---:|---:|---:|']
    for row in yearly:
        lines.append(f"| {row['year']} | {row['observed_hours']} | {row['exceeding_hours']} | {row['exceedance_segments']} | {row['uncensored_segments']} |")
    lines += ['', f"Training 結束界線：{result['train_end_exclusive']}；validation 結束界線：{result['validation_end_exclusive']}。", '',
              '| h | 切割 | Regression windows | Event windows | 正類 | 負類 |', '|---:|---|---:|---:|---:|---:|']
    for row in readiness.to_dict('records'):
        lines.append(f"| {row['horizon_hours']} | {row['split']} | {row['regression_windows']} | {row['event_windows']} | {row['positives']} | {row['negatives']} |")
    lines += ['', f"目標站超標片段依 split：{result['observed_exceedance_segments_by_split']}。邊界完整片段：{result['uncensored_exceedance_segments_by_split']}。", '',
              '相鄰超標片段僅是觀測序列的分段，可能屬於同一場颱風／豪雨；「邊界完整」也不代表已確認不同天氣事件。重疊 forecast windows 的正類數不能當作 independent event 數。', '',
              '即使 validation/test 出現正類，也仍需檢查事件數、同期 predictors、資料品質、發布延遲、圖的物理意義及 simulation domain mismatch。最後研究期間應事先記錄；不能依模型表現挑選 test。', '',
              '此盤點不擬合五模型、不修改既有模擬、不計算預警提前量。近期即時雨量／水情最多覆蓋下載前的短區間，不能補回數年的缺漏。']
    (output / 'FULL_HISTORY_READINESS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()
             if p.is_file() and p.name != marker.name}
    write_json(marker, dict(completed=True, generated_at=datetime.now(timezone.utc).isoformat(),
                            config_sha256=digest(working), raw_fingerprint=fingerprint,
                            source_code_sha256=source_digest(), outputs_sha256=files))
    return result
