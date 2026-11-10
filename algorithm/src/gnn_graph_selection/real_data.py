"""WRA audit and causal hourly alignment. Never manufacture missing targets."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .sources import read_archive, SOURCES, archive_fingerprint
from .io import write_json, digest
from .protocol import Panel


def numeric(column):
    # Censored '<x' or '>x' observations are kept raw but not silently exact.
    value = pd.to_numeric(column, errors='coerce')
    return value.where(np.isfinite(value) & (value >= 0))


def timestamps(column):
    parsed = pd.to_datetime(column, format='mixed', errors='coerce')
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize('Asia/Taipei', ambiguous='NaT', nonexistent='NaT')
    return parsed.dt.tz_convert('UTC')


def load_turbidity(folder):
    frame = pd.DataFrame(read_archive(folder))
    if frame.empty or not {'monitordate', 'monitorlocation', 'turbidity'} <= set(frame):
        raise ValueError('No usable turbidity rows / unexpected source schema')
    frame['time'] = timestamps(frame['monitordate'])
    frame['station'] = frame['monitorlocation'].astype(str)
    frame['value'] = numeric(frame['turbidity'])
    return frame


def audit(raw_root, output, frame=None):
    root, output = Path(raw_root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = load_turbidity(root / 'turbidity') if frame is None else frame
    manifest = json.loads((root / 'turbidity' / 'manifest.json').read_text('utf-8'))
    stations = []
    for station, group in frame.groupby('station'):
        values = group['value']
        times = group['time'].dropna().sort_values().drop_duplicates()
        spacing = times.diff().dt.total_seconds().dropna() / 60
        stations.append({'station': station, 'rows': len(group), 'start': str(times.min()), 'end': str(times.max()),
                         'invalid_values': int(values.isna().sum()), 'invalid_times': int(group.time.isna().sum()),
                         'zero_fraction': float((values == 0).mean()), 'max_ntu': values.max(),
                         'median_spacing_minutes': spacing.median(), 'max_gap_hours': spacing.max() / 60,
                         'duplicate_timestamps': int(group.time.duplicated().sum()),
                         'exceedances_100_ntu': int((values > 100).sum()), 'exceedances_1000_ntu': int((values > 1000).sum()),
                         'exceedances_2000_ntu': int((values > 2000).sum())})
    auxiliary = {}
    for name in SOURCES:
        if name == 'turbidity' or not (root / name / 'manifest.json').exists():
            continue
        spec = SOURCES[name]
        rows = list(read_archive(root / name))
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if 'reservoiridentifier' in df:
            df = df[df.reservoiridentifier.astype(str) == '10201']
        dates = timestamps(df[spec['time_field']]) if spec.get('time_field') in df else pd.Series(dtype='datetime64[ns, UTC]')
        auxiliary[name] = {'total_rows': len(rows), 'shihmen_rows_if_reservoir': len(df),
                           'start': str(dates.min()), 'end': str(dates.max()), 'columns': list(df)}
    result = {'source': manifest['catalog'], 'archive_download_complete': manifest['download_complete'],
              'archive_manifest_hash': digest(manifest), 'total_rows': len(frame), 'stations': stations,
              'auxiliary': auxiliary,
              'warnings': ['An exhausted API is not proof of complete historical coverage.',
                           'Station codes and sensor depths must be checked against station metadata.',
                           'Zeros and long constant runs are flagged, not automatically treated as clean water.',
                           'Observation times do not establish operational publication latency.'],
              'formal_external_validation_ready': False}
    write_json(output / 'data_audit.json', result)
    pd.DataFrame(stations).to_csv(output / 'station_inventory.csv', index=False, encoding='utf-8-sig')
    return result


def hourly_station(group, minimum_per_hour=6, stuck_hours=24):
    """Hour ending t contains (t-1h,t]; last=point proxy, max=event target.

    Conflicting duplicates, insufficient samples and long flat sensor runs are
    missing. No interpolation or forward fill is applied to either label.
    """
    group = group.dropna(subset=['time']).sort_values('time')
    conflicts = group.groupby('time').value.nunique(dropna=False)
    group = group.drop_duplicates('time', keep='first').set_index('time')
    group.loc[conflicts[conflicts > 1].index, 'value'] = np.nan
    series = group.value
    aggregation = series.resample('1h', closed='right', label='right').agg(['last', 'max', 'count'])
    aggregation.loc[aggregation['count'] < minimum_per_hour, ['last', 'max']] = np.nan
    # Detect flat runs causally: invalidate only after the streak reaches length.
    gap = series.index.to_series().diff().dt.total_seconds() > 15 * 60
    run_start = (series.ne(series.shift()) | gap).cumsum()
    run_time = series.index.to_series().groupby(run_start).transform('first')
    stuck = ((series.index.to_series() - run_time).dt.total_seconds() >= stuck_hours * 3600) & series.notna()
    bad_hours = stuck.resample('1h', closed='right', label='right').max().reindex(aggregation.index).fillna(False).astype(bool)
    aggregation.loc[bad_hours, ['last', 'max']] = np.nan
    return aggregation


def prepare_real(raw_root, cfg, output=None, frame=None):
    root = Path(raw_root)
    manifest = json.loads((root / 'turbidity' / 'manifest.json').read_text('utf-8'))
    if not manifest['download_complete'] and not cfg.get('allow_partial_archive', False):
        raise ValueError('Turbidity download is incomplete; resume fetch before external analysis')
    frame = load_turbidity(root / 'turbidity') if frame is None else frame
    stations = cfg['stations']
    if len(set(stations)) != len(stations) or len(stations) < 2:
        raise ValueError('Specify at least two distinct aligned turbidity stations')
    if cfg['target_station'] not in stations:
        raise ValueError('target_station must occur in stations')
    available = set(frame.station.unique())
    if not set(stations) <= available:
        raise ValueError(f'Missing stations: {set(stations) - available}; available: {sorted(available)}')
    hourly = {s:hourly_station(frame[frame.station == s], cfg.get('minimum_observations_per_hour', 6), cfg.get('stuck_hours', 24)) for s in stations}
    start = pd.Timestamp(cfg['start'], tz='Asia/Taipei').tz_convert('UTC')
    end = pd.Timestamp(cfg['end'], tz='Asia/Taipei').tz_convert('UTC')
    index = pd.date_range(start, end, freq='1h', inclusive='left')
    if len(index) < 100:
        raise ValueError('Provide an explicit, sufficiently long analysis period')
    last = np.stack([hourly[s]['last'].reindex(index).to_numpy() for s in stations], axis=1)
    peak = np.stack([hourly[s]['max'].reindex(index).to_numpy() for s in stations], axis=1)
    # Station-level turbidity and log turbidity use identical feature definitions.
    features = np.stack([np.log1p(last), np.log1p(peak)], axis=-1)
    feature_names = ['log1p_last_ntu', 'log1p_hourly_peak_ntu']
    if cfg.get('include_reservoir_covariates', False):
        rows = pd.DataFrame(read_archive(root / 'reservoir'))
        rows = rows[rows.reservoiridentifier.astype(str) == '10201'].copy()
        rows['time'] = timestamps(rows.observationtime)
        rows = rows.dropna(subset=['time']).sort_values('time').drop_duplicates('time').set_index('time')
        # Cumulative rainfall is kept as cumulative: never mislabeled hourly rain.
        for field in ['waterlevel', 'inflowdischarge', 'accumulaterainfallincatchment']:
            values = numeric(rows[field]).resample('1h', closed='right', label='right').last().reindex(index).to_numpy()
            train_end = int(len(index) * cfg.get('train_fraction', .6))
            if np.isfinite(values[:train_end]).mean() < cfg.get('minimum_covariate_coverage', .5):
                raise ValueError(f'{field}: insufficient overlap with turbidity; obtain historical covariates or disable this explicit option')
            global_feature = np.broadcast_to(values[:, None, None], (*last.shape, 1))
            features = np.concatenate([features, global_feature], axis=-1)
            feature_names.append(field + '_reservoir_global')
    metadata = {'kind': 'real', 'timezone': 'Asia/Taipei', 'step_hours': 1,
                'feature_names': feature_names, 'raw_content_sha256': archive_fingerprint(root / 'turbidity'),
                'reservoir_content_sha256': archive_fingerprint(root / 'reservoir') if cfg.get('include_reservoir_covariates', False) else None,
                'regression_target': 'last observed NTU in hour ending t+h, with minimum coverage',
                'event_target': 'any hourly peak > tau within (t,t+h]',
                'latency_assumption': 'observations available at hour close; actual publication latency unverified',
                'coverage_by_station': {s:float(np.isfinite(last[:, i]).mean()) for i,s in enumerate(stations)}}
    panel = Panel(features.astype(np.float32), last, np.log1p(last), stations,
                  [stations.index(cfg['target_station'])], index.astype(str).to_numpy(), metadata, event_response=peak)
    if output is not None:
        save_panel(panel, output)
    return panel


def save_panel(panel, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'hourly_panel.npz', features=panel.features,
                        response=panel.response, event_response=panel.event_response,
                        graph_series=panel.graph_series, timestamps=np.asarray(panel.timestamps, dtype=str))
    write_json(output / 'panel_metadata.json', panel.metadata)
