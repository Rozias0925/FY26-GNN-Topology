"""Import externally obtained station observations using a documented long schema."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from .protocol import Panel
from .real_data import timestamps
from .io import write_json

UNITS = {'turbidity': 'NTU', 'rainfall': 'mm', 'flow': 'm3/s', 'water_level': 'm'}


def prepare_csv(cfg, output=None):
    """Input fields: timestamp, available_at, station_id, variable, value, unit, quality.

    rainfall must be an interval amount whose interval ENDS at timestamp.
    Cumulative/daily rain must be converted before import. quality=good is required.
    available_at is a release time; omission is an explicit zero-latency assumption.
    """
    path = Path(cfg['observations_csv'])
    frame = pd.read_csv(path, dtype={'station_id':str})
    required = {'timestamp', 'station_id', 'variable', 'value', 'unit', 'quality'}
    if not required <= set(frame):
        raise ValueError(f'Missing CSV columns: {required - set(frame)}')
    if not set(frame.variable) <= set(UNITS):
        raise ValueError('Unsupported variable; use turbidity, rainfall, flow or water_level')
    for variable, unit in UNITS.items():
        if not (frame.loc[frame.variable == variable, 'unit'] == unit).all():
            raise ValueError(f'{variable} must use {unit}')
    frame['time'] = timestamps(frame.timestamp)
    if 'available_at' in frame:
        frame['available'] = timestamps(frame.available_at)
    elif cfg.get('assume_zero_latency', False):
        frame['available'] = frame.time
    else:
        raise ValueError('CSV needs available_at or an explicit assume_zero_latency setting')
    if frame.time.isna().any() or frame.available.isna().any() or (frame.available < frame.time).any():
        raise ValueError('Invalid observation/release timestamps')
    frame['value'] = pd.to_numeric(frame.value, errors='coerce')
    invalid = (frame.quality != 'good') | ~np.isfinite(frame.value)
    invalid |= (frame.variable != 'water_level') & (frame.value < 0)
    frame.loc[invalid, 'value'] = np.nan
    # Duplicate versions are ambiguous without versioned availability; fail early.
    if frame.duplicated(['station_id','variable','time']).any():
        raise ValueError('Duplicate station/variable/time rows: resolve revisions before import')
    stations = cfg['stations']
    if not set(stations) <= set(frame.station_id) or cfg['target_station'] not in stations:
        raise ValueError('Configured stations/target missing from CSV')
    index = pd.date_range(pd.Timestamp(cfg['start'],tz='Asia/Taipei').tz_convert('UTC'),
                          pd.Timestamp(cfg['end'],tz='Asia/Taipei').tz_convert('UTC'),freq='1h',inclusive='left')
    variables = list(UNITS)
    x = np.full((len(index),len(stations),len(variables)),np.nan)
    response = np.full((len(index),len(stations)),np.nan)
    peak = response.copy()
    graph = response.copy()
    for ni, station in enumerate(stations):
        for fi, variable in enumerate(variables):
            part = frame[(frame.station_id == station)&(frame.variable == variable)].sort_values('time')
            if part.empty:
                continue
            # Feature data become available no earlier than their recorded release.
            feature = part.sort_values('available').set_index('available').value.resample('1h',closed='right',label='right')
            values = feature.sum(min_count=1) if variable == 'rainfall' else feature.last()
            x[:,ni,fi] = values.reindex(index).to_numpy()
            if variable == 'turbidity':
                label = part.set_index('time').value.resample('1h',closed='right',label='right')
                count = label.count()
                last, maximum = label.last(), label.max()
                minimum = cfg.get('minimum_observations_per_hour',1)
                last[count < minimum] = np.nan
                maximum[count < minimum] = np.nan
                response[:,ni] = last.reindex(index).to_numpy()
                peak[:,ni] = maximum.reindex(index).to_numpy()
        graph_variable = cfg['graph_variable_by_station'][station]
        graph[:,ni] = x[:,ni,variables.index(graph_variable)]
    metadata = {'kind':'real_csv','step_hours':1,'timezone':'Asia/Taipei',
                'csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'variables':variables,
                'graph_variable_by_station':cfg['graph_variable_by_station'],
                'latency_assumption':'explicit available_at' if 'available_at' in frame else 'zero latency, unverified',
                'invalid_rows':int(invalid.sum()),'regression_target':'last observed NTU in ending hour','event_target':'hourly peak NTU'}
    panel = Panel(x.astype(np.float32),response,graph,stations,[stations.index(cfg['target_station'])],
                  index.astype(str).to_numpy(),metadata,event_response=peak)
    if output is not None:
        from .real_data import save_panel
        save_panel(panel, output)
    return panel
