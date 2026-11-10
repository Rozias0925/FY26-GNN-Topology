"""Window metrics and strictly pre-onset, one-to-one event/alarm evaluation."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def divide(a, b):
    return float(a / b) if b else None


def classification(y, probability, threshold=.5):
    y, probability = np.asarray(y).astype(bool).ravel(), np.asarray(probability, float).ravel()
    if y.shape != probability.shape or not len(y) or not np.isfinite(probability).all():
        raise ValueError('Invalid classification inputs')
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError('Probabilities must be in [0,1]')
    predicted = probability >= threshold
    tp, fp = int((predicted & y).sum()), int((predicted & ~y).sum())
    fn, tn = int((~predicted & y).sum()), int((~predicted & ~y).sum())
    p = np.clip(probability, 1e-7, 1 - 1e-7)
    return dict(n=len(y), prevalence=float(y.mean()), tp=tp, fp=fp, fn=fn, tn=tn,
                auroc=float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
                average_precision=float(average_precision_score(y, probability)) if y.any() else None,
                precision=divide(tp, tp + fp), recall=divide(tp, tp + fn),
                f1=divide(2 * tp, 2 * tp + fp + fn), fpr=divide(fp, fp + tn),
                false_alarm_ratio=divide(fp, tp + fp), brier=float(np.mean((probability - y) ** 2)),
                log_loss=float(-np.mean(y * np.log(p) + (~y) * np.log(1 - p))))


def choose_threshold(y, p):
    """Validation F1; a prespecified fixed grid, with high-q tie break."""
    candidates = np.linspace(.05, .95, 37)
    values = [(classification(y, p, q)['f1'] or 0, q) for q in candidates]
    return float(max(values)[1])


def regression(y, predicted):
    diff = np.asarray(predicted) - np.asarray(y)
    if not np.isfinite(diff).all():
        raise ValueError('Nonfinite regression predictions')
    return dict(mae=float(np.mean(np.abs(diff))), rmse=float(np.sqrt(np.mean(diff ** 2))))


def extract_events(series, tau, min_duration=1, merge_gap=0):
    """Consecutive exceedances; never merge across a missing observation."""
    if min_duration < 1 or merge_gap < 0:
        raise ValueError('Invalid event settings')
    y = np.asarray(series, float)
    high = np.isfinite(y) & (y > tau)
    changes = np.diff(np.r_[False, high, False].astype(int))
    runs = list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1) - 1))
    merged = []
    for start, end in runs:
        if merged and start - merged[-1][1] - 1 <= merge_gap and np.isfinite(y[merged[-1][1] + 1:start]).all():
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((int(start), int(end)))
    return [(s, e) for s, e in merged if e - s + 1 >= min_duration]


def event_metrics(series, issue_times, probability, tau, q, horizon, min_duration=1, merge_gap=0, cooldown=1):
    """Only alarms before a new onset count. Repeated alarms remain unmatched.

    Events need a full h-long eligible pre-onset window, known event end, and no
    missing targets. Denominator excludes censored events and reports their count.
    Cooldown uses elapsed time steps, including gaps, not row indices.
    """
    y = np.asarray(series, float)
    times = np.asarray(issue_times, int)
    p = np.asarray(probability, float)
    if times.shape != p.shape or len(times) == 0 or np.any(np.diff(times) <= 0) or cooldown < 1:
        raise ValueError('Event inputs require sorted unique forecast origins and cooldown >= 1')
    valid = {int(t) for t in times if t + horizon < len(y) and np.isfinite(y[t:t + horizon + 1]).all() and y[t] <= tau}
    candidates = extract_events(y, tau, min_duration, merge_gap)
    events, censored = [], 0
    for start, end in candidates:
        if start <= times[0] or start > times[-1] + horizon:
            continue
        if end + 1 >= len(y) or not np.isfinite(y[end + 1]) or not all(t in valid for t in range(start - horizon, start)):
            censored += 1
        else:
            events.append((start, end))
    alarms, last = [], -10**12
    for t, pr in zip(times, p):
        if t in valid and pr >= q and t - last >= cooldown:
            alarms.append(int(t))
            last = int(t)
    unmatched = set(alarms)
    matches = []
    for start, end in events:
        hits = sorted(t for t in unmatched if start - horizon <= t < start)
        alarm = hits[0] if hits else None
        if alarm is not None:
            unmatched.remove(alarm)
        matches.append({'onset': start, 'end': end, 'alarm': alarm, 'lead_time_steps': start - alarm if alarm is not None else None})
    lead = [m['lead_time_steps'] for m in matches if m['alarm'] is not None]
    # Alarms in windows of censored events are excluded from false-alarm counts.
    excluded_alarms = {t for t in unmatched if any(s - horizon <= t < s for s, e in candidates if (s, e) not in events)}
    false = unmatched - excluded_alarms
    duplicate = {t for t in false if any(s - horizon <= t < s for s, e in events)}
    return dict(n_events=len(events), censored_events=censored, detected=len(lead),
                detection_rate=divide(len(lead), len(events)), median_lead_steps=float(np.median(lead)) if lead else None,
                alarms=len(alarms), unmatched_alarms=len(false), duplicate_alarms=len(duplicate),
                event_free_alarms=len(false - duplicate), excluded_alarms=len(excluded_alarms),
                false_alarm_ratio=divide(len(false), len(alarms) - len(excluded_alarms)),
                eligible_observation_steps=len(valid), event_free_alarms_per_1000_steps=divide(1000 * len(false - duplicate), len(valid)),
                events=matches, alarm_times=alarms)


def block_bootstrap_metrics(y, p, block_length=24, repetitions=200, seed=0, issue_times=None):
    """Resample TIME blocks with all nodes together; no IID node/time bootstrap."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    block_length = min(n, block_length)
    times = np.arange(n) if issue_times is None else np.asarray(issue_times)
    if times.shape != (n,) or np.any(np.diff(times) <= 0):
        raise ValueError('Bootstrap origins must be sorted unique times')
    # Blocks stop at missing forecast origins instead of treating gaps as adjacent.
    boundaries = np.r_[0, np.flatnonzero(np.diff(times) != 1) + 1, n]
    ends = np.empty(n, dtype=int)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        ends[start:stop] = stop
    values = {'average_precision': [], 'brier': []}
    for _ in range(repetitions):
        chunks, count = [], 0
        while count < n:
            start = int(rng.integers(0, n))
            block = np.arange(start, min(start + block_length, ends[start]))
            chunks.append(block)
            count += len(block)
        ix = np.concatenate(chunks)[:n]
        row = classification(y[ix], p[ix])
        for metric in values:
            if row[metric] is not None:
                values[metric].append(row[metric])
    return {k: {'low': float(np.quantile(v, .025)), 'high': float(np.quantile(v, .975)), 'valid_repetitions': len(v)} if v else None for k, v in values.items()}
