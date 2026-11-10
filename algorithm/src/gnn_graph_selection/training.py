"""Identical splits, tuning budgets, early stopping and outputs for all models."""
from __future__ import annotations
import copy
import itertools
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .models import ForecastModel, MODEL_NAMES
from .evaluation import classification, regression, choose_threshold, event_metrics, block_bootstrap_metrics
from .protocol import inverse_response
from .io import write_json


def predict(model, data, indices, device, batch_size=256):
    model.eval()
    probabilities, responses = [], []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            ix = indices[start:start + batch_size]
            x = torch.as_tensor(data['x'][ix], device=device)
            graph = data['dynamic'][ix] if model.name.startswith('Dynamic') else data['static']
            logits, response = model(x, torch.as_tensor(graph, device=device), data['target_nodes'])
            probabilities.append(logits.sigmoid().cpu().numpy())
            responses.append(response.cpu().numpy())
    return np.concatenate(probabilities), inverse_response(np.concatenate(responses), data)


def fit_candidate(name, data, cfg, width, learning_rate, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cfg.get('threads', 2))
    torch.use_deterministic_algorithms(True)
    device = torch.device(cfg.get('device', 'cpu'))
    model = ForecastModel(name, data['x'].shape[1], data['x'].shape[-1], width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=cfg.get('weight_decay', 1e-4))
    rng = np.random.default_rng(seed)
    train = data['splits']['train']
    valid = data['splits']['validation']
    batch_size = cfg.get('batch_size', 64)
    best_loss, best, stale, best_epoch = float('inf'), None, 0, 0
    history = []
    for epoch in range(cfg['epochs']):
        model.train()
        shuffled = rng.permutation(train)
        losses = []
        for start in range(0, len(shuffled), batch_size):
            ix = shuffled[start:start + batch_size]
            x = torch.as_tensor(data['x'][ix], device=device)
            a = data['dynamic'][ix] if name.startswith('Dynamic') else data['static']
            logits, predicted = model(x, torch.as_tensor(a, device=device), data['target_nodes'])
            target = torch.as_tensor(data['event'][ix], device=device)
            response = torch.as_tensor(data['scaled_regression'][ix], device=device)
            loss = F.binary_cross_entropy_with_logits(logits, target) + cfg.get('regression_weight', .2) * F.mse_loss(predicted, response)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{name}: nonfinite loss')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        probability, _ = predict(model, data, valid, device)
        score = classification(data['event'][valid], probability)['log_loss']
        history.append({'epoch': epoch + 1, 'training_loss': float(np.mean(losses)), 'validation_log_loss': score})
        if score < best_loss - cfg.get('min_delta', 1e-5):
            best_loss, best_epoch, stale = score, epoch + 1, 0
            best = copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
        else:
            stale += 1
        if stale >= cfg.get('patience', 12):
            break
    model.load_state_dict(best)
    return model, {'width': width, 'learning_rate': learning_rate, 'best_epoch': best_epoch,
                   'validation_log_loss': best_loss, 'history': history, 'seed': seed,
                   'parameter_count': sum(p.numel() for p in model.parameters())}


def evaluate_model(name, data, panel, cfg, folder, seed):
    started = time.perf_counter()
    candidates = []
    best_model, best_meta = None, None
    grid = itertools.product(cfg.get('widths', [32]), cfg.get('learning_rates', [.001]))
    for width, rate in grid:
        model, meta = fit_candidate(name, data, cfg, width, rate, seed)
        candidates.append(meta)
        if best_meta is None or meta['validation_log_loss'] < best_meta['validation_log_loss']:
            best_model, best_meta = model, meta
    model = best_model
    device = next(model.parameters()).device
    vi, ti = data['splits']['validation'], data['splits']['test']
    vp, vr = predict(model, data, vi, device)
    q = choose_threshold(data['event'][vi], vp)
    tp, tr = predict(model, data, ti, device)
    all_events = []
    for j, node in enumerate(data['target_nodes']):
        event_series = panel.event_response if panel.event_response is not None else panel.response
        result = event_metrics(event_series[:, node], data['times'][ti], tp[:, j], data['threshold'], q,
                               data['horizon'], **cfg.get('events', {}))
        result['node'] = panel.node_names[node]
        all_events.append(result)
    result = {'model': name, 'horizon': data['horizon'], 'seed': seed, 'alarm_threshold': q,
              'selected_hyperparameters': {k:v for k,v in best_meta.items() if k != 'history'},
              'validation': {**classification(data['event'][vi], vp, q), **regression(data['regression'][vi], vr)},
              'test': {**classification(data['event'][ti], tp, q), **regression(data['regression'][ti], tr)},
              'events': all_events, 'training_seconds': time.perf_counter() - started,
              'runtime_scope': 'training, validation, test prediction and event scoring; excludes graph construction, bootstrap and plotting',
              'test_block_bootstrap': block_bootstrap_metrics(data['event'][ti], tp,
                  block_length=max(data['horizon'], cfg.get('bootstrap_block', 24)),
                  repetitions=cfg.get('bootstrap_repetitions', 100), seed=seed, issue_times=data['times'][ti])}
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), folder / 'weights.pt')
    np.savez_compressed(folder / 'predictions.npz', validation_times=data['times'][vi], validation_probability=vp,
                        test_times=data['times'][ti], test_probability=tp, test_regression=tr,
                        test_event=data['event'][ti], test_target=data['regression'][ti])
    write_json(folder / 'training.json', candidates)
    if panel.metadata.get('kind', '').startswith('real'):
        from .reporting import event_figures
        event_figures(panel, data, tp, result, folder)
    # Completion marker is written only after every artifact succeeds.
    write_json(folder / 'result.json', result)
    return result


def baselines(data, panel):
    ti, train = data['splits']['test'], data['splits']['train']
    current = panel.response[data['times'][ti]][:, data['target_nodes']]
    event_series = panel.event_response if panel.event_response is not None else panel.response
    current_event = event_series[data['times'][ti]][:, data['target_nodes']]
    valid = np.isfinite(current).all(axis=1) & np.isfinite(current_event).all(axis=1)
    prevalence = data['event'][train].mean(axis=0)
    climatology = np.broadcast_to(prevalence, data['event'][ti].shape)
    return {'climatology': classification(data['event'][ti], climatology),
            'persistence': {**classification(data['event'][ti][valid], (current_event[valid] > data['threshold']).astype(float)),
                            **regression(data['regression'][ti][valid], current[valid]),
                            'excluded_missing_current': int((~valid).sum())} if valid.any() else None}
