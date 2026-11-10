"""Grouped held-out scenario validation for an observable shallow selection rule."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeRegressor, export_text
from .models import MODEL_NAMES
from .descriptors import FEATURE_NAMES
from .io import write_json


def fit_selection(rows, path, gray_margin=.01):
    df = pd.DataFrame(rows)
    outputs = []
    for (mode, horizon), group in df.groupby(['graph_mode', 'horizon']):
        key = ['scenario_id', 'seed']
        validation = group.pivot(index=key, columns='model', values='validation_ap').reindex(columns=MODEL_NAMES)
        test = group.pivot(index=key, columns='model', values='test_ap').reindex(columns=MODEL_NAMES)
        descriptors = group.drop_duplicates(key).set_index(key).reindex(validation.index)
        # Test performance must never determine the selector's training population.
        complete = validation.notna().all(axis=1)
        validation, test, descriptors = validation[complete], test[complete], descriptors[complete]
        scenario_ids = validation.index.get_level_values('scenario_id').to_numpy()
        groups = descriptors['selection_group'].to_numpy() if 'selection_group' in descriptors else scenario_ids
        record = {'graph_mode': mode, 'horizon': int(horizon), 'feature_names': FEATURE_NAMES,
                  'model_names': MODEL_NAMES, 'gray_margin': gray_margin}
        if len(np.unique(groups)) < 4:
            outputs.append({**record, 'status': 'insufficient_scenarios', 'required': 4})
            continue
        x = descriptors[FEATURE_NAMES].to_numpy(float)
        y, yt = validation.to_numpy(float), test.to_numpy(float)
        train, held = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=37).split(x, groups=groups))
        medians = np.nanmedian(x[train], axis=0)
        medians = np.nan_to_num(medians)
        x = np.where(np.isfinite(x), x, medians)
        model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=2, random_state=37)
        model.fit(x[train], y[train])
        estimates = model.predict(x[held])
        chosen = estimates.argmax(axis=1)
        ordered = np.sort(estimates, axis=1)
        gray = ordered[:, -1] - ordered[:, -2] <= gray_margin
        actual = yt[held, chosen]
        best_fixed = int(y[train].mean(axis=0).argmax())
        validation_chosen = y[held].argmax(axis=1)
        # Test oracle is used ONLY as an evaluation upper bound.
        regret = yt[held].max(axis=1) - actual
        heldout = {'train_scenario_ids': sorted(set(scenario_ids[train])), 'heldout_scenario_ids': sorted(set(scenario_ids[held])),
                   'train_generation_groups': sorted(set(groups[train])), 'heldout_generation_groups': sorted(set(groups[held])),
                   'n_heldout_replicates': len(held), 'n_evaluable_test_ap': int(np.isfinite(actual).sum()),
                   'mean_test_ap': float(np.nanmean(actual)) if np.isfinite(actual).any() else None,
                   'mean_selection_regret': float(np.nanmean(regret)) if np.isfinite(regret).any() else None, 'gray_fraction': float(gray.mean()),
                   'best_fixed_model': MODEL_NAMES[best_fixed], 'best_fixed_test_ap': float(np.nanmean(yt[held, best_fixed])) if np.isfinite(yt[held, best_fixed]).any() else None,
                   'per_scenario_validation_selection_test_ap': float(np.nanmean(yt[held, validation_chosen])) if np.isfinite(yt[held, validation_chosen]).any() else None,
                   'scenario_results': [{'scenario_id': scenario_ids[idx], 'seed': int(validation.index[idx][1]),
                                         'selected': MODEL_NAMES[chosen[j]], 'test_ap': float(actual[j]),
                                         'regret': float(regret[j]), 'gray': bool(gray[j])} for j, idx in enumerate(held)]}
        # Freeze grouping assessment above; then fit deployable rule on all simulation
        # VALIDATION scores. Real case test results never enter this fit.
        model.fit(x, y)
        tree = model.tree_
        outputs.append({**record, 'status': 'fitted', 'heldout': heldout,
                        'training_medians': medians, 'feature_min': x.min(0), 'feature_max': x.max(0),
                        'text': export_text(model, feature_names=FEATURE_NAMES),
                        'tree': {'left': tree.children_left, 'right': tree.children_right,
                                 'feature': tree.feature, 'threshold': tree.threshold,
                                 'value': tree.value[:, :, 0]}})
    write_json(path, {'schema_version': 1, 'rules': outputs})
    return outputs


def apply_rule(rule, descriptors):
    if rule['status'] != 'fitted' or rule['graph_mode'] != 'estimated':
        raise ValueError('External application requires a fitted estimated-graph rule')
    x = np.array([descriptors.get(k, np.nan) for k in rule['feature_names']], float)
    missing = ~np.isfinite(x)
    x = np.where(missing, rule['training_medians'], x)
    outside = bool(missing.any() or np.any(x < rule['feature_min']) or np.any(x > rule['feature_max']))
    tree, node = rule['tree'], 0
    while tree['left'][node] != -1:
        node = tree['left'][node] if x[tree['feature'][node]] <= tree['threshold'][node] else tree['right'][node]
    scores = np.asarray(tree['value'][node])
    order = np.argsort(scores)[::-1]
    gray = float(scores[order[0]] - scores[order[1]]) <= rule['gray_margin']
    return {'selected_model': None if outside else rule['model_names'][order[0]],
            'candidate_models': [rule['model_names'][i] for i in order if scores[order[0]] - scores[i] <= rule['gray_margin']],
            'gray_region': gray, 'outside_simulation_domain': outside,
            'predicted_validation_ap': scores.tolist()}
