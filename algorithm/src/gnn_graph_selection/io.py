"""Strict JSON, atomic artifacts and content-addressed experiment provenance."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def digest(data):
    return hashlib.sha256(json.dumps(clean(data), sort_keys=True).encode()).hexdigest()


def source_digest():
    root = Path(__file__).resolve().parents[2]
    h = hashlib.sha256()
    paths = list(root.glob('*.py')) + list((root / 'src').rglob('*.py'))
    for p in sorted(paths):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()
