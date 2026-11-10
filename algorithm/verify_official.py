"""Verify official raw pages and completed analysis artifacts without changing them."""
import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from gnn_graph_selection.sources import SOURCES, archive_records, decode_records, archive_fingerprint
from gnn_graph_selection.io import write_json


def checked_file(folder, name):
    path = (folder / name).resolve()
    if path.parent != folder.resolve() or not path.is_file():
        raise ValueError(f'Invalid artifact path: {name}')
    return path


def verify(raw, analysis):
    counts = {}
    for name, spec in SOURCES.items():
        folder = raw / name
        manifest = json.loads((folder / 'manifest.json').read_text('utf-8'))
        if not manifest.get('download_complete') or manifest.get('error') or manifest['id'] != spec['id']:
            raise ValueError(f'{name}: incomplete or source identity mismatch')
        count = 0
        for number, page in enumerate(manifest['pages'], 1):
            if page['page'] != number or page['file'] != f'page_{number:06d}.json.gz':
                raise ValueError(f'{name}: nonsequential or unexpected page filename')
            payload = gzip.decompress(checked_file(folder, page['file']).read_bytes())
            if hashlib.sha256(payload).hexdigest() != page['sha256']:
                raise ValueError(f'{name}: raw checksum mismatch at page {number}')
            rows = archive_records(decode_records(payload), manifest)
            if len(rows) != page['rows']:
                raise ValueError(f'{name}: row count mismatch at page {number}')
            count += len(rows)
        if count != manifest['rows']:
            raise ValueError(f'{name}: total row count mismatch')
        counts[name] = count
        print(f'{name}: {count:,} raw rows verified', flush=True)
    metadata = raw / 'metadata'
    for name, entry in json.loads((metadata / 'manifest.json').read_text('utf-8'))['files'].items():
        if entry.get('error') or hashlib.sha256(checked_file(metadata, name).read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'{name}: metadata download/checksum failed')
    verified_files = 0
    for folder, marker in [(analysis, 'analysis_manifest.json'),
                           (analysis / 'full_history', 'profile_manifest.json')]:
        manifest = json.loads((folder / marker).read_text('utf-8'))
        if not manifest.get('completed'):
            raise ValueError(f'{folder}: analysis is incomplete')
        for name, expected in manifest['outputs_sha256'].items():
            if hashlib.sha256(checked_file(folder, name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'{folder}: output checksum mismatch: {name}')
            verified_files += 1
        print(f'{folder.name}: {len(manifest["outputs_sha256"])} output files verified', flush=True)
    main = json.loads((analysis / 'analysis_manifest.json').read_text('utf-8'))
    for name in counts:
        if archive_fingerprint(raw / name) != main['raw_source_fingerprints'][name]:
            raise ValueError(f'{name}: analysis points to another raw archive')
    history = json.loads((analysis / 'full_history/profile_manifest.json').read_text('utf-8'))
    if history['raw_fingerprint'] != archive_fingerprint(raw / 'turbidity'):
        raise ValueError('Full-history profile points to another turbidity archive')
    return dict(verified=True, verified_at=datetime.now(timezone.utc).isoformat(),
                raw_row_counts=counts, analysis_output_files=verified_files)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = verify(a.raw, a.analysis)
    if a.output:
        write_json(a.output, result)
    print('All raw, metadata and analysis integrity checks passed.')
