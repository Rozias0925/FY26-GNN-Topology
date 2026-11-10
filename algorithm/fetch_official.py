"""Archive official WRA catalogues and verified datasets with provenance."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from gnn_graph_selection.sources import SOURCES, download, fetch_bytes
from gnn_graph_selection.io import write_json

OPENAPI = 'https://opendata.wra.gov.tw/api/v2/openapi.get'


def archive_metadata(root: Path):
    folder = root / 'metadata'
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / 'manifest.json'
    manifest = json.loads(manifest_path.read_text('utf-8')) if manifest_path.exists() else {}
    entries = manifest.get('files', {})
    urls = {'wra_openapi.json': OPENAPI}
    urls.update({f'{name}_catalog.html': s['catalog'] for name, s in SOURCES.items()
                 if s['catalog'].startswith('https://data.gov.tw/')})
    for name, url in urls.items():
        path = folder / name
        if path.exists() and name in entries:
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != entries[name]['sha256']:
                raise ValueError(f'Metadata checksum mismatch: {path}')
            continue
        try:
            raw = fetch_bytes(url)
            if name.endswith('.json'):
                json.loads(raw.decode('utf-8-sig'))
            path.write_bytes(raw)
            entries[name] = {'url': url, 'file': name, 'bytes': len(raw),
                             'sha256': hashlib.sha256(raw).hexdigest(),
                             'retrieved_at': datetime.now(timezone.utc).isoformat()}
            print(f'metadata: saved {name}', flush=True)
        except (OSError, ValueError) as exc:
            # Missing metadata is explicit; a failed download is never a valid archive.
            entries[name] = {'url': url, 'error': str(exc)}
            print(f'metadata: failed {name}: {exc}', flush=True)
        write_json(manifest_path, {'files': entries})
    path = folder / 'wra_openapi.json'
    if not path.exists():
        raise ValueError('Official OpenAPI is unavailable; cannot verify endpoint identities')
    return json.loads(path.read_text('utf-8-sig'))


def verify_source(spec, openapi):
    matches = [(path, entry) for path, entry in openapi.get('paths', {}).items()
               if spec['id'] in path]
    if not matches:
        raise ValueError(f"Unverified official endpoint: {spec['title']} ({spec['id']})")
    descriptions = json.dumps(matches, ensure_ascii=False)
    if spec['title'] not in descriptions:
        raise ValueError(f"Endpoint title does not match registry: {spec['title']}")
    return matches[0][0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True,
                   help='Use a new dated directory for a fresh official snapshot')
    p.add_argument('--sources', nargs='+', choices=list(SOURCES), default=['turbidity', 'reservoir'])
    p.add_argument('--metadata-only', action='store_true')
    p.add_argument('--max-pages', type=int, default=2000)
    p.add_argument('--workers', type=int, choices=[1, 2, 3, 4], default=3)
    a = p.parse_args()
    spec = archive_metadata(a.output)
    if a.metadata_only:
        return
    for name in a.sources:
        verify_source(SOURCES[name], spec)
        manifest = download(name, a.output, max_pages=a.max_pages, workers=a.workers)
        if not manifest['download_complete']:
            raise ValueError(f'{name}: page limit reached, resume download before analysis')


if __name__ == '__main__':
    main()
