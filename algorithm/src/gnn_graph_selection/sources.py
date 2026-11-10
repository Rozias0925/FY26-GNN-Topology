"""Documented WRA v2 paginated downloads; immutable compressed raw pages."""
from __future__ import annotations
import gzip
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor
from .io import write_json

SOURCES = {
    'turbidity': {'id': '7cd352ef-6518-4396-b2d6-61f1946a5611', 'catalog': 'https://data.gov.tw/dataset/58688',
                  'title': '石門水庫濁度資料', 'time_field': 'monitordate'},
    'reservoir': {'id': '2be9044c-6e44-4856-aad5-dd108c2e6679', 'catalog': 'https://data.gov.tw/dataset/45501',
                  'title': '水庫水情資料', 'time_field': 'observationtime'},
    'river_level': {'id': '73c4c3de-4045-4765-abeb-89f9f9cd5ff0', 'catalog': 'https://data.gov.tw/dataset/25768',
                    'title': '即時水位資料', 'time_field': 'datetime'},
    'river_stations': {'id': 'c4acc691-7416-40ca-9464-292c0c00da92',
                       'catalog': 'https://opendata.wra.gov.tw/api/v2/openapi.get',
                       'title': '河川水位測站站況'},
    'rain_stations': {'id': '15c166e9-800f-4a81-ba60-0ba61b6c9975', 'catalog': 'https://data.gov.tw/dataset/32729',
                     'title': '水利署所屬雨量站基本資料'},
    'catchment_rain': {'id': '5e7c2b19-f3a4-4c6d-b812-9f1a4e3c7d55',
                      'catalog': 'https://opendata.wra.gov.tw/api/v2/openapi.get',
                      'title': '水庫集水區即時降雨量', 'time_field': 'observationtime',
                      'format': 'fews_pi'},
}


def fetch_bytes(url):
    for attempt in range(4):
        try:
            request = Request(url, headers={'User-Agent': 'GNN-Graph-Selection-Research/0.2', 'Accept': 'application/json'})
            with urlopen(request, timeout=90) as response:
                return response.read()
        except (OSError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def decode_records(raw):
    for encoding in ('utf-8-sig', 'cp950'):
        try:
            return json.loads(raw.decode(encoding))
        except (UnicodeError, json.JSONDecodeError):
            continue
    raise ValueError('API did not return valid UTF-8 or Big5 JSON (possibly an HTML error page)')


def archive_records(payload, spec):
    """Normalize the two actual official schemas; never reinterpret an error dict.

    FEWS-PI is an unpaginated snapshot with headers and event arrays, rather
    than a list of observations. Preserve accumulation semantics and flags.
    """
    if spec.get('format') != 'fews_pi':
        if not isinstance(payload, list) or any(not isinstance(r, dict) for r in payload):
            raise ValueError('Expected a list of official observation objects')
        return payload
    if not isinstance(payload, dict) or not isinstance(payload.get('timeSeries'), list):
        raise ValueError('Expected official FEWS-PI timeSeries data')
    from datetime import timedelta
    offset = float(payload['timeZone'])
    zone = timezone(timedelta(hours=offset))
    records = []
    for series in payload['timeSeries']:
        header = series['header']
        step = header.get('timeStep', {})
        for event in series['events']:
            stamp = datetime.fromisoformat(event['date'] + 'T' + event['time']).replace(tzinfo=zone)
            records.append({'locationid': header['locationId'], 'stationname': header.get('stationName'),
                            'observationtime': stamp.isoformat(), 'rainfall_raw': event['value'],
                            'quality_flag': event.get('flag'), 'units': header.get('units'),
                            'value_type': header.get('type'), 'time_step_unit': step.get('unit'),
                            'time_step_multiplier': step.get('multiplier'), 'missing_value': header.get('missVal'),
                            'latitude': header.get('lat'), 'longitude': header.get('lon')})
    return records


def download(source, root, max_pages=2000, page_size=1000, refresh=False, workers=1):
    """No format=JSON: WRA's export path is capped and can ignore pagination.

    Empty/short page marks transport completion, NOT historical completeness.
    Cached pages resume an archive. Use a new root or --refresh for new snapshots.
    """
    if source not in SOURCES or not 1 <= page_size <= 1000 or max_pages < 1 or not 1 <= workers <= 4:
        raise ValueError('Invalid source, page_size or max_pages')
    spec = SOURCES[source]
    folder = Path(root) / source
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / 'manifest.json'
    old = json.loads(manifest_path.read_text('utf-8')) if manifest_path.exists() else {}
    if old and old.get('page_size') != page_size:
        raise ValueError('Use a new raw directory when changing page size')
    seen = set()
    old_times = {r['page']: r['retrieved_at'] for r in old.get('pages', [])}
    pages = []
    row_count = 0
    manifest = {**spec, 'source': source, 'page_size': page_size,
                'license': 'Government Data Open License v1.0',
                'license_url': 'https://data.gov.tw/license',
                'started_at': old.get('started_at', datetime.now(timezone.utc).isoformat()),
                'download_complete': False, 'pages': pages}
    workers = 1 if spec.get('format') == 'fews_pi' else workers
    executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    pending = {}

    def fetch_page(number):
        url = 'https://opendata.wra.gov.tw/api/v2/' + spec['id'] + '?' + urlencode({'page': number, 'size': page_size})
        for attempt in range(4):
            raw = fetch_bytes(url)
            timestamp = datetime.now(timezone.utc).isoformat()
            try:
                data = archive_records(decode_records(raw), spec)
                return raw, timestamp, data
            except ValueError:
                if attempt == 3:
                    failure = folder / f'failed_page_{number:06d}_response.gz'
                    failure.write_bytes(gzip.compress(raw, mtime=0))
                    raise
                time.sleep(2 ** attempt)

    try:
        for page in range(1, max_pages + 1):
            # Bounded independent HTTP reads; only the main thread writes ordered
            # data pages and the manifest. At most four requests are in flight.
            if executor:
                for future_page in range(page, min(page + workers, max_pages + 1)):
                    cached = (folder / f'page_{future_page:06d}.json.gz').exists() and not refresh
                    if not cached and future_page not in pending:
                        pending[future_page] = executor.submit(fetch_page, future_page)
            url = 'https://opendata.wra.gov.tw/api/v2/' + spec['id'] + '?' + urlencode({'page': page, 'size': page_size})
            path = folder / f'page_{page:06d}.json.gz'
            cached = path.exists() and not refresh
            if cached:
                raw = gzip.decompress(path.read_bytes())
                timestamp = old_times.get(page)
                timestamp = timestamp or datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
                data = archive_records(decode_records(raw), spec)
            else:
                # The public API sometimes responds HTTP 200 with an HTML
                # maintenance page. Retry validation too, without caching it as data.
                raw, timestamp, data = pending.pop(page).result() if executor else fetch_page(page)
            sha = hashlib.sha256(raw).hexdigest()
            if data and sha in seen:
                raise ValueError('Repeated page: pagination may have been ignored; archive is incomplete')
            seen.add(sha)
            if not path.exists() or refresh:
                tmp = path.with_suffix('.tmp')
                tmp.write_bytes(gzip.compress(raw, mtime=0))
                tmp.replace(path)
            pages.append({'page': page, 'url': url, 'rows': len(data), 'sha256': sha, 'retrieved_at': timestamp, 'file': path.name})
            row_count += len(data)
            manifest.update(rows=row_count, last_checked_at=datetime.now(timezone.utc).isoformat())
            manifest['download_complete'] = spec.get('format') == 'fews_pi' or len(data) < page_size
            write_json(manifest_path, manifest)
            if page == 1 or (not cached and page % 25 == 0) or manifest['download_complete']:
                print(f'{source}: page {page}, {row_count} rows, complete={manifest["download_complete"]}', flush=True)
            if manifest['download_complete']:
                break
            if not cached:
                time.sleep(.08)
    except Exception as exc:
        manifest.update(error=str(exc), rows=row_count)
        write_json(manifest_path, manifest)
        raise
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    return manifest


def read_archive(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text('utf-8'))
    for page in manifest['pages']:
        raw = gzip.decompress((folder / page['file']).read_bytes())
        if hashlib.sha256(raw).hexdigest() != page['sha256']:
            raise ValueError(f'Checksum mismatch: {page["file"]}')
        yield from archive_records(decode_records(raw), manifest)


def archive_fingerprint(folder):
    """Content identity is unchanged by a cached archive's last-check timestamp."""
    manifest = json.loads((Path(folder) / 'manifest.json').read_text('utf-8'))
    from .io import digest
    return digest({'source':manifest['source'], 'id':manifest['id'], 'page_size':manifest['page_size'],
                   'pages':[{'page':p['page'],'rows':p['rows'],'sha256':p['sha256']} for p in manifest['pages']]})
