"""Archive complete E9 models and release only acknowledged, isolated weight caches."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
from common import atomic_status, load, sha, verify_bindings, write_new

def now(): return datetime.now(timezone.utc).isoformat()

def member_name(path, plan):
    p = Path(path)
    if p.is_symlink() or not p.is_file(): raise ValueError('Evidence must be a regular nonsymlink file')
    p = p.resolve()
    if p.suffix.lower() in ('.safetensors', '.bin', '.pt', '.pth', '.pem', '.key') or p.name.lower() in (
        'api.txt', 'hug.txt', 'token', 'id_rsa', 'id_ed25519'):
        raise ValueError('Credential or weight files cannot enter evidence archives')
    for prefix, root in [('original', Path(plan['base'])), ('extension', Path(plan['extension']))]:
        if p.is_relative_to(root):
            relative = p.relative_to(root)
            if any(s.startswith('.') for s in relative.parts): raise ValueError('Hidden credential/cache route forbidden')
            return prefix + '/' + str(relative)
    raise ValueError('Evidence lies outside registered project roots')

def archive_model(plan_path, model, analysis_entry):
    plan = load(plan_path); root = Path(plan['extension']); base = Path(plan['base'])
    sys.path[:0] = [str(base / 'setup'), str(base / 'code')]
    from e9_queue import score_closed
    analysis_path = Path(analysis_entry['analysis'])
    if sha(analysis_path) != analysis_entry['sha256']: raise ValueError('Completed analysis changed')
    result = load(analysis_path)
    if result.get('status') != 'complete_verified' or result['repo_id'] != model['repo_id']:
        raise ValueError('Incomplete model analysis')
    for job in plan['jobs']:
        if job['model'] == model['repo_id'] and score_closed(job, model) is None:
            raise ValueError('Incomplete score dependency')
    bindings = {**plan['frozen_file_sha256'], **result['source_sha256'],
                str(analysis_path): sha(analysis_path), str(Path(__file__).resolve()): sha(__file__)}
    verify_bindings(bindings)
    names = {member_name(path, plan): path for path in bindings}
    if len(names) != len(bindings): raise ValueError('Ambiguous evidence member names')
    size = sum(Path(path).stat().st_size for path in bindings)
    if shutil.disk_usage(root).free - size < 12 * 1024 ** 3:
        raise OSError('Conservative archive reserve would fall below 12 GiB')
    folder = root / 'archives'; folder.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = folder / (stamp + '_E9_' + model['slug'] + '.tar.gz')
    receipt_member = 'extension/archives/' + archive.stem + '_receipt.json'
    stamps = {path: (Path(path).stat().st_size, Path(path).stat().st_mtime_ns) for path in bindings}
    receipt = {'schema': 'voices_v3_extension_evidence_archive_v1', 'kind': 'E9_full_model',
        'repo_id': model['repo_id'], 'analysis': str(analysis_path), 'analysis_sha256': sha(analysis_path),
        'created_utc': now(), 'files': {name: {'sha256': bindings[path], 'bytes': Path(path).stat().st_size}
                                     for name, path in names.items()},
        'source_sha256': bindings, 'roots': {'original': str(base), 'extension': str(root)},
        'scope': 'Complete raw scores, analyses and transitive frozen plan/input/model-metadata dependencies; no weights or credentials'}
    encoded = (json.dumps(receipt, ensure_ascii=False, indent=2) + '\n').encode()
    with tarfile.open(archive, 'x:gz', compresslevel=1) as tar:
        for name, path in sorted(names.items()): tar.add(path, arcname=name, recursive=False)
        item = tarfile.TarInfo(receipt_member); item.size = len(encoded); tar.addfile(item, io.BytesIO(encoded))
    if stamps != {path: (Path(path).stat().st_size, Path(path).stat().st_mtime_ns) for path in bindings}:
        raise ValueError('Evidence changed during archival')
    return {'archive': str(archive), 'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size,
            'receipt_member': receipt_member, 'repo_id': model['repo_id'], 'analysis_sha256': sha(analysis_path),
            'source_file_count': len(bindings)}

def release_acknowledged_cache(plan, model, info, acknowledgement):
    root = Path(plan['extension']).resolve(); cache = Path(model['execution_model_dir']).resolve()
    if cache != root / 'model_cache' / model['slug']: return {'status': 'original_checkpoint_preserved'}
    ack = load(acknowledgement)
    if (ack.get('schema') != 'voices_v3_off_server_backup_ack_v1' or ack.get('archive_sha256') != info['archive_sha256']
        or ack.get('analysis_sha256') != info['analysis_sha256'] or ack.get('all_member_and_source_hashes_verified') is not True):
        raise ValueError('Cache release requires the exact verified off-server archive acknowledgement')
    for process in Path('/proc').glob('[0-9]*'):
        try: args = (process / 'cmdline').read_bytes().split(b'\0')
        except OSError: continue
        if str(cache).encode() in args: return {'status': 'deferred_cache_in_use'}
    from checkpoint import verify_checkpoint, safe_relative
    manifest, absent = verify_checkpoint(cache, model, full_weights=True)
    release_path = cache / 'cache_release_plan.json'
    weights = {name: spec for name, spec in manifest['files'].items() if name.endswith('.safetensors')}
    release = {'schema': 'voices_v3_regenerable_cache_release_v1', 'repo_id': model['repo_id'],
               'analysis_sha256': info['analysis_sha256'], 'off_server_archive_sha256': info['archive_sha256'],
               'weight_files': weights, 'original_model_directory_untouched': True,
               'raw_scores_and_all_metadata_preserved': True}
    if absent and not release_path.exists(): raise ValueError('Unrecorded missing cache files; inspect before rotation')
    write_new(release_path, release)
    for name, spec in weights.items():
        path = safe_relative(cache, name)
        if path.exists():
            if path.is_symlink() or sha(path) != spec['sha256']: raise ValueError('Unexpected cache file preserved')
            path.unlink()
    receipt = {**release, 'status': 'released_isolated_regenerable_weights',
               'bytes_released': sum(spec['bytes'] for spec in weights.values())}
    write_new(cache / 'cache_release_receipt.json', receipt)
    return receipt

def run(plan_path, once=False):
    plan_path = Path(plan_path).resolve(); plan = load(plan_path); root = Path(plan['extension'])
    folder = root / 'archives'; folder.mkdir(exist_ok=True)
    lock = (folder / 'keeper.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    index_path = folder / 'index.json'
    index = load(index_path) if index_path.exists() else {'schema': 'voices_v3_extension_archive_index_v1', 'archives': {}}
    models = {m['repo_id']: m for m in plan['models']}
    os.nice(10)
    while True:
        status = {'utc': now(), 'pid': os.getpid(), 'GPU_work_started': False, 'errors': []}
        try:
            state = load(plan_path.parent / 'status.json')
            if state['plan_sha256'] != sha(plan_path): raise ValueError('Dispatcher plan changed')
            for entry in state.get('models', []):
                model = models[entry['repo_id']]; key = model['repo_id'] + '|' + entry['sha256']
                if key not in index['archives']:
                    index['archives'][key] = archive_model(plan_path, model, entry)
                    index['updated_utc'] = now(); atomic_status(index_path, index)
                info = index['archives'][key]
                ack = folder / 'off_server_ack' / (info['archive_sha256'] + '.json')
                if ack.exists() and not info.get('cache_release'):
                    info['cache_release'] = release_acknowledged_cache(plan, model, info, ack)
                    if info['cache_release']['status'] == 'deferred_cache_in_use': del info['cache_release']
                    atomic_status(index_path, index)
            status['verified_server_archives'] = len(index['archives'])
            status['off_server_acknowledged'] = sum(bool(i.get('cache_release')) for i in index['archives'].values())
        except Exception as error:
            status['errors'].append({'type': type(error).__name__, 'error': str(error)})
        atomic_status(folder / 'keeper_status.json', status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
        if once: return status
        time.sleep(45)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--plan', required=True); p.add_argument('--once', action='store_true')
    a = p.parse_args(); run(a.plan, a.once)
