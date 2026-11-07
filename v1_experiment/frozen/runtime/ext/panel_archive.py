"""Archive a completed expanded checkpoint's complete registered evidence."""
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
import tarfile
from common import atomic_status, load, sha, verify_bindings
from e9_archive import member_name

def archive(plan_path, completion_path):
    plan = load(plan_path); completion = load(completion_path)
    root = Path(plan['extension']); folder = root / 'archives'; folder.mkdir(exist_ok=True)
    index_path = folder / 'panel_index.json'
    index = load(index_path) if index_path.exists() else {'schema': 'voices_v3_panel_archive_index_v1', 'archives': {}}
    key = plan['model']['repo_id'] + '|' + sha(completion_path)
    if key in index['archives']:
        info = index['archives'][key]
        if sha(info['archive']) != info['archive_sha256']: raise ValueError('Closed panel archive changed')
        return info
    if completion['status'] != 'complete_registered_available_scope' or completion['model'] != plan['model']['repo_id']:
        raise ValueError('Expanded model does not have a complete execution receipt')
    bindings = {**plan['source_sha256'], **completion['source_sha256'],
        str(plan_path): sha(plan_path), str(completion_path): sha(completion_path), str(Path(__file__)): sha(__file__)}
    verify_bindings(bindings)
    names = {member_name(path, plan): path for path in bindings}
    if len(names) != len(bindings): raise ValueError('Ambiguous archive names')
    estimated = sum(Path(p).stat().st_size for p in bindings)
    if shutil.disk_usage(root).free - estimated < 12 * 1024 ** 3:
        raise OSError('Conservative archive bound would violate the storage reserve; preserve evidence')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = folder / (stamp + '_expanded_' + plan['model']['slug'] + '.tar.gz')
    receipt_member = 'extension/archives/' + path.stem + '_receipt.json'
    receipt = {'schema': 'voices_v3_extension_evidence_archive_v1', 'kind': 'expanded_model',
        'repo_id': plan['model']['repo_id'], 'analysis': str(completion_path), 'analysis_sha256': sha(completion_path),
        'roots': {'original': plan['base'], 'extension': plan['extension']}, 'source_sha256': bindings,
        'files': {name: {'sha256': bindings[source], 'bytes': Path(source).stat().st_size} for name, source in names.items()},
        'scope': 'Registered E0 attempts, native scores, analyses, G6 where applicable and all source dependencies; no weights/credentials'}
    stamps = {p: (Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in bindings}
    data = (json.dumps(receipt, ensure_ascii=False, indent=2) + '\n').encode()
    with tarfile.open(path, 'x:gz', compresslevel=1) as tar:
        for name, source in sorted(names.items()): tar.add(source, arcname=name, recursive=False)
        item = tarfile.TarInfo(receipt_member); item.size = len(data); tar.addfile(item, io.BytesIO(data))
    if stamps != {p: (Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in bindings}:
        raise ValueError('Evidence changed during panel archive creation')
    info = {'archive': str(path), 'archive_sha256': sha(path), 'archive_bytes': path.stat().st_size,
            'analysis_sha256': sha(completion_path), 'repo_id': plan['model']['repo_id'],
            'receipt_member': receipt_member, 'source_file_count': len(bindings)}
    index['archives'][key] = info; atomic_status(index_path, index)
    return info
