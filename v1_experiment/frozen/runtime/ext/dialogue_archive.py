"""Shared source/analysis archives complement separately verified native run bundles."""
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
import tarfile
from common import atomic_status, load, sha
from dialogue_provenance import closure
from e9_archive import member_name


def archive(plan_path, analysis_path=None):
    plan = load(plan_path); root = Path(plan['extension']); target = Path(analysis_path) if analysis_path else Path(plan_path)
    bindings = closure({str(plan_path): sha(plan_path), str(target): sha(target), str(Path(__file__)): sha(__file__),
                        **plan['source_sha256']}, plan['base'], plan['extension'])
    # Native bundle bytes live in their own permanent archives, listed and
    # independently replay-verified before analysis; don't duplicate them here.
    if analysis_path:
        analysis = load(analysis_path)
        if analysis['status'] != 'complete_verified' or analysis['model_plan_sha256'] != sha(plan_path):
            raise ValueError('Dialogue analysis is not closed under this plan')
        if len(analysis['native_bundle_sha256']) != plan['planned_runs']:
            raise ValueError('Dialogue analysis does not bind every native run bundle')
        for path, digest in analysis['native_bundle_sha256'].items():
            if sha(path) != digest: raise ValueError('Permanent native bundle changed')
            ack = load(root / 'archives/dialogue_off_server_ack' / (digest + '.json'))
            if not ack.get('off_server_copy_verified') or ack['archive_sha256'] != digest:
                raise ValueError('Independent native raw archive missing')
    names = {member_name(path, plan): path for path in bindings}; kind = 'E6_analysis' if analysis_path else 'E6_shared_sources'
    if len(names) != len(bindings): raise ValueError('Ambiguous archive paths')
    index_path = root / 'archives/panel_index.json'; index = load(index_path)
    key = kind + '|' + plan['model']['repo_id'] + '|' + sha(target)
    if key in index['archives']:
        info = index['archives'][key]
        if sha(info['archive']) != info['archive_sha256']: raise ValueError('Dialogue metadata archive changed')
        return info
    if shutil.disk_usage(root).free - sum(Path(p).stat().st_size for p in bindings) < 12 * 1024**3:
        raise OSError('Conservative shared-source archive estimate violates the storage reserve')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = root / 'archives' / (stamp + '_' + kind + '_' + plan['model']['slug'] + '.tar.gz')
    receipt_member = 'extension/archives/' + path.stem + '_receipt.json'
    receipt = {'schema': 'voices_v3_extension_evidence_archive_v1', 'kind': kind,
        'repo_id': plan['model']['repo_id'], 'analysis': str(target), 'analysis_sha256': sha(target),
        'roots': {'original': plan['base'], 'extension': plan['extension']}, 'source_sha256': bindings,
        'files': {name: {'sha256': bindings[source], 'bytes': Path(source).stat().st_size} for name, source in names.items()},
        'scope': 'shared frozen sources and model analysis; native E6 bytes are separately SHA-bound run bundles with independent replay acknowledgements'}
    encoded = (json.dumps(receipt, ensure_ascii=False, indent=2) + '\n').encode()
    stamps = {p: (Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in bindings}
    with tarfile.open(path, 'x:gz', compresslevel=1) as tar:
        for name, source in sorted(names.items()): tar.add(source, arcname=name, recursive=False)
        item = tarfile.TarInfo(receipt_member); item.size = len(encoded); tar.addfile(item, io.BytesIO(encoded))
    if stamps != {p: (Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in bindings}:
        raise ValueError('Dialogue source changed during archive')
    info = {'archive': str(path), 'archive_bytes': path.stat().st_size, 'archive_sha256': sha(path),
        'analysis_sha256': sha(target), 'repo_id': plan['model']['repo_id'], 'receipt_member': receipt_member,
        'source_file_count': len(bindings)}
    index['archives'][key] = info; atomic_status(index_path, index)
    return info
