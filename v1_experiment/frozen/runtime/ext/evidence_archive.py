"""Verify and mirror prefixed extension archives without trusting tar paths."""
from __future__ import annotations
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from common import load, sha, write_new

def destination(name, mirrors):
    p = PurePosixPath(name)
    if p.is_absolute() or not p.parts or any(x in ('', '.', '..') for x in p.parts):
        raise ValueError('Unsafe archive member')
    if p.parts[0] not in mirrors or len(p.parts) < 2:
        raise ValueError('Unknown evidence root')
    if any(x.startswith('.') for x in p.parts[1:]):
        raise ValueError('Hidden paths are not evidence')
    if p.suffix.lower() in ('.safetensors', '.bin', '.pt', '.pth', '.pem', '.key') or p.name.lower() in (
        'api.txt', 'hug.txt', 'token', 'id_rsa', 'id_ed25519'):
        raise ValueError('Credential/weight member forbidden')
    root = Path(mirrors[p.parts[0]]).resolve()
    target = root.joinpath(*p.parts[1:])
    cursor = target
    while cursor != root:
        if cursor.is_symlink(): raise ValueError('Symlink in mirror path')
        cursor = cursor.parent
    if not target.resolve().is_relative_to(root): raise ValueError('Mirror path escapes root')
    return target

def verify_and_extract(archive, info, mirrors, expected_roots):
    archive = Path(archive)
    if archive.stat().st_size != info['archive_bytes'] or sha(archive) != info['archive_sha256']:
        raise ValueError('Archive transport identity mismatch')
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        if len(set(names)) != len(names) or any(not m.isfile() for m in members):
            raise ValueError('Duplicate/nonregular archive entry')
        receipt_member = info['receipt_member']
        if receipt_member not in names: raise ValueError('Missing archive receipt')
        for m in members: destination(m.name, mirrors)
        receipt = json.load(tar.extractfile(receipt_member))
        if receipt['schema'] != 'voices_v3_extension_evidence_archive_v1' or receipt['roots'] != expected_roots:
            raise ValueError('Unknown archive provenance')
        if receipt['analysis_sha256'] != info['analysis_sha256'] or receipt['repo_id'] != info['repo_id']:
            raise ValueError('Analysis identity mismatch')
        if set(names) != set(receipt['files']) | {receipt_member}:
            raise ValueError('Unexpected/missing archive entries')
        # Validate all bytes before any final file is installed.
        for member in members:
            if member.name == receipt_member: continue
            spec = receipt['files'][member.name]
            if member.size != spec['bytes']: raise ValueError('Member size mismatch')
            import hashlib
            digest = hashlib.sha256()
            with tar.extractfile(member) as inp:
                for chunk in iter(lambda: inp.read(1024 * 1024), b''): digest.update(chunk)
            if digest.hexdigest() != spec['sha256']: raise ValueError('Member SHA mismatch')
            target = destination(member.name, mirrors)
            if target.exists() and (not target.is_file() or sha(target) != spec['sha256']):
                raise ValueError('Existing mirrored artifact differs: ' + member.name)
        for member in members:
            target = destination(member.name, mirrors)
            if member.name == receipt_member:
                write_new(target, receipt); continue
            if target.exists(): continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.verified_transfer_tmp')
            if temporary.exists(): raise ValueError('Incomplete mirror write needs inspection')
            with tar.extractfile(member) as inp, temporary.open('xb') as out: shutil.copyfileobj(inp, out)
            if sha(temporary) != receipt['files'][member.name]['sha256']:
                raise ValueError('Mirror storage verification failed')
            temporary.replace(target)
    source_names = {}
    for source, digest in receipt['source_sha256'].items():
        p = Path(source)
        candidates = [(prefix, Path(root)) for prefix, root in expected_roots.items() if p.is_relative_to(Path(root))]
        if len(candidates) != 1: raise ValueError('Unknown or ambiguous source root')
        prefix, root = candidates[0]; name = prefix + '/' + p.relative_to(root).as_posix()
        if name not in receipt['files'] or receipt['files'][name]['sha256'] != digest:
            raise ValueError('Transitive dependency omitted or changed')
        target = destination(name, mirrors)
        if sha(target) != digest: raise ValueError('Transitive source hash mismatch')
        source_names[name] = digest
    if set(source_names) != set(receipt['files']): raise ValueError('Unbound archive evidence')
    return {'schema': 'voices_v3_off_server_backup_ack_v1',
            'archive_sha256': info['archive_sha256'], 'analysis_sha256': info['analysis_sha256'],
            'repo_id': info['repo_id'], 'all_member_and_source_hashes_verified': True,
            'verified_files': len(source_names), 'local_archive': str(archive.resolve()),
            'mirrors': {k: str(Path(v).resolve()) for k, v in mirrors.items()},
            'retention_note': 'Additional Codex workspace copy; originals/raw retained. Only reproducible isolated weights may rotate.'}
