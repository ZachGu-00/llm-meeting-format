"""Incremental evidence archives with independently reverified external byte dependencies."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
from common import atomic_status, load, sha, write_new
from evidence_archive import destination

SCHEMA = 'voices_v3_incremental_evidence_archive_v1'


def member_name(path, roots):
    p = Path(path); candidates = [(prefix, Path(root)) for prefix, root in roots.items() if p.is_relative_to(Path(root))]
    if len(candidates) != 1: raise ValueError('Unknown or ambiguous evidence root')
    prefix, root = candidates[0]; return prefix + '/' + p.relative_to(root).as_posix()


def create(base, extension, artifact, repo, kind, bindings, index_name='e7_index.json', reserve_bytes=12*1024**3):
    base, root, artifact = Path(base).resolve(), Path(extension).resolve(), Path(artifact).resolve()
    roots = {'original': str(base), 'extension': str(root)}; bindings = dict(bindings)
    if bindings.get(str(artifact)) != sha(artifact): raise ValueError('Archive root artifact is not hash-bound')
    for source, digest in bindings.items():
        member_name(source, roots)
        if sha(source) != digest: raise ValueError('Current evidence source changed')
    index_path = root / 'archives' / index_name
    index = load(index_path) if index_path.exists() else {'schema': 'voices_v3_incremental_archive_index_v1', 'archives': {}}
    index_key = kind + '|' + repo + '|' + sha(artifact)
    if index_key in index['archives']:
        info = index['archives'][index_key]
        if sha(info['archive']) != info['archive_sha256']: raise ValueError('Prior incremental archive changed')
        return info
    proven = {}; proofs = {}
    for path in sorted((root / 'archives/verified_source_manifests').glob('*.json')):
        proof = load(path); digest = proof['archive_sha256']; ack_path = root / 'archives/off_server_ack' / (digest + '.json')
        if proof['schema'] != 'voices_v3_verified_source_manifest_v1' or not ack_path.exists(): continue
        ack = load(ack_path)
        if (ack.get('schema') != 'voices_v3_off_server_backup_ack_v1' or not ack.get('all_member_and_source_hashes_verified') or
            any(ack.get(k) != proof[k] for k in ('archive_sha256', 'analysis_sha256', 'repo_id'))):
            raise ValueError('Prior evidence manifest lacks its exact independent acknowledgement')
        proofs[digest] = {'manifest': str(path), 'manifest_sha256': sha(path), 'ack': str(ack_path), 'ack_sha256': sha(ack_path)}
        for source, value in proof['source_sha256'].items():
            if bindings.get(source) == value and source not in proven: proven[source] = {'sha256': value, 'prior_archive_sha256': digest}
    used = {v['prior_archive_sha256'] for v in proven.values()}; proofs = {k: v for k, v in proofs.items() if k in used}
    for proof in proofs.values():
        for key in ('manifest', 'ack'):
            bindings[proof[key]] = proof[key + '_sha256']; proven.pop(proof[key], None)
    included = {member_name(p, roots): p for p in bindings if p not in proven}
    if len(included) != len(bindings) - len(proven): raise ValueError('Ambiguous archive source paths')
    if shutil.disk_usage(root).free - sum(Path(p).stat().st_size for p in included.values()) < reserve_bytes:
        raise OSError('Incremental evidence still violates storage reserve')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target = root / 'archives' / (stamp + '_' + kind + '_' + repo.replace('/', '--') + '.tar.gz'); target.parent.mkdir(parents=True, exist_ok=True)
    receipt_member = 'extension/archives/' + target.stem + '_receipt.json'
    receipt = {'schema': SCHEMA, 'roots': roots, 'repo_id': repo, 'kind': kind, 'analysis': str(artifact),
        'analysis_sha256': sha(artifact), 'source_sha256': bindings, 'external_sources': proven, 'prior_proofs': proofs,
        'files': {name: {'sha256': bindings[p], 'bytes': Path(p).stat().st_size} for name, p in included.items()},
        'scope': 'New evidence bytes plus exact SHA-bound dependencies already independently backed up; independent verifier must hash every external mirrored file again.'}
    encoded = (json.dumps(receipt, ensure_ascii=False, indent=2) + '\n').encode()
    stamps = {p: (Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in bindings}
    with tarfile.open(target, 'x:gz', compresslevel=1) as tar:
        for name, source in sorted(included.items()): tar.add(source, arcname=name, recursive=False)
        entry = tarfile.TarInfo(receipt_member); entry.size = len(encoded); tar.addfile(entry, io.BytesIO(encoded))
    if stamps != {p: (Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in bindings}:
        raise ValueError('Evidence changed during incremental archive')
    info = {'archive': str(target), 'archive_bytes': target.stat().st_size, 'archive_sha256': sha(target),
        'analysis_sha256': sha(artifact), 'repo_id': repo, 'receipt_member': receipt_member,
        'source_file_count': len(bindings), 'included_file_count': len(included), 'external_source_count': len(proven), 'archive_schema': SCHEMA}
    index['archives'][index_key] = info; atomic_status(index_path, index); return info


def verify_and_extract(archive, info, mirrors, roots):
    archive = Path(archive)
    if archive.stat().st_size != info['archive_bytes'] or sha(archive) != info['archive_sha256']:
        raise ValueError('Incremental archive transport identity mismatch')
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers(); names = [m.name for m in members]
        if len(set(names)) != len(names) or any(not m.isfile() for m in members): raise ValueError('Duplicate/nonregular archive entry')
        for name in names: destination(name, mirrors)
        receipt = json.load(tar.extractfile(info['receipt_member']))
        if receipt['schema'] != SCHEMA or receipt['roots'] != roots or any(receipt[k] != info[k] for k in ('analysis_sha256', 'repo_id')):
            raise ValueError('Incremental archive provenance differs')
        if set(names) != set(receipt['files']) | {info['receipt_member']}: raise ValueError('Unexpected or omitted archive member')
        sources = receipt['source_sha256']; external = receipt['external_sources']
        included_sources = {p for p in sources if p not in external}
        if {member_name(p, roots) for p in included_sources} != set(receipt['files']): raise ValueError('Unbound or omitted included evidence')
        if not set(external) <= set(sources): raise ValueError('Unbound external dependency')
        for member in members:
            if member.name == info['receipt_member']: continue
            spec = receipt['files'][member.name]
            if member.size != spec['bytes']: raise ValueError('Included evidence size differs')
            digest = hashlib.sha256()
            with tar.extractfile(member) as handle:
                for chunk in iter(lambda: handle.read(1024*1024), b''): digest.update(chunk)
            if digest.hexdigest() != spec['sha256']: raise ValueError('Included evidence SHA differs')
            target = destination(member.name, mirrors)
            if target.exists() and (not target.is_file() or sha(target) != spec['sha256']): raise ValueError('Existing mirror differs')
        for source in included_sources:
            if receipt['files'][member_name(source, roots)]['sha256'] != sources[source]: raise ValueError('Included source binding differs')
        loaded_proofs = {}
        for digest, proof in receipt['prior_proofs'].items():
            contents = {}
            for key in ('manifest', 'ack'):
                source = proof[key]
                if source not in included_sources or sources[source] != proof[key + '_sha256']: raise ValueError('Prior proof must be included and hash-bound')
                contents[key] = json.load(tar.extractfile(member_name(source, roots)))
            manifest, ack = contents['manifest'], contents['ack']
            if (manifest['schema'] != 'voices_v3_verified_source_manifest_v1' or manifest['archive_sha256'] != digest or
                ack.get('schema') != 'voices_v3_off_server_backup_ack_v1' or not ack.get('all_member_and_source_hashes_verified') or
                any(ack.get(k) != manifest[k] for k in ('archive_sha256', 'analysis_sha256', 'repo_id'))):
                raise ValueError('Invalid prior independent verification proof')
            loaded_proofs[digest] = manifest
        for source, spec in external.items():
            if spec['sha256'] != sources[source] or spec['prior_archive_sha256'] not in loaded_proofs:
                raise ValueError('External source lacks a matching declared proof')
            proof = loaded_proofs[spec['prior_archive_sha256']]
            if proof['source_sha256'].get(source) != sources[source]: raise ValueError('Prior proof did not verify this source version')
            target = destination(member_name(source, roots), mirrors)
            if not target.is_file() or sha(target) != sources[source]: raise ValueError('External mirrored evidence is absent or changed')
        # Install only after every new byte and every external dependency passes.
        for member in members:
            target = destination(member.name, mirrors)
            if member.name == info['receipt_member']: write_new(target, receipt); continue
            if target.exists(): continue
            target.parent.mkdir(parents=True, exist_ok=True); temporary = target.with_name(target.name + '.verified_delta_tmp')
            with tar.extractfile(member) as inp, temporary.open('xb') as out: shutil.copyfileobj(inp, out)
            if sha(temporary) != receipt['files'][member.name]['sha256']: raise ValueError('Mirrored write SHA differs')
            temporary.replace(target)
    return {'schema': 'voices_v3_off_server_backup_ack_v1', 'archive_schema': SCHEMA,
        'archive_sha256': info['archive_sha256'], 'analysis_sha256': info['analysis_sha256'], 'repo_id': info['repo_id'],
        'all_member_and_source_hashes_verified': True, 'verified_files': len(sources), 'external_sources_reverified': len(external),
        'local_archive': str(archive.resolve()), 'mirrors': {k: str(Path(v).resolve()) for k, v in mirrors.items()},
        'retention_note': 'All source bytes independently present and hash-verified in the mirror; immutable prior archives retained.'}
