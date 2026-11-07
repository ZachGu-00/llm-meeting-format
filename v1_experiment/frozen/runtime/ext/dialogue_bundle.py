"""Permanent native dialogue bundles, independent replay, and exact duplicate cleanup."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
from common import atomic_status, canonical, load, sha, write_new
from dialogue_core import (close_meeting, digest, generation_messages, generation_seed,
                           meeting_schedule, normalized, render_readouts)
from dialogue_provenance import VerifiedSources


def safe_member(name):
    p = PurePosixPath(name)
    if (p.is_absolute() or str(p) != name or not p.parts or any(x in ('.', '..', '') or x.startswith('.') for x in p.parts)
        or p.suffix not in ('.json', '.jsonl')):
        raise ValueError('Unsafe/non-evidence dialogue member')
    return p


def replay(get, names, external):
    """get returns exact member bytes; external verifies an absolute source SHA."""
    config = json.loads(get('run_config.json')); completion = json.loads(get('run_completion.json'))
    plan = json.loads(external(config['model_plan'], config['model_plan_sha256']))
    model = plan['model']; original = next(r for r in plan['runs'] if r['run_id'] == config['run_id'])
    if config != {**original, 'model_plan': config['model_plan'], 'model_plan_sha256': config['model_plan_sha256']}:
        raise ValueError('Run does not match its external registered model plan')
    if completion['status'] != 'complete_verified_native_run' or completion['run_id'] != config['run_id']:
        raise ValueError('Native work run is not closed')
    if completion['model_plan_sha256'] != config['model_plan_sha256'] or completion['run_config_sha256'] != digest(config):
        raise ValueError('Native work completion identity differs')
    for path, value in plan['source_sha256'].items():
        external(path, value)
    calibration = json.loads(external(model['calibration'], plan['source_sha256'][model['calibration']]))
    if calibration['engine'] != model['engine'] or calibration['option_ids'] != model['option_ids']:
        raise ValueError('External calibration differs')
    root = Path(config['work_dir']); used = {'run_config.json', 'run_completion.json'}
    state = list(config['initial_state']); memory = {}; previous = None; events = []; quality = []
    def check_sources(sources):
        for path, value in sources.items():
            p = Path(path)
            if not p.is_relative_to(root):
                raise ValueError('Work artifact references an unexpected external dependency')
            name = p.relative_to(root).as_posix(); safe_member(name)
            if hashlib.sha256(get(name)).hexdigest() != value:
                raise ValueError('Run source closure changed')
    for meeting in meeting_schedule(config):
        prefix = f'meetings/{meeting["index"]:03d}/'; generations = []
        for focal in meeting['members']:
            name = prefix + f'generation_{focal:02d}.json'; g = json.loads(get(name)); used.add(name)
            if (g['status'] != 'complete_native_generation' or g['meeting'] != meeting or
                g['run_id'] != config['run_id'] or g['focal_agent_id'] != focal or
                g['premeeting_state'] != state or g['memory_before_sha256'] != digest(memory) or
                g['messages'] != generation_messages(config, meeting, focal, state, memory) or
                g['messages_sha256'] != digest(g['messages']) or
                g['seed'] != generation_seed(config, meeting, focal) or g['quality_retry'] or
                g['actual_generation_batch_size'] != 1 or g['free_generation_remedy'] != 'original' or
                g['engine_sha256'] != digest(model['engine']) or g['thinking_enabled']):
                raise ValueError('Generation provenance, memory or random seed differs')
            if (g['prompt_text_sha256'] != hashlib.sha256(g['prompt_text'].encode()).hexdigest() or
                g['prompt_token_ids_sha256'] != digest(g['prompt_token_ids']) or
                g['sampled_token_ids_sha256'] != digest(g['sampled_token_ids']) or
                len(g['sampled_token_ids']) != g['generated_token_count'] or
                not 1 <= g['generated_token_count'] <= 128 or
                g['normalized_display_text'] != normalized(g['text']) or g['word_count'] != len(g['text'].split())):
                raise ValueError('Exact generated-token/text receipt changed')
            if (g['overrides'] != {**plan['generation_overrides'], 'pad_token_id': g['overrides']['pad_token_id']} or
                g['terminated_by_eos'] != (g['sampled_token_ids'][-1] in g['official_eos_token_ids']) or
                g['reached_token_budget'] != (g['generated_token_count'] == 128)):
                raise ValueError('Generation stopping or sampling policy differs')
            generations.append(g)
            quality.append({k: g[k] for k in ('word_count', 'generated_token_count', 'terminated_by_eos', 'reached_token_budget')})
            quality[-1]['empty_display_text'] = not normalized(g['text'])
        source_name = prefix + 'input.jsonl'; raw_name = prefix + 'raw.jsonl'
        source_bytes, raw_bytes = get(source_name), get(raw_name)
        if not source_bytes.endswith(b'\n') or not raw_bytes.endswith(b'\n'):
            raise ValueError('Truncated native JSONL evidence')
        source = [json.loads(line) for line in source_bytes.splitlines()]
        raw = [json.loads(line) for line in raw_bytes.splitlines()]
        run_bytes = get(raw_name + '.run.json'); run = json.loads(run_bytes)
        summary = json.loads(get(raw_name + '.summary.json')); run_sha = hashlib.sha256(run_bytes).hexdigest()
        wanted = render_readouts(config, meeting, state, generations)
        if source != wanted or len(raw) != 42:
            raise ValueError('Registered exact 42-row prompt panel changed')
        identity = {'engine': model['engine'], 'readout': 'R0', 'input_file_sha256': hashlib.sha256(source_bytes).hexdigest(),
            'input_rows_sha256': digest(source), 'input_record_ids_sha256': digest([r['record_id'] for r in source]),
            'n_input': 42, 'calibration_file_sha256': plan['source_sha256'][model['calibration']],
            'option_ids': model['option_ids'], 'batch_size': model['batch_size'], 'format_record_ids': [], 'r2_record_ids': []}
        if any(run.get(k) != v or summary.get(k) != v for k, v in identity.items()):
            raise ValueError('Calibrated native readout identity changed')
        if (summary['status'] != 'complete' or summary['n_output'] != 42 or summary['run_file_sha256'] != run_sha or
            summary['output_sha256'] != hashlib.sha256(raw_bytes).hexdigest() or summary['output_bytes'] != len(raw_bytes)):
            raise ValueError('Native readout byte/count closure differs')
        for i, row in enumerate(raw):
            expected_batch = min(model['batch_size'], 42 - i // model['batch_size'] * model['batch_size'])
            if (row.get('run_file_sha256') != run_sha or row.get('option_ids') != model['option_ids'] or
                row.get('actual_batch_size_R0') != expected_batch or row.get('batch_size') != model['batch_size']):
                raise ValueError('Native R0 batch or calibration changed')
        event, state, memory = close_meeting(config, meeting, state, memory, generations, raw, plan['kernels'])
        name = prefix + 'completion.json'; closure_bytes = get(name); closure = json.loads(closure_bytes)
        if (closure['event'] != event or closure['previous_meeting_sha256'] != previous or
            closure['status'] != 'complete_verified' or closure['run_config_sha256'] != digest(config)):
            raise ValueError('Meeting state, transcript or predecessor replay differs')
        check_sources(closure['source_sha256'])
        previous = hashlib.sha256(closure_bytes).hexdigest()
        expected_event = {'index': meeting['index'], 'path': str(root / name), 'sha256': previous}
        if completion['meetings'][meeting['index']] != expected_event:
            raise ValueError('Run meeting closure index differs')
        used.update((source_name, raw_name, raw_name + '.run.json', raw_name + '.summary.json', name)); events.append(event)
    if (len(completion['meetings']) != len(events) or completion['final_state'] != state or
        completion['final_memory_sha256'] != digest(memory) or used != set(names)):
        raise ValueError('Unregistered artifact, incomplete meeting schedule or final state')
    check_sources(completion['source_sha256'])
    expected_sources = {str(root / name) for name in used - {'run_completion.json'}}
    if set(completion['source_sha256']) != expected_sources:
        raise ValueError('Run does not bind every generation/readout byte')
    return {'config': config, 'model': model['repo_id'], 'events': events, 'final_state': state, 'generation_quality': quality,
            'source_closure_verified': True, 'state_and_memory_replayed': True}


local_external = VerifiedSources()


def verify_bundle(path, info, external=local_external, return_events=False):
    path = Path(path)
    if path.stat().st_size != info['archive_bytes'] or sha(path) != info['archive_sha256']:
        raise ValueError('Dialogue archive transport differs')
    with tarfile.open(path, 'r:gz') as tar:
        members = tar.getmembers(); names = [m.name for m in members]
        if len(names) != len(set(names)) or any(not m.isfile() for m in members):
            raise ValueError('Duplicate or nonregular dialogue archive member')
        for name in names: safe_member(name)
        manifest_bytes = tar.extractfile('bundle_manifest.json').read(); manifest = json.loads(manifest_bytes)
        if (manifest['schema'] != 'voices_v3_dialogue_bundle_v1' or manifest['run_id'] != info['run_id'] or
            manifest['model_plan_sha256'] != info['model_plan_sha256'] or
            hashlib.sha256(manifest_bytes).hexdigest() != info['manifest_sha256'] or
            set(names) != set(manifest['files']) | {'bundle_manifest.json'}):
            raise ValueError('Dialogue bundle manifest/identity differs')
        if sum(m.size for m in members) > 512 * 1024**2 or any(m.size > 32 * 1024**2 for m in members):
            raise ValueError('Unexpectedly large single-run evidence needs explicit inspection')
        # A run is at most a few dozen MiB; loading once avoids seeking through
        # compressed bytes for every later token/readout/state check.
        data = {}
        for name, spec in manifest['files'].items():
            if tar.getmember(name).size != spec['bytes']:
                raise ValueError('Dialogue archive member size differs')
            content = tar.extractfile(name).read()
            if len(content) != spec['bytes'] or hashlib.sha256(content).hexdigest() != spec['sha256']:
                raise ValueError('Dialogue bundle member changed')
            data[name] = content
    replayed = replay(data.__getitem__, set(data), external)
    ack = {'schema': 'voices_v3_dialogue_backup_ack_v1', 'run_id': info['run_id'],
        'model_plan_sha256': info['model_plan_sha256'], 'archive_sha256': info['archive_sha256'],
        'manifest_sha256': info['manifest_sha256'], 'archive_bytes': info['archive_bytes'],
        'files_verified': len(data), 'native_source_closure_verified': True,
        'state_and_memory_replayed': True, 'full_raw_bytes_retained_in_compressed_bundle': True}
    return (ack, replayed) if return_events else ack


def archive_run(plan_path, run_id):
    plan_path = Path(plan_path); plan = load(plan_path); root = Path(plan['extension'])
    config = next(r for r in plan['runs'] if r['run_id'] == run_id); work = Path(config['work_dir'])
    final = work.parent.parent / 'bundles' / (run_id + '.tar.gz')
    receipt = final.with_suffix(final.suffix + '.json')
    if receipt.exists():
        info = load(receipt); verify_bundle(final, info)
        if info['run_id'] != run_id or info['model_plan_sha256'] != sha(plan_path):
            raise ValueError('Existing bundle belongs to another run registration')
        publish_index(root, info); return info
    if final.exists():
        raise ValueError('Unclosed archive exists; preserve and inspect it')
    files = sorted(p for p in work.rglob('*') if p.is_file())
    if any(p.is_symlink() for p in work.rglob('*')):
        raise ValueError('Symlink in native dialogue work tree')
    specs = {}
    for path in files:
        name = path.relative_to(work).as_posix(); safe_member(name)
        specs[name] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    replay(lambda name: (work / name).read_bytes(), set(specs), local_external)
    manifest = {'schema': 'voices_v3_dialogue_bundle_v1', 'run_id': run_id, 'repo_id': plan['model']['repo_id'],
        'model_plan': str(plan_path), 'model_plan_sha256': sha(plan_path), 'work_dir': str(work),
        'files': specs, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'permanent_raw_format': 'all exact native uncompressed bytes retained as checked tar.gz members'}
    encoded = (canonical(manifest) + '\n').encode(); final.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(final, 'x:gz', compresslevel=1) as tar:
        for name in sorted(specs): tar.add(work / name, arcname=name, recursive=False)
        item = tarfile.TarInfo('bundle_manifest.json'); item.size = len(encoded); tar.addfile(item, io.BytesIO(encoded))
    info = {'archive': str(final), 'archive_bytes': final.stat().st_size, 'archive_sha256': sha(final),
        'run_id': run_id, 'model_plan': str(plan_path), 'model_plan_sha256': sha(plan_path),
        'manifest_sha256': hashlib.sha256(encoded).hexdigest(), 'repo_id': plan['model']['repo_id']}
    verify_bundle(final, info); write_new(receipt, info)
    publish_index(root, info)
    return info


def publish_index(root, info):
    index_path = Path(root) / 'archives/dialogue_index.json'; index_path.parent.mkdir(parents=True, exist_ok=True)
    index = load(index_path) if index_path.exists() else {'schema': 'voices_v3_dialogue_archive_index_v1', 'archives': {}}
    key = info['model_plan_sha256'] + '|' + info['run_id']
    if key in index['archives'] and index['archives'][key] != info:
        raise ValueError('Conflicting permanent dialogue run')
    index['archives'][key] = info; atomic_status(index_path, index)


def release_work(info, ack, extension):
    if (ack.get('schema') != 'voices_v3_dialogue_backup_ack_v1' or
        any(ack.get(k) != info[k] for k in ('run_id', 'model_plan_sha256', 'archive_sha256', 'manifest_sha256', 'archive_bytes')) or
        not ack.get('native_source_closure_verified') or not ack.get('state_and_memory_replayed') or
        not ack.get('full_raw_bytes_retained_in_compressed_bundle') or not ack.get('off_server_copy_verified')):
        raise ValueError('Independent verified backup acknowledgement required')
    archive = Path(info['archive']); root = Path(extension).resolve()
    if not archive.resolve().is_relative_to(root / 'results/dialogue') or sha(archive) != info['archive_sha256']:
        raise ValueError('Permanent archived raw bytes are absent or changed')
    with tarfile.open(archive, 'r:gz') as tar: manifest = json.load(tar.extractfile('bundle_manifest.json'))
    work = Path(manifest['work_dir'])
    if (work.resolve() != archive.parent.parent / 'work' / info['run_id'] or work.is_symlink() or
        hashlib.sha256((canonical(manifest) + '\n').encode()).hexdigest() != info['manifest_sha256']):
        raise ValueError('Work cleanup target/manifest differs')
    receipt = archive.parent.parent / 'released_work' / (info['run_id'] + '.json')
    if receipt.exists():
        old = load(receipt)
        if old['archive_sha256'] != info['archive_sha256']: raise ValueError('Cleanup receipt differs')
        return old
    allowed = manifest['files']; existing = {p.relative_to(work).as_posix(): p for p in work.rglob('*') if p.is_file()}
    if any(p.is_symlink() for p in work.rglob('*')) or not set(existing) <= set(allowed):
        raise ValueError('Unexpected file/symlink preserved in run work tree')
    for name, path in existing.items():
        safe_member(name)
        if sha(path) != allowed[name]['sha256'] or path.stat().st_size != allowed[name]['bytes']:
            raise ValueError('Changed duplicate preserved rather than removed')
    # A killed cleanup can resume: only exact remaining allowlisted duplicates.
    for path in existing.values(): path.unlink()
    for folder in sorted((p for p in work.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True): folder.rmdir()
    if work.exists(): work.rmdir()
    result = {'schema': 'voices_v3_dialogue_duplicate_release_v1', 'run_id': info['run_id'],
        'archive_sha256': info['archive_sha256'], 'manifest_sha256': info['manifest_sha256'],
        'off_server_ack_sha256': digest(ack), 'raw_bytes_preserved_compressed': True,
        'allowlisted_uncompressed_bytes': sum(s['bytes'] for s in allowed.values())}
    write_new(receipt, result); return result
