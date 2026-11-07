"""Restore missing checkpoint bytes into an isolated cache, preserving calibration identity."""
from pathlib import Path
import shutil
from common import load, sha, write_new

RESERVE = 12 * 1024 ** 3

def safe_relative(root, name):
    target = (Path(root) / name).resolve()
    if not target.is_relative_to(Path(root).resolve()) or target == Path(root).resolve():
        raise ValueError('Snapshot filename escapes the checkpoint root')
    return target

def verify_checkpoint(root, model, full_weights=False):
    root = Path(root)
    manifest = load(root / 'snapshot_manifest.json')
    if (manifest.get('repo_id') != model['repo_id'] or manifest.get('revision') != model['revision']
        or sha(root / 'snapshot_manifest.json') != model['engine']['parent_loader']['snapshot_manifest_sha256']
        or (root / 'RESEARCH_REVISION.txt').read_text().strip() != model['revision']):
        raise ValueError('Checkpoint identity differs from the frozen calibration')
    absent = []
    for name, spec in manifest['files'].items():
        path = safe_relative(root, name)
        if not path.is_file():
            absent.append(name); continue
        if path.stat().st_size != spec['bytes']:
            raise ValueError('Checkpoint file size differs: ' + name)
        if (full_weights or not name.endswith('.safetensors')) and sha(path) != spec['sha256']:
            raise ValueError('Checkpoint file hash differs: ' + name)
    return manifest, absent

def prepare_checkpoint(model, extension):
    original = Path(model['model_dir'])
    manifest, absent = verify_checkpoint(original, model)
    destination = Path(model['execution_model_dir'])
    if destination == original:
        if absent:
            raise FileNotFoundError('A previously retained checkpoint is now incomplete; inspect before restoring')
        return {'path': str(original), 'downloaded_bytes': 0, 'existing_original': True}
    expected = Path(extension).resolve() / 'model_cache' / model['slug']
    if destination.resolve() != expected:
        raise ValueError('Model cache is outside the registered isolated directory')
    if any(not name.endswith('.safetensors') for name in absent):
        raise ValueError('Original metadata is missing; cannot preserve frozen identity')
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('snapshot_manifest.json', 'RESEARCH_REVISION.txt', *manifest['files']):
        src, dst = safe_relative(original, name), safe_relative(destination, name)
        if name.endswith('.safetensors'):
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if sha(src) != sha(dst):
                raise ValueError('Existing cache metadata differs')
        else:
            with dst.open('xb') as out, src.open('rb') as inp:
                shutil.copyfileobj(inp, out)
    _, missing = verify_checkpoint(destination, model, full_weights=True)
    needed = sum(manifest['files'][name]['bytes'] for name in missing)
    if shutil.disk_usage(destination).free - needed < RESERVE + 1024 ** 3:
        raise OSError('Insufficient storage for this pinned checkpoint plus 12 GiB reserve and output allowance')
    token_path = Path('/workspace/.hf_home/token')
    token = token_path.read_text().strip() if token_path.is_file() else None
    from huggingface_hub import hf_hub_download
    for name in missing:
        if not name.endswith('.safetensors'):
            raise ValueError('Only missing pinned weights may be downloaded')
        try:
            path = Path(hf_hub_download(model['repo_id'], name, revision=model['revision'],
                                       local_dir=str(destination), token=token))
        except Exception as error:
            status = getattr(getattr(error, 'response', None), 'status_code', None)
            raise RuntimeError('Pinned download failed: ' + type(error).__name__ + '; HTTP ' + str(status)) from None
        spec = manifest['files'][name]
        if path.stat().st_size != spec['bytes'] or sha(path) != spec['sha256']:
            raise ValueError('Downloaded checkpoint differs from the original recorded SHA-256')
    _, missing = verify_checkpoint(destination, model, full_weights=True)
    if missing:
        raise ValueError('Restored checkpoint is incomplete')
    receipt = {'repo_id': model['repo_id'], 'revision': model['revision'], 'path': str(destination),
               'snapshot_manifest_sha256': sha(destination / 'snapshot_manifest.json'),
               'all_file_hashes_verified': True, 'original_files_modified': False}
    write_new(destination / 'extension_restore_receipt.json', receipt)
    return {**receipt, 'downloaded_bytes': needed, 'existing_original': False}
