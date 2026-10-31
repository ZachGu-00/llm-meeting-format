"""Read-only tensor reclamation preview; never deletes or loads model weights.

E9 completion and verified local delivery are operator prerequisites, not claims
made by this tool. A successful preview is a point-in-time inventory, not a lock
preventing another inference process from starting afterwards.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat


ROOT = Path(__file__).resolve().parents[1]
PROC = Path('/proc')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_path(value, root):
    text = str(value).replace('\\', '/')
    remote = '/workspace/voices_nonadditivity_v3'
    if text.startswith(remote + '/'):
        return root / text[len(remote) + 1:]
    path = Path(value)
    return path if path.is_absolute() else root / path


def regular_contained(path, root):
    """Check lexical and resolved containment and every intervening symlink."""
    path, root = Path(path), Path(root).resolve(strict=True)
    relative = path.relative_to(root)
    if not relative.parts:
        raise ValueError('A candidate must be strictly inside its model directory')
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('Symlink is forbidden: ' + str(current))
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError('Candidate escapes its model directory')
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError('Candidate is not a regular file: ' + str(path))
    return resolved


def validate_selection(root, selection_path):
    selection = read(selection_path)
    queue_path = root / 'setup/download_queue.json'
    queue = read(queue_path)
    by_repo = {entry['repo_id']: entry for entry in queue}
    if len(queue) != 10 or len(by_repo) != 10:
        raise ValueError('Current download queue must contain exactly ten unique models')
    selected = selection.get('selected_repos', [])
    if (selection.get('schema') != 'voices_v3_frozen_E4_model_selection_v1'
            or selection.get('status') != 'frozen' or selection.get('performed') is not True
            or selection.get('all_requested_E1_complete') is not True
            or not isinstance(selected, list) or len(selected) != 6
            or len(set(selected)) != 6 or not set(selected) <= set(by_repo)):
        raise ValueError('Selection must freeze exactly six distinct members of the current ten-model queue')
    panel_path = source_path(selection['panel_path'], root)
    if sha(panel_path) != selection['panel_sha256']:
        raise ValueError('Frozen selection panel SHA differs')
    panel = read(panel_path)
    if (panel.get('schema') != 'voices_v3_E1_panel_and_E4_selection_v1'
            or panel.get('all_requested_E1_complete') is not True
            or panel.get('queue_sha256') != sha(queue_path)
            or panel.get('E4_selection', {}).get('selected_repos') != selected
            or panel['E4_selection'].get('status') != 'frozen'
            or panel['E4_selection'].get('performed') is not True):
        raise ValueError('Frozen panel and selected model list or queue disagree')
    return selection, by_repo


def inventory_model(root, entry):
    repo = entry['repo_id']
    if len(repo.split('/')) != 2 or any(p in ('', '.', '..') for p in repo.split('/')) or '\\' in repo:
        raise ValueError('Invalid queue repository identifier')
    slug = repo.replace('/', '--')
    models = root / 'models'
    folder = models / slug
    if models.is_symlink() or folder.is_symlink() or not folder.is_dir():
        raise ValueError('Model directory is missing or a symlink')
    if models.resolve(strict=True).parent != root or folder.resolve(strict=True).parent != models.resolve():
        raise ValueError('Model directory is not an immediate contained models child')
    folder = folder.resolve()
    manifest_path = regular_contained(folder / 'snapshot_manifest.json', folder)
    manifest = read(manifest_path)
    state_path = root / 'results' / slug / 'e0/e0_driver_state.json'
    state = read(state_path)
    remedy = state['selected_remedy']
    if not isinstance(remedy, str) or remedy in ('', '.', '..') or '/' in remedy or '\\' in remedy:
        raise ValueError('Invalid E0 remedy directory')
    calibration_path = state_path.parent / remedy / 'token_calibration.json'
    if source_path(state['selected_calibration'], root).resolve() != calibration_path.resolve():
        raise ValueError('Selected E0 calibration path differs')
    calibration = read(calibration_path)
    loader = calibration['engine']['parent_loader']
    digest = sha(manifest_path)
    if (manifest.get('status') != 'complete' or state.get('status') != 'complete'
            or any(obj.get('repo_id') != repo or obj.get('revision') != entry['revision']
                   for obj in (manifest, state, loader))
            or loader.get('snapshot_manifest_sha256') != digest):
        raise ValueError('Manifest does not match the completed E0 model/revision/SHA')
    content = {key: value for key, value in calibration.items() if key != 'lock_content_sha256'}
    canonical = hashlib.sha256(json.dumps(content, ensure_ascii=False, sort_keys=True,
                                         separators=(',', ':')).encode()).hexdigest()
    if calibration.get('lock_content_sha256') != canonical:
        raise ValueError('E0 calibration lock content hash differs')
    snapshot = root / 'provenance/model_metadata_snapshot/models' / slug / 'snapshot_manifest.json'
    if snapshot.is_file() and sha(snapshot) != digest:
        raise ValueError('Preserved E0 metadata snapshot manifest SHA differs')
    files = []
    for name, metadata in manifest['files'].items():
        if PurePosixPath(name).suffix != '.safetensors':
            continue
        if ('\\' in name or ':' in name or name.startswith('/')
                or any(part in ('', '.', '..') for part in name.split('/'))):
            raise ValueError('Unsafe manifest tensor path: ' + name)
        path = regular_contained(folder / name, folder)
        expected = metadata['bytes']
        if (type(expected) is not int or expected < 0 or path.stat().st_size != expected
                or not isinstance(metadata.get('sha256'), str)
                or re.fullmatch(r'[0-9a-f]{64}', metadata['sha256']) is None):
            raise ValueError('Tensor size or manifest SHA invalid: ' + name)
        files.append({'path': str(path), 'manifest_name': name, 'bytes': expected,
                      'recorded_weight_sha256': metadata['sha256'], 'regular_non_symlink_verified': True})
    if not files:
        raise ValueError('No manifest-listed safetensors files')
    return {'repo_id': repo, 'revision': entry['revision'], 'model_dir': str(folder),
            'manifest': str(manifest_path), 'manifest_sha256': digest,
            'E0_calibration': str(calibration_path), 'E0_calibration_sha256': sha(calibration_path),
            'saved_metadata_snapshot_checked': snapshot.is_file(),
            'files': sorted(files, key=lambda row: row['path']), 'total_bytes': sum(f['bytes'] for f in files)}


def inspect_processes(model_dirs, proc_root=PROC):
    """Conservatively block any live command referring to a candidate directory.

    Only PID/executable/affected directory are returned; raw argv could contain
    credentials. No signals, GPU queries or child processes are used.
    """
    proc_root = Path(proc_root)
    if not proc_root.is_dir():
        return {'status': 'unavailable', 'references': [], 'errors': ['/proc unavailable']}
    references, errors = [], []
    for process in sorted(proc_root.iterdir()):
        if not process.name.isdigit():
            continue
        try:
            args = [arg.decode('utf-8', errors='replace') for arg in (process / 'cmdline').read_bytes().split(b'\0') if arg]
            if not args:
                continue
            matched = set()
            for folder in model_dirs:
                pattern = re.escape(str(folder).replace('\\', '/')) + r"(?=$|[/\s'\"\\])"
                if any(re.search(pattern, arg.replace('\\', '/')) for arg in args):
                    matched.add(str(folder))
            # Also resolve relative --model-dir paths against that process cwd.
            for index, arg in enumerate(args):
                value = (args[index + 1] if arg == '--model-dir' and index + 1 < len(args)
                         else arg.split('=', 1)[1] if arg.startswith('--model-dir=') else None)
                if value is not None:
                    path = Path(value)
                    if not path.is_absolute():
                        path = (process / 'cwd').resolve(strict=True) / path
                    resolved = path.resolve()
                    matched.update(str(folder) for folder in model_dirs if resolved == Path(folder))
            if matched:
                references.append({'pid': int(process.name), 'executable': Path(args[0]).name,
                                   'candidate_model_dirs': sorted(matched)})
        except FileNotFoundError:
            if process.exists():
                errors.append('Unreadable process metadata for live PID ' + process.name)
        except (OSError, ValueError) as error:
            errors.append('Cannot inspect PID ' + process.name + ': ' + type(error).__name__)
    return {'status': 'incomplete' if errors else 'references_present' if references else 'no_references_observed',
            'references': references, 'errors': errors}


def build(root, selection_path, proc_root=PROC):
    root, selection_path = Path(root).resolve(strict=True), Path(selection_path).resolve(strict=True)
    result = {'schema': 'voices_v3_nonselected_weight_preview_v1',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'root': str(root),
              'selection': str(selection_path), 'selection_sha256': sha(selection_path),
              'models': [], 'blockers': [], 'deletion_performed': False,
              'E9_closure_and_local_backup_checked': False,
              'operator_prerequisites': ['All ten E9 scores/analyses closed and backed up locally',
                                         'Review exact frozen selection and this file list immediately before any action'],
              'preserve': 'All non-tensor files, especially original snapshot_manifest, model index, config, tokenizer, RESEARCH_REVISION, results and provenance',
              'scope': 'Point-in-time read-only preview; not a deletion approval, process lock or guarantee for later E6 inference',
              'weight_content_hash_recomputed': False}
    try:
        selection, queue = validate_selection(root, selection_path)
        selected = selection['selected_repos']
        unselected = sorted(set(queue) - set(selected))
        result.update(selected_repos=selected, unselected_repos=unselected)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result['blockers'].append({'scope': 'selection', 'reason': str(error)})
        unselected = []
    directories = []
    for repo in unselected:
        directories.append(root / 'models' / repo.replace('/', '--'))
        try:
            result['models'].append(inventory_model(root, queue[repo]))
        except (OSError, ValueError, KeyError, TypeError) as error:
            result['blockers'].append({'scope': repo, 'reason': str(error)})
    process = inspect_processes(directories, proc_root)
    result['process_check'] = process
    if process['status'] != 'no_references_observed':
        result['blockers'].append({'scope': 'processes', 'reason': process['status']})
    total = sum(model['total_bytes'] for model in result['models'])
    complete = not result['blockers'] and len(result['models']) == 4
    result.update(status='preview_complete_requires_operator_prerequisites' if complete else 'blocked',
                  inventory_complete=complete, total_validated_candidate_bytes=total,
                  total_candidate_files=sum(len(model['files']) for model in result['models']),
                  reclaimable_bytes_if_operator_prerequisites_met=total if complete else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.root, args.selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'status': result['status'], 'output': str(args.output.resolve()),
                      'total_validated_candidate_bytes': result['total_validated_candidate_bytes'],
                      'n_blockers': len(result['blockers']), 'deletion_performed': False}))
    return 0 if result['inventory_complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
