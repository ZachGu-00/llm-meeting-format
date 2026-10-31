"""Sent over authenticated SSH stdin by control.py; no API credentials."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    path = Path('/proc') / str(pid)
    argv = [v.decode(errors='replace') for v in (path / 'cmdline').read_bytes().split(b'\0') if v]
    status = (path / 'stat').read_text().rsplit(')', 1)[1].split()[0]
    return argv, status


def main(payload):
    state = payload['state']
    base = Path(state['server']['root']).resolve()
    phase = read(base / 'setup/e4_phase_status.json')
    bridge = read(base / 'setup/after_prerequisite_status.json')
    action = payload['action']
    if action == 'resume':
        pause = read(base / 'provenance/user_pause_E4_20261031.json')
        if phase['score_plan'] != state['E4']['score_plan'] or sha(phase['score_plan']) != state['E4']['score_plan_sha256']:
            raise ValueError('Exact frozen score plan changed; do not resume.')
        if pause['processes'] != state['E4']['pause']['processes']:
            raise ValueError('Pause receipt changed; inspect before resuming.')
        for row in pause['processes']:
            argv, current = identity(row['pid'])
            if not any(Path(v).name == row['script'] for v in argv) or current != 'T':
                raise ValueError('Expected paused process missing or changed: ' + str(row['pid']))
        if pause['screen'] not in subprocess.run(['screen', '-ls'], capture_output=True, text=True).stdout:
            raise ValueError('Original screen is absent.')
        if not (base / 'results/Qwen--Qwen3.5-4B/e4_cache/V3_E4_TOPIC_01_R0.jsonl').is_file():
            raise ValueError('Partial score output missing.')
        for row in reversed(pause['processes']):
            os.kill(row['pid'], signal.SIGCONT)
        time.sleep(1)
        result = {'status': 'resumed_original_processes', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'score_plan': phase['score_plan'], 'processes': [{'pid': r['pid'], 'state': identity(r['pid'])[1]} for r in pause['processes']]}
        path = base / 'provenance' / ('cloud_resume_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
        path.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result))
        return
    if action == 'fetch-reports':
        paths = [base / 'reports/E0_E3_prerequisite_20261031.json', base / 'reports/E0_E3_prerequisite_20261031.md',
                 base / 'reports/E9_affine_all10_20261031.json', base / 'reports/E9_affine_all10_20261031.md',
                 base / 'provenance/user_pause_E4_20261031.json', base / 'provenance/nonselected_weight_cleanup_20261031.json']
        if phase.get('E4_complete') is True:
            final = Path(phase['report_json']).resolve()
            final.relative_to(base)
            if sha(final) != phase['report_sha256']:
                raise ValueError('Final report differs from phase receipt.')
            paths += [final, Path(phase['report_markdown'])]
        print(json.dumps({'status': 'report_transport_manifest', 'E4_complete': phase.get('E4_complete', False),
                          'scope': 'Reports only; raw outputs and dependency closure are separate backups.',
                          'files': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in paths]}))
        return
    live = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            argv, current = identity(int(proc.name))
            if not argv or not Path(argv[0]).name.startswith('python'):
                continue
            scripts = []
            for value in argv[1:]:
                if not value.endswith('.py'):
                    continue
                path = Path(value)
                if not path.is_absolute():
                    path = (proc / 'cwd').resolve() / path
                if path.resolve().is_relative_to(base):
                    scripts.append(path)
            if scripts:
                item = {'pid': int(proc.name), 'script': scripts[0].name, 'state': current}
                if '--output' in argv:
                    output = argv[argv.index('--output') + 1]
                    item['output'] = output
                    log = Path(output + '.log')
                    if log.exists():
                        with log.open('rb') as stream:
                            stream.seek(max(0, log.stat().st_size - 1800))
                            lines = stream.read().decode(errors='replace').splitlines()
                        for line in reversed(lines):
                            try:
                                value = json.loads(line)
                            except ValueError:
                                continue
                            if 'completed' in value:
                                item['progress'] = value
                                break
                live.append(item)
        except (OSError, ValueError):
            continue
    print(json.dumps({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'bridge_status': bridge['status'],
                      'phase_status': phase['status'], 'E9_complete': bridge['E9_complete'], 'E4_complete': phase['E4_complete'],
                      'score_plan': phase.get('score_plan'), 'score_plan_sha256': phase.get('score_plan_sha256'),
                      'processes': live, 'free_bytes': shutil.disk_usage(base).free,
                      'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader'], text=True).strip()}))


if __name__ == '__main__':
    main(PAYLOAD)
