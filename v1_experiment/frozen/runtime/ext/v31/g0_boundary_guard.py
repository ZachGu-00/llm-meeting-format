"""Operational G0 boundary: leave frozen inference and its registration intact."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def process(pid):
    root = Path('/proc') / str(pid)
    try:
        stat = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'state': stat[0], 'ppid': int(stat[1]),
                'start_ticks': int(stat[19]),
                'command': (root / 'cmdline').read_text().replace('\0', ' ')}
    except FileNotFoundError:
        return None


def checked(spec):
    actual = process(spec['pid'])
    if actual is None or actual['start_ticks'] != spec['start_ticks']:
        raise RuntimeError('Registered process identity is absent or reused')
    if spec['command_contains'] not in actual['command']:
        raise RuntimeError('Registered process command changed')
    return actual


def status(path, value):
    value = {**value, 'utc': datetime.now(timezone.utc).isoformat(), 'guard_pid': os.getpid()}
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def stop_tree(plan):
    # Screen must stop first: otherwise it can continue a stopped foreground child.
    stopped = []
    for key in ('screen', 'manager'):
        p = checked(plan[key])
        os.kill(p['pid'], signal.SIGSTOP)
        stopped.append(p)
    parents = {plan['manager']['pid']}
    for _ in range(8):
        added = set()
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            p = process(int(entry.name))
            if p and p['ppid'] in parents and p['pid'] not in parents:
                # Recheck identity immediately before signalling.
                again = process(p['pid'])
                if again and again['start_ticks'] == p['start_ticks']:
                    os.kill(p['pid'], signal.SIGSTOP)
                    stopped.append(p)
                    added.add(p['pid'])
        if not added:
            break
        parents.update(added)
    time.sleep(0.1)
    for p in stopped:
        current = process(p['pid'])
        if current and current['start_ticks'] == p['start_ticks'] and current['state'] != 'T':
            raise RuntimeError('A registered process did not stop')
    return stopped


def boundary(plan, state):
    if state.get('current_model') != plan['model']:
        return 'unexpected_model_fail_closed'
    if state.get('status') == 'execution_error_not_model_rejection':
        return 'G0_execution_error_fail_closed'
    # The original controller writes completion before archiving a full model and
    # awaiting an independently verified backup. Intercept the preceding boundary.
    if state.get('status') in ('archiving_completed_model', 'waiting_for_verified_backup', 'advancing'):
        return 'G0_boundary_reached'
    path = Path(plan['completion'])
    if path.exists():
        try:
            done = json.loads(path.read_text())
        except json.JSONDecodeError:
            return None  # write_new may be in the middle of its single fsynced write
        if done.get('model') != plan['model'] or done.get('task_plan_sha256') != plan['task_sha256']:
            return 'completion_identity_error_fail_closed'
        return 'G0_boundary_reached'
    return None


def run(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    target = Path(plan['guard_status'])
    try:
        for p, digest in plan['source_sha256'].items():
            if sha(p) != digest:
                raise RuntimeError('Boundary source changed: ' + p)
        checked(plan['screen']); checked(plan['manager'])
        state = json.loads(Path(plan['controller_status']).read_text())
        if state.get('current_model') != plan['model']:
            raise RuntimeError('G0 is not the current model')
        status(target, {'status': 'armed', 'plan_sha256': sha(plan_path)})
        heartbeat = 0
        while True:
            checked(plan['screen']); checked(plan['manager'])
            state = json.loads(Path(plan['controller_status']).read_text())
            reason = boundary(plan, state)
            if reason:
                stopped = stop_tree(plan)
                status(target, {'status': reason, 'stopped': stopped,
                                'plan_sha256': sha(plan_path),
                                'controller_state_at_boundary': state,
                                'weights_preserved': True, 'next_GPU_queue_started': False})
                return
            if time.monotonic() - heartbeat > 10:
                status(target, {'status': 'armed_G0_running', 'plan_sha256': sha(plan_path),
                                'current_job': state.get('current_job')})
                heartbeat = time.monotonic()
            time.sleep(0.1)
    except BaseException as error:
        try:
            stopped = stop_tree(plan)
        except Exception as stopping:
            stopped = {'error_type': type(stopping).__name__}
        status(target, {'status': 'guard_error_fail_closed', 'error_type': type(error).__name__,
                        'stopped': stopped, 'next_GPU_queue_started': False})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--plan', required=True)
    run(p.parse_args().plan)
