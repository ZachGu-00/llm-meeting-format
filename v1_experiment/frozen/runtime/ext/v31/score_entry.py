"""Exact registered native scorer, with the previously audited Smol template date."""
import argparse
import importlib
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(); p.add_argument('original_script'); p.add_argument('--plan', required=True)
    p.add_argument('--job-index', required=True, type=int); a, remaining = p.parse_known_args()
    root = Path(a.plan).parent; ext = root.parent
    sys.path.insert(0, str(ext))
    from common import load, sha, verify_bindings
    from smol_date_context import install_clock, annotate_artifact, read_policy, REPO
    plan = load(a.plan); verify_bindings(plan['source_sha256'])
    job = plan['jobs'][a.job_index]
    if [sys.executable, a.original_script, *remaining] != job['command'] or job['model'] != REPO:
        raise ValueError('Unregistered v3.1 native invocation')
    registration, policy = read_policy(plan['smol_clock_policy'])
    sys.path.insert(0, str(Path(plan['base']) / 'code'))
    native = importlib.import_module('inference_v3')
    methods = {n: getattr(native.Engine, n) for n in ('r0', 'r1', 'render', 'greedy', 'calibrate', 'validate_calibration', 'tensors')}
    context = {'schema': 'voices_v31_declared_template_clock_v1', 'fixed_utc_date': policy['fixed_date'],
               'official_template_sha256': policy['official_template_sha256'],
               'registration': str(registration), 'registration_sha256': sha(registration),
               'command_receipt': str(Path(a.plan)), 'command_receipt_sha256': sha(a.plan),
               'job_index': a.job_index, 'native_numeric_methods_unchanged': True}
    restore = install_clock(policy); original_write = native.write_new
    native.write_new = lambda path, value: original_write(path, annotate_artifact(path, value, context))
    try:
        if any(getattr(native.Engine, n) is not m for n, m in methods.items()):
            raise ValueError('Native numerical methods changed')
        sys.argv = [a.original_script, *remaining]; native.main()
    finally:
        native.write_new = original_write; restore()


if __name__ == '__main__': main()
