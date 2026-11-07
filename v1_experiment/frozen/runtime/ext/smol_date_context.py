"""Explicit historic clock context for the unchanged official SmolLM3 template."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
from common import canonical, load, sha, verify_bindings, write_new
from dialogue_provenance import closure

REPO = 'HuggingFaceTB/SmolLM3-3B'
DATE = '2026-10-31'
FILES = ('smol_date_context.py', 'smol_date_inference_entry.py',
         'smol_date_analysis_entry.py', 'smol_date_queue_entry.py',
         'test_smol_date_context.py', 'SMOL_DATE_CONTEXT.md')


def build_policy(base, extension, historical_audit, calibration_audit):
    base, root = Path(base).resolve(), Path(extension).resolve()
    e9 = root / 'plans/e9_full/plan.json'; e7 = root / 'plans/e7/plan.json'
    plan = load(e9); model = next(m for m in plan['models'] if m['repo_id'] == REPO)
    jobs = [j for j in plan['jobs'] if j['model'] == REPO]
    if len(jobs) != 7 or any(Path(j['output'] + '.run.json').exists() for j in jobs):
        raise ValueError('Date registration must precede every SmolLM3 full-E9 native measurement')
    if load(e7.parent / 'status.json').get('GPU_work_started'):
        raise ValueError('E7 must still be a passive predecessor waiter')
    historical, calibration = load(historical_audit), load(calibration_audit)
    verify_bindings(historical['source_sha256']); verify_bindings(calibration['source_sha256'])
    if calibration['all_calibration_prompts_checked'] != 20:
        raise ValueError('All twenty original calibration prompts must be checked')
    matches = [d for d, checks in calibration['results'].items()
               if len(checks) == 20 and all(x['text_match'] and x['tokens_match'] for x in checks)]
    if matches != [DATE]: raise ValueError('The historic calibration clock is not uniquely identified')
    for record in historical['records']:
        checks = record['date_matches'][DATE]
        if not checks or not all(x['text_matches'] and x['tokens_match'] for x in checks):
            raise ValueError('Sampled historic E1 prompt hashes disagree with the calibration clock')
    paths = [e9, e7, root / 'plans/e9_storage_wait/plan.json', Path(historical_audit),
             Path(calibration_audit), Path(model['calibration']), base / 'code/inference_v3.py',
             base / 'code/loader_v3.py', root / 'common.py', root / 'dialogue_provenance.py']
    paths += [root / name for name in FILES]
    return {'schema': 'voices_v3_SmolLM3_historic_template_clock_v1',
            'created_utc': datetime.now(timezone.utc).isoformat(), 'base': str(base),
            'extension': str(root), 'repo_id': REPO, 'revision': model['revision'],
            'fixed_date': DATE, 'official_template_sha256': model['engine']['parent_loader']['template_sha256'],
            'model': model, 'E9_plan': str(e9), 'E7_plan': str(e7),
            'storage_policy': str(root / 'plans/e9_storage_wait/plan.json'),
            'E9_original_commands': [j['command'] for j in jobs],
            'E7_scope': 'SmolLM3 fast20 only; no manipulation/defense redesign',
            'native_numerical_methods_changed': False,
            'historic_data_rewritten': False,
            'source_sha256': {str(p): sha(p) for p in paths}}


def read_policy(registration):
    path = Path(registration).resolve(); policy = load(path)
    if (policy.get('schema') != 'voices_v3_SmolLM3_historic_template_clock_v1'
            or policy['repo_id'] != REPO or policy['fixed_date'] != DATE):
        raise ValueError('Unexpected template-clock registration')
    verify_bindings(policy['source_sha256'])
    return path, policy


def argument_map(command):
    if len(command) < 4 or len(command[2:]) % 2:
        raise ValueError('Expected the original scorer flag/value command')
    pairs = list(zip(command[2::2], command[3::2]))
    if any(not k.startswith('--') for k, _ in pairs) or len({k for k, _ in pairs}) != len(pairs):
        raise ValueError('Duplicate or malformed native scorer arguments')
    return dict(pairs)


def command_receipt(command, registration, stage, job=None):
    registration, policy = read_policy(registration); root, base = Path(policy['extension']), Path(policy['base'])
    command = list(command)
    if len(command) < 2 or command[1] != str(base / 'code/inference_v3.py'):
        return None
    args = argument_map(command)
    if args.get('--repo-id') != REPO: return None
    model = policy['model']
    for key, value in {'--mode': 'score', '--repo-id': REPO, '--revision': policy['revision'],
                       '--model-dir': model['execution_model_dir'], '--calibration': model['calibration'],
                       '--remedy': model['remedy'], '--model-kind': model['model_kind'],
                       '--attn-implementation': 'eager', '--max-length': str(model['max_length'])}.items():
        if args.get(key) != value: raise ValueError('SmolLM3 command identity differs: ' + key)
    if command[0] != sys.executable: raise ValueError('Native interpreter differs')
    sources = {str(registration): sha(registration), args['--calibration']: sha(args['--calibration'])}
    if stage == 'E9':
        if command not in policy['E9_original_commands']:
            raise ValueError('SmolLM3 E9 command is not exactly registered')
        source_plan = Path(policy['E9_plan'])
    elif stage == 'E7':
        source_plan = root / 'plans/e7/models' / model['slug'] / 'tasks.json'
        task = load(source_plan)
        if job != task['jobs']['fast20'] or job['name'] != 'fast20':
            raise ValueError('Only the original registered SmolLM3 E7 fast20 job is in scope')
        # Compare all task-controlled values directly, independent of a wrapper installed on that function.
        checks = {'--input': job['input'], '--output': job['output'], '--readouts': job['readout'],
                  '--batch-size': str(job['batch_size']), '--format-n': str(job['format_n']), '--r2-n': str(job['r2_n'])}
        if any(args.get(k) != v for k, v in checks.items()):
            raise ValueError('The E7 command differs from its frozen per-model task')
        if sha(job['input']) != job['input_sha256']: raise ValueError('E7 native input changed')
    else: raise ValueError('Unknown date-context stage')
    sources[str(source_plan)] = sha(source_plan); sources[args['--input']] = sha(args['--input'])
    value = {'schema': 'voices_v3_SmolLM3_date_bound_command_v1', 'stage': stage,
             'registration': str(registration), 'registration_sha256': sha(registration),
             'original_command': command, 'fixed_date': DATE, 'source_sha256': sources}
    digest = hashlib.sha256(canonical(value).encode()).hexdigest()
    path = root / 'provenance/smol_date_commands' / (digest + '.json')
    write_new(path, value)
    return path


def adapt_command(command, registration, stage, job=None):
    receipt = command_receipt(command, registration, stage, job)
    if receipt is None: return list(command)
    policy = load(registration)
    # Keep the original scorer path as a positional argument for the existing process scanner.
    return [command[0], str(Path(policy['extension']) / 'smol_date_inference_entry.py'),
            command[1], '--command-receipt', str(receipt), *command[2:]]


def install_clock(policy):
    from transformers.utils import chat_template_utils
    original = chat_template_utils._compile_jinja_template
    date = datetime.fromisoformat(policy['fixed_date']).replace(tzinfo=timezone.utc)
    previous = {}
    def compile_fixed(template):
        if hashlib.sha256(template.encode()).hexdigest() != policy['official_template_sha256']:
            raise ValueError('The official SmolLM3 template bytes changed')
        compiled = original(template)
        if id(compiled) not in previous:
            previous[id(compiled)] = (compiled, compiled.environment.globals['strftime_now'])
        compiled.environment.globals['strftime_now'] = lambda fmt: date.strftime(fmt)
        return compiled
    chat_template_utils._compile_jinja_template = compile_fixed
    def restore():
        chat_template_utils._compile_jinja_template = original
        for compiled, clock in previous.values(): compiled.environment.globals['strftime_now'] = clock
    return restore


def native_context(registration, command_path):
    registration, policy = read_policy(registration); command = load(command_path)
    if command['registration'] != str(registration) or command['registration_sha256'] != sha(registration):
        raise ValueError('Command date registration differs')
    verify_bindings(command['source_sha256'])
    return {'schema': 'voices_v3_declared_native_template_clock_v1', 'fixed_utc_date': policy['fixed_date'],
            'official_template_sha256': policy['official_template_sha256'],
            'registration': str(registration), 'registration_sha256': sha(registration),
            'command_receipt': str(command_path), 'command_receipt_sha256': sha(command_path),
            'native_numeric_methods_unchanged': True}, policy


def annotate_artifact(path, value, context):
    if value.get('mode') != 'score' or not str(path).endswith(('.run.json', '.summary.json')):
        return value
    if value.get('engine', {}).get('parent_loader', {}).get('repo_id') != REPO:
        raise ValueError('Date context cannot annotate a different model')
    result = deepcopy(value)
    if 'native_template_clock' in result and result['native_template_clock'] != context:
        raise ValueError('Existing native clock context differs')
    result['native_template_clock'] = deepcopy(context)
    sources = result.setdefault('source_sha256', {})
    for path_key, digest in ((context['registration'], context['registration_sha256']),
                             (context['command_receipt'], context['command_receipt_sha256'])):
        if path_key in sources and sources[path_key] != digest: raise ValueError('Conflicting clock provenance')
        sources[path_key] = digest
    return result


def annotate_analysis(value, registration):
    registration, policy = read_policy(registration)
    if value.get('repo_id') != REPO or value.get('status') != 'complete_verified':
        raise ValueError('Only closed SmolLM3 E9 analysis can receive this provenance supplement')
    plan = load(policy['E9_plan']); bindings = {str(registration): sha(registration)}
    for job in plan['jobs']:
        if job['model'] != REPO: continue
        run_path = Path(job['output'] + '.run.json'); run = load(run_path)
        clock = run.get('native_template_clock', {})
        if clock.get('registration_sha256') != sha(registration) or clock.get('fixed_utc_date') != DATE:
            raise ValueError('A SmolLM3 E9 run lacks the declared fixed historic clock')
        bindings[str(run_path)] = sha(run_path)
    result = deepcopy(value)
    result['source_sha256'].update(closure(bindings, policy['base'], policy['extension']))
    result['native_template_clock'] = {'registration': str(registration), 'sha256': sha(registration),
                                       'fixed_utc_date': DATE, 'historic_E0_and_sampled_E1_prompt_hashes_matched': True}
    return result
