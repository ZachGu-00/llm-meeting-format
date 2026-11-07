"""Native BF16 sampled text followed by the unchanged calibrated R0 scorer."""
import argparse
from argparse import Namespace
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
from common import atomic_status, canonical, load, sha, verify_bindings, write_new
from dialogue_core import (digest, generation_messages, generation_seed, meeting_schedule,
                           normalized, render_readouts, close_meeting)


GENERATION = {'do_sample': True, 'temperature': .7, 'top_p': 1., 'top_k': 0,
    'typical_p': 1., 'min_p': None, 'epsilon_cutoff': 0., 'eta_cutoff': 0.,
    'num_beams': 1, 'num_return_sequences': 1, 'max_new_tokens': 128,
    'min_new_tokens': 0, 'min_length': 0, 'use_cache': True,
    'return_dict_in_generate': True, 'output_scores': False,
    'repetition_penalty': 1., 'no_repeat_ngram_size': 0,
    'forced_bos_token_id': None, 'forced_eos_token_id': None,
    'bad_words_ids': None, 'suppress_tokens': None, 'begin_suppress_tokens': None}


def generate(engine, inference, config, meeting, focal, state, memory, target):
    messages = generation_messages(config, meeting, focal, state, memory)
    rendered = inference.render_prompt(engine.tok, messages, engine.loader.model_kind,
                                       'original', 'R0', engine.loader.max_length - 128)
    overrides = {**GENERATION, 'pad_token_id': engine.tok.pad_token_id}
    identity = {'schema': 'voices_v3_native_utterance_v1', 'run_id': config['run_id'],
        'meeting': meeting, 'focal_agent_id': focal,
        'messages': messages, 'prompt_text': rendered['text'], 'prompt_token_ids': rendered['ids'],
        'prompt_text_sha256': rendered['sha256'], 'prompt_token_ids_sha256': rendered['ids_sha256'],
        'messages_sha256': rendered['messages_sha256'],
        'seed': generation_seed(config, meeting, focal), 'overrides': overrides,
        'model_generation_config_sha256': digest(engine.model.generation_config.to_dict()),
        'engine_sha256': digest(engine.identity), 'actual_generation_batch_size': 1,
        'free_generation_remedy': 'original', 'thinking_enabled': False,
        'memory_before_sha256': digest(memory), 'premeeting_state': state}
    target = Path(target)
    if target.exists():
        previous = load(target)
        if any(previous.get(k) != v for k, v in identity.items()):
            raise ValueError('Saved generation belongs to different native inputs')
        return previous
    torch = engine.torch; enc = engine.tensors([rendered['ids']]); width = len(rendered['ids'])
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(identity['seed']); torch.cuda.manual_seed_all(identity['seed'])
        with torch.inference_mode():
            sampled = engine.model.generate(**enc, generation_config=deepcopy(engine.model.generation_config), **overrides).sequences
    if (sampled.ndim != 2 or sampled.shape[0] != 1 or not width < sampled.shape[1] <= width + 128
        or not torch.equal(sampled[:, :width], enc['input_ids'])):
        raise ValueError('Native sampled generation returned unexpected token scope')
    tokens = sampled[0, width:].tolist()
    text = engine.tok.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    eos = engine.model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else (eos or [])
    result = {**identity, 'sampled_token_ids': tokens, 'sampled_token_ids_sha256': digest(tokens),
        'decoded_with_special_tokens': engine.tok.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False),
        'text': text, 'normalized_display_text': normalized(text), 'word_count': len(text.split()),
        'generated_token_count': len(tokens), 'terminated_by_eos': tokens[-1] in eos,
        'official_eos_token_ids': eos, 'reached_token_budget': len(tokens) == 128,
        'empty_display_text': not normalized(text), 'quality_retry': False,
        'probability_self_report_requested': False, 'status': 'complete_native_generation'}
    write_new(target, result)
    return result


def write_rows(path, rows):
    path = Path(path); data = ''.join(canonical(r) + '\n' for r in rows).encode()
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Generated readout input changed')
    else:
        with path.open('xb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())


def execute_run(model_plan, config, engine, inference):
    model = model_plan['model']; folder = Path(config['work_dir']); folder.mkdir(parents=True, exist_ok=True)
    write_new(folder / 'run_config.json', config)
    from analyze_v3 import read_closed
    schedule = meeting_schedule(config); state = list(config['initial_state']); memory = {}
    events = []; previous = None; files = {str(folder / 'run_config.json'): sha(folder / 'run_config.json')}
    for meeting in schedule:
        if shutil.disk_usage(folder).free < 12 * 1024**3 + 512 * 1024**2:
            raise OSError('Dialogue storage reserve reached; preserve all partial native evidence')
        work = folder / 'meetings' / f'{meeting["index"]:03d}'; work.mkdir(parents=True, exist_ok=True)
        generations = [generate(engine, inference, config, meeting, focal, state, memory,
            work / f'generation_{focal:02d}.json') for focal in meeting['members']]
        source = work / 'input.jsonl'; raw = work / 'raw.jsonl'
        rows = render_readouts(config, meeting, state, generations); write_rows(source, rows)
        if raw.exists():
            data = raw.read_bytes()
            if data and not data.endswith(b'\n'):
                raise ValueError('Truncated score record preserved; do not silently rerun it')
            prefix = [json.loads(line) for line in data.splitlines()]
            if len(prefix) != len(rows) and len(prefix) % model['batch_size']:
                raise ValueError('Partial native batch preserved; resuming would alter batch composition')
        args = Namespace(input=str(source), output=str(raw), calibration=model['calibration'],
            readouts='R0', batch_size=model['batch_size'], format_n=0, r2_n=0, progress_every=1000000)
        inference.run_score(engine, inference.normalized_items(source), args, load(model['calibration']))
        measured = list(read_closed(source, raw))
        event, next_state, next_memory = close_meeting(config, meeting, state, memory, generations,
                                                     measured, model_plan['kernels'])
        sources = {str(p): sha(p) for p in work.iterdir() if p.is_file() and p.name != 'completion.json'}
        closure = {'schema': 'voices_v3_dialogue_meeting_v1', 'status': 'complete_verified', 'event': event,
                   'run_config_sha256': digest(config), 'previous_meeting_sha256': previous,
                   'source_sha256': sources}
        write_new(work / 'completion.json', closure)
        events.append({'index': meeting['index'], 'path': str(work / 'completion.json'),
                       'sha256': sha(work / 'completion.json')})
        files.update(sources); files[str(work / 'completion.json')] = sha(work / 'completion.json')
        previous = sha(work / 'completion.json'); state, memory = next_state, next_memory
        print(json.dumps({'run_id': config['run_id'], 'completed_meetings': len(events), 'total_meetings': len(schedule)}), flush=True)
    result = {'schema': 'voices_v3_dialogue_work_completion_v1', 'status': 'complete_verified_native_run',
        'run_id': config['run_id'], 'model_plan_sha256': config['model_plan_sha256'],
        'run_config_sha256': digest(config), 'meetings': events, 'final_state': state,
        'final_memory_sha256': digest(memory), 'source_sha256': files}
    write_new(folder / 'run_completion.json', result)
    return result


def execute(plan_path, run_id=None):
    plan = load(plan_path); verify_bindings(plan['source_sha256'])
    if plan['schema'] != 'voices_v3_dialogue_model_plan_v1':
        raise ValueError('Unknown dialogue registration')
    if not os.environ.get('STY'):
        raise RuntimeError('Native dialogue requires persistent screen')
    base = Path(plan['base']); lease = load(base / 'setup/e1_e2_queue.lock')
    if lease.get('kind') != 'v3_E6_E7' or lease.get('pid') != os.getppid():
        raise RuntimeError('Dialogue child requires its parent to own the shared GPU lease')
    model = plan['model']; configs = [r for r in plan['runs'] if run_id is None or r['run_id'] == run_id]
    if not configs: raise ValueError('Requested run is not in the registration')
    sys.path.insert(0, str(base / 'code')); inference = importlib.import_module('inference_v3')
    from dialogue_bundle import archive_run, release_work
    engine = None; bundles = []; root = Path(plan['extension'])
    folder = root / 'results/dialogue' / model['slug'] / plan['experiment']; folder.mkdir(parents=True, exist_ok=True)
    for spec in configs:
        config = {**spec, 'model_plan': str(Path(plan_path).resolve()), 'model_plan_sha256': sha(plan_path)}
        expected = folder / 'work' / config['run_id']
        if Path(config['work_dir']).resolve() != expected:
            raise ValueError('Unexpected native run work directory')
        receipt = folder / 'bundles' / (config['run_id'] + '.tar.gz.json')
        if not receipt.exists():
            if engine is None:
                engine = inference.Engine(model['execution_model_dir'], model['repo_id'], model['revision'], model['max_length'],
                                          model['remedy'], model['model_kind'], 'eager')
                if engine.identity != model['engine'] or engine.validate_calibration(load(model['calibration'])) != model['option_ids']:
                    raise ValueError('Dialogue model differs from the frozen native E0 calibration')
            execute_run(plan, config, engine, inference)
        info = archive_run(plan_path, config['run_id']); bundles.append(info)
        # The independent off-server verifier writes these acknowledgements.
        # Revisit previous runs so transfer can overlap the next native run.
        for item in bundles:
            release = folder / 'released_work' / (item['run_id'] + '.json')
            ack = root / 'archives/dialogue_off_server_ack' / (item['archive_sha256'] + '.json')
            if not release.exists() and ack.exists(): release_work(item, load(ack), root)
        atomic_status(folder / 'native_status.json', {'status': 'running_serial_native_dialogue',
            'completed_runs': len(bundles), 'planned_runs': len(configs), 'last_run_id': config['run_id'],
            'model_plan_sha256': sha(plan_path), 'GPU_owner_parent_pid': os.getppid()})
    result = {'schema': 'voices_v3_dialogue_native_panel_v1', 'status': 'complete_native_runs_backups_may_be_pending',
        'model_plan_sha256': sha(plan_path), 'run_ids': [x['run_id'] for x in bundles],
        'bundles': bundles}
    if run_id is None: write_new(folder / 'native_completion.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--plan', required=True); p.add_argument('--run-id')
    a = p.parse_args(); print(json.dumps({'status': execute(a.plan, a.run_id)['status']}))
