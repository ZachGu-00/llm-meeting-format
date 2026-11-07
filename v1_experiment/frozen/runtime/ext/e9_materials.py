"""Deterministic full E9 materials; no model outcomes are consulted."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from common import canonical, load, sha, write_new

VARIANTS = ('baseline', 'member_labels', 'option_order', 'instruction_paraphrase')
PARAPHRASED_SYSTEM = ('You are taking part in a discussion of a policy proposal with a small group.\n'
                      'Consider the comments from the other participants and indicate your position now.')
SINGLE = 'Answer with a single letter.'

def transform(source, variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown template condition')
    row = copy.deepcopy(source)
    user = row['messages'][-1]['content']
    if not user.endswith(SINGLE):
        raise ValueError('Unexpected answer instruction')
    if variant == 'member_labels':
        user, count = re.subn(r'^Participant ([1-5]):', r'Member \1:', user, flags=re.M)
        if count != 5:
            raise ValueError('Expected exactly five participant slot labels')
    elif variant == 'option_order':
        a, b = row['candidates']['A'], row['candidates']['B']
        suffix = f'\n(A) {a}\n(B) {b}\n{SINGLE}'
        if not user.endswith(suffix):
            raise ValueError('Unexpected option block')
        user = user[:-len(suffix)] + f'\n(B) {b}\n(A) {a}\n{SINGLE}'
    elif variant == 'instruction_paraphrase':
        if len(row['messages']) != 2 or row['messages'][0]['role'] != 'system':
            raise ValueError('Expected original system/user message pair')
        row['messages'][0]['content'] = PARAPHRASED_SYSTEM
    row['messages'][-1]['content'] = user
    source_id = source.get('record_id', source['id'])
    row.update(id=source_id + '__E9_' + variant,
               source_record_id=source_id, experiment='E9_full',
               condition=variant, template_variant='E9_' + variant)
    if 'record_id' in row:
        row['record_id'] = row['id']
    if 'prompt_hash' in row:
        row['prompt_hash'] = hashlib.sha256(canonical(row['messages']).encode()).hexdigest()
    return row

def validate_panel(rows, item_ids):
    expected = {(i, m, p, l) for i in item_ids for m in range(32)
                for p in ([0] if m == 0 else range(10)) for l in (0, 1)}
    observed = [(r['item_id'], r['subset_mask'], r['permutation'], r['label_map']) for r in rows]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError('E9 requires complete fixed item/subset/slot/label coverage')
    if len({r.get('record_id', r['id']) for r in rows}) != len(rows):
        raise ValueError('Duplicate prompt IDs')
    by_cell = {key: r for key, r in zip(observed, rows)}
    for i, m, p, l in expected:
        if l == 0:
            a, b = by_cell[i, m, p, 0], by_cell[i, m, p, 1]
            if a['positive_letter'] == b['positive_letter'] or a['candidates']['A'] != b['candidates']['B']:
                raise ValueError('Label reversal is incomplete')
    return {'items': len(item_ids), 'rows': len(rows), 'full32': True,
            'empty_baseline_physical_replicates': 2, 'slot_orders': 10, 'label_maps': 2}

def build(base, extension):
    base, extension = Path(base).resolve(), Path(extension).resolve()
    folder = extension / 'data/e9_full'
    manifest_path = folder / 'manifest.json'
    if manifest_path.exists():
        old = load(manifest_path)
        from common import verify_bindings
        verify_bindings(old['source_sha256'])
        verify_bindings({v['path']: v['sha256'] for v in old['inputs'].values()})
        return old
    folder.mkdir(parents=True, exist_ok=True)
    core_path = base / 'data/core_data_manifest.json'
    items_path = base / 'data/e9_affine_balance/items20.json'
    renderer_path = base / 'data/e9_affine_balance/source_render_v2.py'
    source_path = base / 'data/e1_holdout200_prompts.jsonl'
    pilot = base / 'data/pilot7_prompts.jsonl'
    source_paths = [core_path, items_path, renderer_path, source_path, pilot,
                    Path(__file__), Path(__file__).with_name('common.py')]
    source_hashes = {str(p): sha(p) for p in source_paths}
    core, items = load(core_path), load(items_path)
    ids = sorted(core['reused20_ids'])
    if len(ids) != 20 or {i['item_id'] for i in items} != set(ids):
        raise ValueError('The original fixed twenty-item panel is required')
    if sha(source_path) != core['outputs'][source_path.name]['sha256']:
        raise ValueError('Original E1 prompt source differs from core manifest')
    rows = []
    with source_path.open() as f:
        for line in f:
            row = json.loads(line)
            if row['item_id'] in ids:
                rows.append(row)
    proof = validate_panel(rows, ids)
    spec = importlib.util.spec_from_file_location('frozen_renderer', renderer_path)
    renderer = importlib.util.module_from_spec(spec); spec.loader.exec_module(renderer)
    by_id = {i['item_id']: i for i in items}
    for row in rows:
        original = renderer.render(by_id[row['item_id']], row['subset_mask'], row['permutation'], row['label_map'])
        for key in ('messages', 'candidates', 'positive_letter'):
            if row[key] != original[key]:
                raise ValueError('Original row differs from pinned renderer')
    inputs = {}
    for variant in VARIANTS:
        changed = [transform(r, variant) for r in rows]
        validate_panel(changed, ids)
        path = folder / (variant + '.jsonl')
        content = ''.join(canonical(r) + '\n' for r in changed)
        if path.exists() and path.read_text() != content:
            raise ValueError('Different existing material preserved')
        if not path.exists():
            with path.open('x') as f: f.write(content)
        inputs[variant] = {'path': str(path), 'sha256': sha(path), 'rows': len(changed)}
    with pilot.open() as f: n_pilot = sum(1 for line in f if line.strip())
    if n_pilot != 1464:
        raise ValueError('Expected the original 1464-row numerical pilot')
    inputs['numerical_pilot'] = {'path': str(pilot), 'sha256': sha(pilot), 'rows': n_pilot}
    result = {'schema': 'voices_v3_E9_full_materials_v1', 'item_ids': ids, 'coverage': proof,
              'source_sha256': source_hashes, 'inputs': inputs, 'variants': list(VARIANTS),
              'read_model_outcomes': False, 'original_files_modified': False,
              'scope': 'previously-used diagnostic20; not fresh confirmation',
              'template_batch': 16, 'numerical_batches': [1, 4, 16],
              'numerical_batch16_rationale': 'Also compare the frozen production batch16 against batch1, as required by the execution amendment',
              'R2_definition': 'native greedy first token; categorical direction and non-answer rate, not logit/probability'}
    write_new(manifest_path, result)
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--base', required=True); p.add_argument('--extension', required=True)
    a = p.parse_args(); result = build(a.base, a.extension)
    print(json.dumps({'coverage': result['coverage'], 'inputs': result['inputs']}, indent=2))
