"""Source-only compact detector and literal count-first defense inputs."""
import argparse
import copy
import json
from pathlib import Path
from common import canonical, load, sha, verify_bindings, write_new

SINGLE = 'Answer with a single letter.'
COUNT = "First list each participant's position and count the supporters and opponents. Then give your own position."
MASKS = (0, 1, 2, 4, 8, 16, 3, 5, 6, 9, 10, 12, 17, 18, 20, 24)
ORDERS = (0, 2, 5, 7)


def transform(source, condition):
    if condition not in ('baseline', 'count_first', 'fast20'): raise ValueError('Unknown E7 prompt transformation')
    row = copy.deepcopy(source); old = source.get('record_id', source['id'])
    suffix = '__E7_' + condition; row.update(id=old + suffix, record_id=old + suffix, source_record_id=old,
        experiment='E7d_compact' if condition == 'fast20' else 'E7c_count_first', E7_condition=condition,
        template_variant='E7_' + condition)
    if condition == 'count_first':
        message = row['messages'][-1]
        if message['role'] != 'user' or not message['content'].endswith(SINGLE):
            raise ValueError('Original final answer-format instruction changed')
        message['content'] = message['content'][:-len(SINGLE)] + COUNT + '\n' + SINGLE
    if 'prompt_hash' in row:
        from dialogue_core import digest
        row['prompt_hash'] = digest(row['messages'])
    return row


def save_rows(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise ValueError('Partial unregistered E7 input preserved for inspection')
    seen = set()
    with path.open('x') as f:
        for row in rows:
            if row['record_id'] in seen: raise ValueError('Duplicate E7 prompt identity')
            seen.add(row['record_id']); f.write(canonical(row) + '\n')
    return {'path': str(path), 'sha256': sha(path), 'rows': len(seen)}


def build(base, extension):
    base, root = Path(base).resolve(), Path(extension).resolve(); output = root / 'data/e7/prompts_manifest.json'
    if output.exists():
        previous = load(output); verify_bindings(previous['source_sha256'])
        verify_bindings({s['path']: s['sha256'] for s in previous['inputs'].values()}); return previous
    core_path = base / 'data/core_data_manifest.json'; core = load(core_path)
    semantic_path = base / 'data/e1_semantic64_prompts.jsonl'; full_path = base / 'data/e1_holdout200_prompts.jsonl'
    topics_path = base / 'data/e4_cache_manifest.json'; topics = sorted(load(topics_path)['topics'], key=lambda t: t['topic_id'])[:2]
    source_paths = [core_path, semantic_path, full_path, topics_path, Path(__file__), root / 'E7_PROTOCOL.md']
    for path in (semantic_path, full_path):
        if sha(path) != core['outputs'][path.name]['sha256']: raise ValueError('Original E1 source changed')
    semantic = [json.loads(line) for line in semantic_path.open()]
    if len(semantic) != 39808 or len({r['item_id'] for r in semantic}) != 64:
        raise ValueError('Expected complete registered semantic64 material')
    inputs = {}
    for condition in ('baseline', 'count_first'):
        inputs['E1_' + condition] = save_rows(output.parent / ('semantic64_' + condition + '.jsonl'),
                                             (transform(r, condition) for r in semantic))
    ids = set(core['reused20_ids']); fast = []
    for line in full_path.open():
        row = json.loads(line)
        if row['item_id'] in ids and row['subset_mask'] in MASKS and row['permutation'] in ((0,) if row['subset_mask'] == 0 else ORDERS):
            fast.append(transform(row, 'fast20'))
    expected = {(i, m, p, l) for i in ids for m in MASKS for p in ((0,) if m == 0 else ORDERS) for l in (0, 1)}
    actual = [(r['item_id'], r['subset_mask'], r['permutation'], r['label_map']) for r in fast]
    if len(fast) != 2440 or len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError('Compact detector must retain every registered item/subset/slot/label cell')
    inputs['fast20'] = save_rows(output.parent / 'fast20.jsonl', fast)
    for topic in topics:
        source = base / topic['output']['path']
        if sha(source) != topic['output']['sha256']: raise ValueError('Original E4 topic input changed')
        inputs['E4_count_first_' + topic['topic_id']] = save_rows(output.parent / (topic['topic_id'] + '_count_first.jsonl'),
            (transform(json.loads(line), 'count_first') for line in source.open()))
        source_paths.append(source)
    result = {'schema': 'voices_v3_E7_prompt_materials_v1', 'inputs': inputs, 'topic_ids': [t['topic_id'] for t in topics],
        'fast_item_ids': sorted(ids), 'fast_masks': list(MASKS), 'fast_nonempty_orders': list(ORDERS),
        'fast_empty_rows_per_item': 2, 'literal_instruction': COUNT, 'model_outcomes_read': False,
        'fast_execution': 'actual compact native scores; not merely retrospective extraction from E1 outputs',
        'source_sha256': {str(p): sha(p) for p in source_paths}}
    write_new(output, result); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--base', required=True); p.add_argument('--extension', required=True)
    a = p.parse_args(); result = build(a.base, a.extension)
    print({'inputs': {k: v['rows'] for k, v in result['inputs'].items()}})
