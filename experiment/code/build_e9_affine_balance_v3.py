"""Source-only E9 supplement: complete pairwise slot balance without changing E1.

Only the missing affine permutations and two-member subsets are scored. Baseline
and singleton means retain their original measured duplicate-prompt averages.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROLES = ('A', 'Aprime', 'C', 'D', 'Dprime')
PAIR_MASKS = sorted((1 << a) | (1 << b) for a, b in itertools.combinations(range(5), 2))
REUSE_MASKS = [0, 1, 2, 4, 8, 16]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n')


def renderer(path):
    spec = importlib.util.spec_from_file_location('e9_original_renderer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert tuple(module.ROLES) == ROLES
    return module


def affine_order(a, b):
    if a not in (1, 2, 3, 4) or b not in range(5):
        raise ValueError('Affine parameters must be nonzero a and any b modulo five')
    return [ROLES[(a * slot + b) % 5] for slot in range(5)]


def additional_row(original, item, mask, a, b, label):
    row = original.render(item, mask, 0, label)
    order = affine_order(a, b)
    lines = []
    for slot, role in enumerate(order, 1):
        peer = item['roles'][role]
        body = (original.stance_phrase(peer['stance']) + ' ' + peer['argument']
                if mask & (1 << ROLES.index(role)) else '(has not commented yet)')
        lines.append(f'Participant {slot}: {body}')
    user = row['messages'][1]['content']
    prefix, rest = user.split('Comments from the other participants in this round:\n', 1)
    _, suffix = rest.split('\n\nWhat is your position now?', 1)
    row['messages'][1]['content'] = (prefix + 'Comments from the other participants in this round:\n'
                                    + '\n'.join(lines) + '\n\nWhat is your position now?' + suffix)
    permutation = 10 + (a - 2) * 5 + b
    row.update(id=f'{item["item_id"]}_E9_m{mask:02d}_a{a}_b{b}_l{label}',
               permutation=permutation, replicate_slot_indices=[permutation],
               experiment='E9_affine_balance', condition='additional_affine_a2_a3',
               affine_multiplier=a, affine_offset=b, slot_role_order=order,
               affine_convention='role_index_at_zero_based_slot_s=(a*s+b)%5',
               supplemental_only=True, main_E1_inputs_unchanged=True)
    return row


def message_hash(row):
    # Exactly the scorer's source text/options; omit record IDs and permutation metadata.
    payload = {key: row[key] for key in ('messages', 'candidates', 'positive_letter', 'template_variant')}
    return hashlib.sha256(canonical(payload).encode('utf-8')).hexdigest()


def multiset_hash(counter):
    return hashlib.sha256(canonical(sorted(counter.items())).encode('utf-8')).hexdigest()


def coverage_proof(original):
    old = original.permutations()
    new = [affine_order(a, b) for a in (2, 3) for b in range(5)]
    assert set(map(tuple, old)) == {tuple(affine_order(a, b)) for a in (1, 4) for b in range(5)}
    assert len(set(map(tuple, old + new))) == 20
    target = set(itertools.permutations(range(5), 2))
    result = {}
    for first, second in itertools.combinations(ROLES, 2):
        histograms = {}
        for name, orders in [('old10', old), ('additional10', new), ('combined20', old + new)]:
            counter = Counter((order.index(first), order.index(second)) for order in orders)
            histograms[name] = [{'first_slot': x, 'second_slot': y, 'count': n}
                                for (x, y), n in sorted(counter.items())]
            if name == 'combined20':
                assert set(counter) == target and set(counter.values()) == {1}
        result[first + '|' + second] = histograms
    return {'old10_equals_affine_multipliers_1_4': True,
            'all20_distinct': True, 'every_ordered_role_pair_covers_20_ordered_slot_pairs_once': True,
            'slot_indices_are_zero_based': True, 'per_pair_histograms': result}


def reuse_proof(original, items):
    records = []
    for item in items:
        for mask in REUSE_MASKS:
            old = Counter(message_hash(original.render(item, mask, p, label))
                          for p in range(10) for label in (0, 1))
            new = Counter(message_hash(additional_row(original, item, mask, a, b, label))
                          for a in (2, 3) for b in range(5) for label in (0, 1))
            assert old == new, (item['item_id'], mask)
            assert len(old) == (2 if mask == 0 else 10)
            assert set(old.values()) == ({10} if mask == 0 else {2})
            records.append({'item_id': item['item_id'], 'subset_mask': mask,
                            'old_logical_rows': 20, 'new_logical_rows': 20,
                            'distinct_source_prompts': len(old),
                            'old_multiset_sha256': multiset_hash(old),
                            'new_multiset_sha256': multiset_hash(new), 'equal': True})
    return {'all_equal': True, 'n_item_mask_checks': len(records), 'checks': records,
            'physical_old_empty_rows_per_item': 2,
            'physical_old_singleton_rows_per_item': 100,
            'new_reuse_prompts_are_not_scored_or_independent_replicates': True}


def verify_existing_source(path, original, by_id):
    seen = set()
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            row = json.loads(line)
            if row['item_id'] not in by_id or row['subset_mask'] not in REUSE_MASKS + PAIR_MASKS:
                continue
            key = (row['item_id'], row['subset_mask'], row['permutation'], row['label_map'])
            assert key not in seen
            seen.add(key)
            expected = original.render(by_id[row['item_id']], row['subset_mask'], row['permutation'], row['label_map'])
            assert row == expected, ('Original source disagrees with frozen renderer', key)
    expected = {(item, mask, p, label) for item in by_id for mask in REUSE_MASKS + PAIR_MASKS
                for p in ([0] if mask == 0 else range(10)) for label in (0, 1)}
    assert seen == expected, ('Missing original source cells', len(seen), len(expected))
    return len(seen)


def build(base=BASE):
    base = Path(base).resolve()
    folder = base / 'data/e9_affine_balance'
    destinations = [folder / name for name in ('additional_pair_prompts.jsonl', 'manifest.json',
                    'material_validation.json', 'items20.json', 'source_render_v2.py')]
    if any(path.exists() for path in destinations):
        raise FileExistsError('Refusing to overwrite any E9 material; inspect existing manifest')
    source_renderer = base.parent / 'voices_nonadditivity_v2_20261030/code/render_v2.py'
    original = renderer(source_renderer)
    sources = {'core_manifest': base / 'data/core_data_manifest.json',
               'holdout200_items': base / 'data/e1_holdout200_items.json',
               'e1_holdout200_prompts': base / 'data/e1_holdout200_prompts.jsonl',
               'e1_semantic64_prompts': base / 'data/e1_semantic64_prompts.jsonl',
               'amendment': base / 'provenance/E9_affine_pair_balance_20261031.txt'}
    source_hashes_before = {key: digest(path) for key, path in sources.items()}
    renderer_hash = digest(source_renderer)
    core = load(sources['core_manifest'])
    for key in ('e1_holdout200_prompts', 'e1_semantic64_prompts'):
        assert source_hashes_before[key] == core['outputs'][sources[key].name]['sha256']
    ids = sorted(core['reused20_ids'])
    assert len(ids) == len(set(ids)) == 20
    all_items = load(sources['holdout200_items'])
    by_id = {item['item_id']: item for item in all_items if item['item_id'] in ids}
    assert set(by_id) == set(ids)
    items = [by_id[item_id] for item_id in ids]
    coverage = coverage_proof(original)
    reuse = reuse_proof(original, items)
    checked = {key: verify_existing_source(sources[key], original, by_id)
               for key in ('e1_holdout200_prompts', 'e1_semantic64_prompts')}
    assert set(checked.values()) == {6040}
    folder.mkdir(parents=True, exist_ok=True)
    snapshot = folder / 'source_render_v2.py'
    with snapshot.open('xb') as stream:
        stream.write(source_renderer.read_bytes())
    assert digest(snapshot) == renderer_hash
    sources['renderer'] = snapshot
    write_new(folder / 'items20.json', items)
    output = folder / 'additional_pair_prompts.jsonl'
    n_rows = 0
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        for item, mask, a, b, label in itertools.product(items, PAIR_MASKS, (2, 3), range(5), (0, 1)):
            stream.write(canonical(additional_row(original, item, mask, a, b, label)) + '\n')
            n_rows += 1
    seen = set()
    with output.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            assert row['id'] not in seen
            seen.add(row['id'])
            assert row['subset_mask'].bit_count() == 2
            assert row == additional_row(original, by_id[row['item_id']], row['subset_mask'],
                                         row['affine_multiplier'], row['affine_offset'], row['label_map'])
            expected_stance = original.stance_phrase(-by_id[row['item_id']]['focal_stance'])
            assert row['candidates'][row['positive_letter']] == expected_stance
    assert n_rows == len(seen) == 4000
    assert source_hashes_before == {key: digest(sources[key]) for key in source_hashes_before}
    assert digest(source_renderer) == renderer_hash

    def receipt(path, rows=None):
        value = {'path': Path(path).relative_to(base).as_posix(),
                 'sha256': digest(path), 'bytes': Path(path).stat().st_size}
        if rows is not None:
            value['n_rows'] = rows
        return value

    proofs = {'pair_slot_coverage': coverage, 'baseline_singleton_message_multisets': reuse}
    validation = {'schema': 'voices_v3_E9_affine_pair_balance_validation_v1', 'status': 'pass',
                  'source_only_no_outcomes_read': True, 'original_files_unchanged': True,
                  'source_rows_verified_against_frozen_renderer': checked,
                  'new_unique_rows_checked': 4000, 'only_two_member_masks': True,
                  'label_orientation_verified_each_row': True, 'proofs': proofs,
                  'additional_prompts': receipt(output, n_rows),
                  'source_sha256': {key: digest(path) for key, path in sources.items()}}
    write_new(folder / 'material_validation.json', validation)
    manifest = {'schema': 'voices_v3_E9_affine_pair_balance_materials_v1',
                'builder': receipt(Path(__file__).resolve()),
                'amendment': receipt(sources['amendment']),
                'source_only_no_outcomes_read': True, 'supplemental_only': True,
                'main_E1_E2_selection_and_PK_unchanged': True,
                'item_ids': ids, 'items': receipt(folder / 'items20.json'),
                'item_selection': 'Exact sorted reused20_ids from original core_data_manifest; no outcome selection',
                'original_sources': {key: receipt(path) for key, path in sources.items()},
                'renderer_snapshot_is_byte_identical_to_original': True,
                'additional_prompts': receipt(output, n_rows),
                'validation': receipt(folder / 'material_validation.json'),
                'pair_masks': PAIR_MASKS, 'reuse_masks': REUSE_MASKS, 'roles': list(ROLES),
                'old_permutations': list(range(10)), 'new_permutations': list(range(10, 20)),
                'new_permutation_mapping': [{'permutation': 10 + (a - 2) * 5 + b,
                    'affine_multiplier': a, 'affine_offset': b, 'slot_role_order': affine_order(a, b)}
                    for a in (2, 3) for b in range(5)],
                'affine_convention': 'role_index_at_zero_based_slot_s=(a*s+b)%5',
                'proofs': proofs,
                'reuse_policy': {
                    'R0_raw_template': 'results/<slug>/e1/holdout200_R0.jsonl',
                    'R1_raw_template': 'results/<slug>/e1/semantic64_R1.jsonl',
                    'require_closed_original_raw_same_engine_remedy_batch_calibration_and_options': True,
                    'baseline': 'Use original two label rows, averaged after orientation; no new empty rows',
                    'singletons': 'Keep original average over all ten permutations and both labels, including original duplicate-score means',
                    'denominator': 'Same original singleton denominator in old10, additional10 and combined20',
                    'pairs': 'Old10 and additional10 each have two balanced labels per permutation; combined20 gives each of the twenty permutations equal weight',
                    'not_independent_replicates': 'Duplicated singleton/empty prompts are reused means, not new measured or independent observations',
                    'scope': 'Paired diagnostic on the same previously used twenty items; no fresh confirmatory cohort claim'},
                'cost': {'new_prompts_per_model': 4000, 'output_rows_if_R0_R1': 4000,
                         'scalar_readouts_if_R0_R1': 8000, 'new_empty_or_singleton_rows': 0,
                         'planned_models_R0_R1': 9, 'planned_models_R1_only': 1,
                         'planned_total_new_output_rows': 40000,
                         'planned_total_scalar_readouts': 76000}}
    write_new(folder / 'manifest.json', manifest)
    return {'manifest': receipt(folder / 'manifest.json'), 'additional_prompts': receipt(output, n_rows),
            'validation': receipt(folder / 'material_validation.json'), 'status': 'pass'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=BASE)
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.root), ensure_ascii=False, indent=2))
