"""Outcome-blind v3.1 M1 selection and native E4 prompt construction."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

SEED = 'voices-v31-M1-20261101-v1'


def rank(*parts):
    return hashlib.sha256((SEED + '|' + '|'.join(map(str, parts))).encode()).hexdigest()


def new_rows(builder, topic, meta):
    for row in builder.e4_rows(topic, meta):
        # Only metadata changes; the existing prompt renderer remains byte-for-byte.
        row.update(record_id=row['id'], experiment='E4_v31_new_topics', split='v31_sealed_new_E4_topics')
        yield row


def select(source, holdout, development, previous):
    old_topics = {x['topic'] for x in previous}
    hold_topics = {x['topic'] for x in holdout}
    dev_topics = {x['topic'] for x in development} - old_topics
    if len(hold_topics) != 19 or len(dev_topics) != 3 or hold_topics & old_topics:
        raise ValueError('Expected original nineteen holdout plus three unused development topics')
    used = {a['argument_id'] for x in holdout for a in [x['focal_argument'], *x['roles'].values()]}
    audit = []
    for topic in sorted(hold_topics | dev_topics):
        eligible = {str(s): [a for a in source if a['topic'] == topic and a['stance'] == s] for s in (-1, 1)}
        audit.append({'topic': topic, 'rank': rank('topic', topic),
                      'counts': {s: len(rows) for s, rows in eligible.items()},
                      'unused_E1_counts': {s: sum(a['argument_id'] not in used for a in rows) for s, rows in eligible.items()},
                      'eligible': all(len(rows) >= 15 for rows in eligible.values()),
                      'E1_topic_overlap_requires_LOTO_kernel': topic in hold_topics})
    chosen = sorted((a for a in audit if a['eligible']), key=lambda a: a['rank'])[:14]
    if len(chosen) < 8:
        raise ValueError('Fewer than eight eligible new topics; preserve audit and inspect')
    topics = []
    for i, item in enumerate(chosen):
        topic = item['topic']
        by = {s: sorted((a for a in source if a['topic'] == topic and a['stance'] == s),
                        key=lambda a: (a['argument_id'] in used, rank('argument', a['argument_id'])))[:15]
              for s in (-1, 1)}
        agents = [{'agent_id': j, 'arguments': {str(s): {**by[s][j], 'used_in_E1': by[s][j]['argument_id'] in used}
                                              for s in (-1, 1)}} for j in range(15)]
        if len({a['argument_id'] for agent in agents for a in agent['arguments'].values()}) != 30:
            raise ValueError('Repeated selected argument identity')
        topics.append({'topic_id': f'V31_E4_TOPIC_{i+1:02d}', 'topic': topic, 'agents': agents,
                       'E1_topic_overlap_requires_LOTO_kernel': item['E1_topic_overlap_requires_LOTO_kernel']})
    return audit, topics


def build(base, extension):
    base, ext = Path(base), Path(extension); root = ext / 'v31'
    sys.path[:0] = [str(ext), str(base / 'code')]
    from common import load, sha, verify_bindings, write_new
    from legacy_materials import load_builder
    from e4_graphs import validate_sts, pair_projection
    from e7_prompt_materials import save_rows
    target = root / 'data/M1_manifest.json'
    if target.exists():
        result = load(target); verify_bindings(result['source_sha256'])
        for spec in result['topics']:
            verify_bindings({spec['output']['path']: spec['output']['sha256']})
        return result
    builder, sources = load_builder(base, ext)
    source, raw_hashes = builder.source_arguments()
    old_manifest = base / 'data/e4_agent_manifest.json'
    old_hashes = load(old_manifest)['source_file_sha256']
    for old, digest in old_hashes.items():
        path = ext / 'data/argkp_source' / old.split('/kpa/', 1)[1]
        if sha(path) != digest: raise ValueError('Original ArgKP source differs')
        sources[str(path)] = digest
    for relative, digest in raw_hashes.items():
        sources[str(ext / relative)] = digest
    hold = base / 'data/e1_holdout200_items.json'; dev = base / 'data/development_items.json'
    prior = base / 'data/e4_agents.json'; sts = base / 'data/e4_graphs/sts15_all80.json'
    for path in [hold, dev, prior, sts, old_manifest, Path(__file__), root / 'research_plan_v31_user.md',
                 ext / 'e7_prompt_materials.py', base / 'code/e4_graphs.py']:
        sources[str(path)] = sha(path)
    graphs = load(sts)['systems']
    for g in graphs:
        validate_sts(g['blocks'])
    if len(graphs) != 80 or any(pair_projection(g['blocks']) != pair_projection(graphs[0]['blocks']) for g in graphs):
        raise ValueError('All eighty STS pair projections must agree')
    audit, agents = select(source, load(hold), load(dev), load(prior))
    audit_path = root / 'data/M1_topic_selection_v2.json'
    write_new(audit_path, {'seed': SEED, 'candidates': audit, 'selected_topics': [t['topic'] for t in agents],
                          'rule': 'SHA256 rank, first fourteen eligible; per stance unused E1 arguments first then SHA256',
                          'model_outputs_read': False, 'source_sha256': sources})
    agent_path = root / 'data/e4_agents_new.json'; write_new(agent_path, agents)
    sources.update({str(audit_path): sha(audit_path), str(agent_path): sha(agent_path)})
    results = []
    for topic in agents:
        meta = builder.e4_group_metadata(topic)
        if pair_projection(meta['g6_pasch_left']) != pair_projection(meta['g6_pasch_right']):
            raise ValueError('Pasch pair projections differ')
        path = root / 'data/e4_cache' / (topic['topic_id'] + '_prompts.jsonl')
        # Preserve a completed unregistered input on a failed material-build retry.
        if not path.exists():
            spec = save_rows(path, new_rows(builder, topic, meta))
        else:
            spec = {'path': str(path), 'sha256': sha(path), 'rows': sum(1 for _ in path.open())}
        validation = builder.validate_e4_topic(path, topic, meta)
        if spec['rows'] != 49020 or validation['status'] != 'pass': raise ValueError('Incomplete new E4 input')
        # Verify literal rendering on retries too, without reading any model outputs.
        with path.open() as handle:
            for expected, line in zip(new_rows(builder, topic, meta), handle, strict=True):
                if json.loads(line) != expected: raise ValueError('New prompt differs from the frozen native renderer')
        results.append({**meta, 'output': {**spec, 'n_rows': spec['rows'], 'bytes': path.stat().st_size},
                        'validation': validation,
                        'E1_topic_overlap_requires_LOTO_kernel': topic['E1_topic_overlap_requires_LOTO_kernel']})
        print(json.dumps({'topic_ready': topic['topic_id'], 'rows': spec['rows']}), flush=True)
    result = {'schema': 'voices_v31_M1_new_topics_v1', 'seed': SEED, 'topics': results,
              'n_topics': len(results), 'n_rows': len(results) * 49020, 'candidate_topics': len(audit),
              'model_outputs_read': False, 'all_80_STS_pair_projections_verified': True,
              'source_sha256': sources, 'kernel_policy': 'LOTO on E1-overlap topics; never fit to new E4 outcomes',
              'author_identity_limitation': 'ArgKP does not provide author identity; argument uniqueness is not author independence'}
    write_new(target, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--base', required=True); p.add_argument('--extension', required=True)
    a = p.parse_args(); result = build(a.base, a.extension)
    print(json.dumps({'status': 'M1_materials_ready', 'topics': result['n_topics'], 'rows': result['n_rows']}))
