"""Materialize the Gate1-frozen G4 design policies without reading sealed effects."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys


def make_runs(model, topics, metas, contexts, extension, system_message):
    from dialogue_core import meeting_schedule
    result = []
    for context in contexts:
        run = context['run']; topic = topics[context['topic_id']]; meta = metas[context['topic_id']]
        for family in ('G6', 'STS15'):
            if family == 'G6':
                agents = meta['g6_agent_ids']; minority = context['G6_minority_count']; tau = context['G6_target_stance']
                designs = [{'design_id': 'left', 'blocks': meta['g6_pasch_left']},
                           {'design_id': 'right', 'blocks': meta['g6_pasch_right']}]
            else:
                agents = list(range(15)); minority = context['minority_count']; tau = context['target_stance']
                designs = context['designs']
            for ordinal, design in enumerate(designs):
                rid = f'E6v31_{family}_d{ordinal:02d}_r{run:02d}'
                config = {'run_id': rid, 'experiment': 'E6_v31', 'run_number': run, 'family': family,
                    'design_id': design['design_id'], 'design_ordinal': ordinal,
                    'source_system_id': design.get('source_system_id'), 'topic_id': topic['topic_id'], 'topic': topic['topic'],
                    'agent_ids': agents, 'agents': [a for a in topic['agents'] if a['agent_id'] in agents],
                    'initial_state': [tau if k < minority else -tau for k in range(len(agents))],
                    'minority_count': minority, 'target_stance': tau,
                    'mapping_index': context['mapping_index'] if family == 'STS15' else None,
                    'blocks': sorted([sorted(b) for b in design['blocks']]), 'rounds': 2,
                    'coupling_key': f'E6_v31|{family}|{topic["topic_id"]}|run{run}', 'system_message': system_message,
                    'work_dir': str(Path(extension) / 'results/dialogue' / model['slug'] / 'E6_v31/work' / rid)}
                meeting_schedule(config); result.append(config)
    if len(result) != 600 or len({r['run_id'] for r in result}) != 600:
        raise ValueError('G4 must contain 600 paired runs per model')
    for family, expected in [('G6', 2), ('STS15', 10)]:
        rows = [r for r in result if r['family'] == family]
        names = {r['design_id'] for r in rows}
        if len(names) != expected or {(r['run_number'], r['design_id']) for r in rows} != {(i, n) for i in range(50) for n in names}:
            raise ValueError('Incomplete paired policy panel')
    return result


def build(base, extension):
    base, ext = Path(base), Path(extension); root = ext / 'v31'
    sys.path[:0] = [str(root), str(ext), str(base / 'code')]
    from common import load, sha, verify_bindings, write_new
    from dialogue_native import GENERATION
    from dialogue_provenance import closure
    from legacy_materials import load_builder
    from g4_designs import MODELS
    freeze_path = root / 'gate1/freeze.json'; freeze = load(freeze_path); verify_bindings(freeze['source_sha256'])
    selection_path = root / 'gate1/M3_design_selection.json'; selection = load(selection_path)
    verify_bindings(selection['source_sha256'])
    first_path = ext / 'plans/e9_full/plan.json'; first = load(first_path)
    old_path = Path(first['e4_predecessor']['score_plan']); old = load(old_path)
    topics_path = base / 'data/e4_agents.json'; meta_path = base / 'data/e4_cache_manifest.json'
    topics = {t['topic_id']: t for t in load(topics_path)}; metas = {t['topic_id']: t for t in load(meta_path)['topics']}
    builder, renderer_sources = load_builder(base, ext)
    paths = [freeze_path, selection_path, first_path, old_path, topics_path, meta_path,
             root / 'g4_plan.py', root / 'g4_queue.py', root / 'test_g4_plan.py']
    paths += [ext / n for n in ['common.py', 'checkpoint.py', 'panel_checkpoint.py', 'dialogue_native.py',
        'dialogue_core.py', 'dialogue_bundle.py', 'dialogue_analysis.py', 'dialogue_provenance.py', 'evidence_delta.py']]
    sources = {str(p): sha(p) for p in paths}; sources.update(renderer_sources)
    models = [{**next(m for m in first['models'] if m['repo_id'] == repo), 'batch_size': 16} for repo in MODELS]
    for model in models:
        model['execution_model_dir'] = str(ext / 'model_cache' / model['slug'])
        if not model['R0_operational']: raise ValueError('G4 requires operational native R0')
    policy_path = root / 'G4_plan.json'
    policy = {'schema': 'voices_v31_G4_registration_v1', 'base': str(base), 'extension': str(ext),
        'created_utc': datetime.now(timezone.utc).isoformat(), 'experiment': 'E6_v31', 'models': models,
        'predecessor': str(root / 'G2_G3_complete.json'), 'Gate1_sha256': sha(freeze_path),
        'source_sha256': closure(sources, base, ext), 'planned_runs': 1800, 'planned_runs_per_model': 600,
        'rounds': 2, 'bootstrap_draws': 10000, 'primary_family_size': 6,
        'scope': 'Gate1 empirical extreme policies and eight random controls; original Pasch pair; Qwen/LFM/Phi',
        'generation_overrides': GENERATION, 'native_workload': selection['native_workload'],
        'storage': 'Permanent raw compressed run bundles, independent replay ACK before redundant work release'}
    write_new(policy_path, policy)
    plans = []
    for model in models:
        kernel_path = Path(next(m['kernel_analysis'] for m in old['selected_models'] if m['repo_id'] == model['repo_id']))
        kernel = load(kernel_path); verify_bindings(kernel['file_sha256'])
        if kernel['execution_identity']['critical_identity']['revision'] != model['revision']:
            raise ValueError('Frozen E1 kernel revision differs')
        runs = make_runs(model, topics, metas, selection['models'][model['repo_id']], ext, builder.original.SYSTEM)
        initial = {**sources, str(policy_path): sha(policy_path), str(kernel_path): sha(kernel_path),
                   model['calibration']: sha(model['calibration'])}
        plan = {'schema': 'voices_v3_dialogue_model_plan_v1', 'base': str(base), 'extension': str(ext),
            'experiment': 'E6_v31', 'registration': str(policy_path), 'registration_sha256': sha(policy_path),
            'model': model, 'macro_design_selection': str(selection_path), 'runs': runs,
            'kernels': kernel['point']['relation_kernels'], 'E1_kernel_analysis': str(kernel_path),
            'generation_overrides': GENERATION, 'source_sha256': closure(initial, base, ext),
            'bootstrap_draws': 10000, 'random_seed': 20261101, 'primary_family_size': 6, 'planned_runs': 600}
        if any(v is None for v in plan['kernels'].values()): raise ValueError('Undefined E1 kernel')
        path = root / 'G4' / model['slug'] / 'plan.json'; write_new(path, plan); plans.append(path)
    manifest = root / 'G4_model_plans.json'
    write_new(manifest, {'status': 'prospectively_registered_1800_G4_runs', 'policy': str(policy_path),
        'plans': [str(p) for p in plans], 'source_sha256': {str(p): sha(p) for p in [policy_path] + plans},
        'native_workload': selection['native_workload']})
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--base', required=True); p.add_argument('--extension', required=True)
    a = p.parse_args(); print(build(a.base, a.extension))
