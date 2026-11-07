"""Register generated-dialogue scope, then derive paired runs from PK-only selection."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from common import load, sha, verify_bindings, write_new
from dialogue_core import seed, meeting_schedule
from dialogue_native import GENERATION
from dialogue_provenance import closure
from legacy_materials import load_builder
from macro_designs_v2 import prepare as select_macro

MODELS = ['Qwen/Qwen3.5-4B', 'LiquidAI/LFM2.5-1.2B-Instruct', 'ibm-granite/granite-4.2-3b']
SOURCES = ['common.py', 'checkpoint.py', 'panel_checkpoint.py', 'legacy_materials.py',
    'macro_designs_v2.py', 'low_order.py', 'dialogue_core.py', 'dialogue_native.py', 'dialogue_bundle.py',
    'dialogue_provenance.py', 'dialogue_plan_v2.py', 'dialogue_analysis.py', 'dialogue_queue_v2.py',
    'dialogue_archive.py', 'e9_queue.py', 'DIALOGUE_PROTOCOL.md', 'DIALOGUE_STORAGE_PROTOCOL.md',
    'DIALOGUE_ANALYSIS_PROTOCOL.md', 'ADVANCEMENT_PROTOCOL.md']


def register(base, extension):
    base, root = Path(base).resolve(), Path(extension).resolve(); output = root / 'plans/dialogue_v2/plan.json'
    if output.exists():
        result = load(output); verify_bindings(result['frozen_file_sha256']); return result
    first_path = root / 'plans/e9_full/plan.json'; first = load(first_path)
    gate_path = root / 'plans/advancement/plan.json'; gate = load(gate_path)
    models = []
    for repo in MODELS:
        model = next(m for m in first['models'] if m['repo_id'] == repo)
        if not model['R0_operational']: raise ValueError('Preselected E6 model lacks operational native R0')
        label = next(m for m in gate['models'] if m['repo_id'] == repo)
        models.append({**model, 'batch_size': 16, 'execution_model_dir': str(root / 'model_cache' / model['slug']),
            'E1_saturation_label': label['E1_saturation_label'], 'E1_C': label['E1_C']})
    if not any(m['E1_saturation_label'] for m in models): raise ValueError('E6 must include a measured saturated model')
    predecessor = root / 'plans/e8/plan.json'
    superseded = root / 'plans/dialogue/plan.json'; amendment = root / 'plans/dialogue/superseded_before_execution.json'
    note = load(amendment)
    if note['native_E6_started'] or note['superseded_sha256'] != sha(superseded):
        raise ValueError('Implementation amendment must preserve the original pre-execution registration')
    files = [root / name for name in SOURCES] + list((base / 'code').glob('*.py')) + [first_path, gate_path, predecessor, superseded, amendment,
        base / 'data/e4_agents.json', base / 'data/e4_cache_manifest.json', base / 'data/e4_graphs/sts15_all80.json',
        base / 'data/e9_affine_balance/source_render_v2.py']
    validation_path = root / 'provenance/dialogue_v2_cpu_validation.json'; validation = load(validation_path)
    if validation['status'] != 'passed' or validation['GPU_work_started']:
        raise ValueError('Successful CPU checks must precede registration')
    verify_bindings(validation['source_sha256']); files.append(validation_path)
    result = {'schema': 'voices_v3_dialogue_registration_v1', 'base': str(base), 'extension': str(root),
        'created_utc': datetime.now(timezone.utc).isoformat(), 'experiment': 'E6', 'models': models,
        'implementation_revision': 2, 'amends_registration': str(superseded),
        'amendment_receipt': str(amendment), 'scientific_scope_changed_by_amendment': False,
        'predecessor_plan': str(predecessor), 'predecessor_plan_sha256': sha(predecessor),
        'advancement_registration': str(gate_path), 'advancement_registration_sha256': sha(gate_path),
        'frozen_file_sha256': {str(p): sha(p) for p in files}, 'runs_per_design': 50, 'rounds_per_run': 2,
        'G6_designs': 2, 'STS15_designs': 10, 'planned_runs_per_model': 600,
        'generation_overrides': GENERATION, 'generation_batch_size': 1, 'native_readout_batch_size': 16,
        'readouts_per_meeting': 42, 'bootstrap_draws': 10000, 'primary_family_size': 6,
        'primary_comparison': 'conditional P0 minus PK squared-logit error; three models by G6/STS15; Holm six',
        'uniform_policy': 'open-interval 53-bit SHA-derived common uniforms indexed by run/round/canonical triple/member',
        'E3_E5': 'excluded_by_user', 'E7': 'separately registered after the recorded v3 advancement gate',
        'single_GPU_policy': 'wait for E8 terminal closure and hold the original shared lease; no competing inference'}
    write_new(output, result); return result


def paired_runs(model, topics, metas, systems, macro, root, system_message):
    selected = macro['selected_system_ids']; graphs = {s['id']: s['blocks'] for s in systems}
    mappings = {t['topic_id']: t['point_to_agent_mappings'] for t in macro['topics']}; result = []
    for run in range(50):
        ti = run % 6; cycle = run // 6; topic = topics[ti]; meta = metas[topic['topic_id']]
        for family in ('G6', 'STS15'):
            if family == 'G6':
                agents = meta['g6_agent_ids']; minority = (2, 3)[(cycle + ti) % 2]
                tau = 1 if (cycle // 2 + ti) % 2 == 0 else -1
                designs = [('left', meta['g6_pasch_left']), ('right', meta['g6_pasch_right'])]
                mapping_index = None
            else:
                agents = list(range(15)); minority = (3, 5, 7)[(cycle + ti) % 3]
                tau = 1 if (cycle + ti) % 2 == 0 else -1
                mapping_index = seed('mapping', topic['topic_id'], run) % 20
                mapping = mappings[topic['topic_id']][mapping_index]
                designs = [(sid, sorted([sorted(mapping[v] for v in b) for b in graphs[sid]])) for sid in selected]
            for ordinal, (sid, blocks) in enumerate(designs):
                rid = f'E6_{family}_d{ordinal:02d}_r{run:02d}'
                config = {'run_id': rid, 'experiment': 'E6', 'run_number': run, 'family': family,
                    'design_id': sid, 'design_ordinal': ordinal, 'topic_id': topic['topic_id'], 'topic': topic['topic'],
                    'agent_ids': agents, 'agents': [a for a in topic['agents'] if a['agent_id'] in agents],
                    'initial_state': [tau if k < minority else -tau for k in range(len(agents))],
                    'minority_count': minority, 'target_stance': tau, 'mapping_index': mapping_index,
                    'blocks': [list(b) for b in sorted(tuple(sorted(b)) for b in blocks)], 'rounds': 2,
                    'coupling_key': f'E6|{family}|{topic["topic_id"]}|run{run}', 'system_message': system_message,
                    'work_dir': str(Path(root) / 'results/dialogue' / model['slug'] / 'E6/work' / rid)}
                meeting_schedule(config); result.append(config)
    if len(result) != 600 or len({r['run_id'] for r in result}) != 600:
        raise ValueError('Registered 12 by 50 run scope differs')
    return result


def model_plan(registration, model):
    registration = Path(registration); policy = load(registration); verify_bindings(policy['frozen_file_sha256'])
    root, base = Path(policy['extension']), Path(policy['base'])
    output = root / 'plans/dialogue_v2/models' / model['slug'] / 'plan.json'
    if output.exists():
        old = load(output); verify_bindings(old['source_sha256']); return output, old
    macro_path = output.parent / 'PK_design_selection.json'
    macro = select_macro(base, root, model['repo_id'], macro_path)
    agents_path = base / 'data/e4_agents.json'; metadata_path = base / 'data/e4_cache_manifest.json'
    systems_path = base / 'data/e4_graphs/sts15_all80.json'
    topics = sorted(load(agents_path), key=lambda t: t['topic_id']); metadata = load(metadata_path)
    if len(topics) != 6: raise ValueError('All original topics required')
    builder, renderer_sources = load_builder(base, root)
    runs = paired_runs(model, topics, {t['topic_id']: t for t in metadata['topics']}, load(systems_path)['systems'],
                       macro, root, builder.original.SYSTEM)
    first = load(root / 'plans/e9_full/plan.json'); e4 = load(first['e4_predecessor']['score_plan'])
    kernel_path = Path(next(m['kernel_analysis'] for m in e4['selected_models'] if m['repo_id'] == model['repo_id']))
    kernel = load(kernel_path); verify_bindings(kernel['file_sha256'])
    if kernel['execution_identity']['critical_identity']['revision'] != model['revision']:
        raise ValueError('E6 independent E1 kernel belongs to another model revision')
    if any(v is None for v in kernel['point']['relation_kernels'].values()):
        raise ValueError('Undefined E1 kernels require explicit unresolved reporting')
    initial = {str(p): sha(p) for p in (registration, macro_path, agents_path, metadata_path, systems_path,
                                      kernel_path, Path(model['calibration']))}
    initial.update(renderer_sources); initial.update(policy['frozen_file_sha256'])
    sources = closure(initial, base, root)
    result = {'schema': 'voices_v3_dialogue_model_plan_v1', 'base': str(base), 'extension': str(root), 'experiment': 'E6',
        'registration': str(registration), 'registration_sha256': sha(registration), 'model': model,
        'macro_design_selection': str(macro_path), 'runs': runs, 'kernels': kernel['point']['relation_kernels'],
        'E1_kernel_analysis': str(kernel_path), 'generation_overrides': GENERATION, 'source_sha256': sources,
        'bootstrap_draws': policy['bootstrap_draws'], 'random_seed': 20261031,
        'primary_family_size': policy['primary_family_size'], 'planned_runs': 600,
        'raw_evidence_format': 'individually verified immutable tar.gz run bundles; see storage protocol'}
    write_new(output, result); return output, result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--base', required=True); p.add_argument('--extension', required=True)
    a = p.parse_args(); value = register(a.base, a.extension); print({'models': len(value['models']), 'runs_per_model': 600})
