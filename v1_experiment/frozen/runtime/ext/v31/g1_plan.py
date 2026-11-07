"""Freeze thirty score-only G1 tasks before any new model-topic effects exist."""
import argparse
from pathlib import Path
import sys


def build(base, extension):
    base, ext = Path(base), Path(extension); root = ext / 'v31'
    sys.path[:0] = [str(ext), str(base / 'setup')]
    from common import load, sha, verify_bindings, write_new
    from run_e1_e2_queue import score_job, runtime
    from dialogue_provenance import closure
    target = root / 'G1_plan.json'
    if target.exists():
        result = load(target); verify_bindings(result['source_sha256']); return result
    e9 = ext / 'plans/e9_full/plan.json'; initial = load(e9)
    repos = ['google/gemma-4-E4B-it', 'Qwen/Qwen3.5-2B', 'HuggingFaceTB/SmolLM3-3B',
             'mistralai/Ministral-3-3B-Instruct-2512-BF16', 'nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16']
    sources = {str(e9): sha(e9)}; models = []
    for repo in repos:
        if repo in (repos[0], repos[-1]):
            source = ext / 'plans/expansion/models' / repo.replace('/', '--') / 'tasks.json'
            model = dict(load(source)['model']); sources[str(source)] = sha(source)
            model['execution_model_dir'] = model['model_dir']
        else:
            model = dict(next(m for m in initial['models'] if m['repo_id'] == repo))
        if not model['R0_operational']: raise ValueError('G1 requires qualified operational R0')
        model['conditional_quality_caveat'] = repo in (repos[2], repos[3])
        # Preserve all pre-existing qualification tiers, including R1-primary flags.
        models.append(model)
        for p in (model['calibration'], str(Path(model['model_dir']) / 'snapshot_manifest.json')):
            sources[p] = sha(p)
    material_path = base / 'data/e4_cache_manifest.json'; manifest = load(material_path)
    if manifest['n_topics'] != 6 or manifest['n_rows'] != 294120: raise ValueError('Original E4 inputs differ')
    decision = runtime(base)
    if decision['batch_size'] != 16 or decision['attn_implementation'] != 'eager': raise ValueError('Unexpected frozen runtime')
    jobs = []
    for model in models:
        for topic in manifest['topics']:
            spec = topic['output']; path = base / spec['path']
            if sha(path) != spec['sha256'] or spec['n_rows'] != 49020: raise ValueError('Original E4 input changed')
            sources[str(path)] = spec['sha256']
            job = score_job(base, model, decision, {**spec, 'path': path}, topic['topic_id'], 'G1')
            output = root / 'sealed/G1' / model['slug'] / (topic['topic_id'] + '_R0.jsonl')
            cmd = job['command']; cmd[cmd.index('--output')+1] = str(output)
            cmd[cmd.index('--model-dir')+1] = model['execution_model_dir']
            job.update(output=str(output), log=str(output) + '.log', topic_id=topic['topic_id'],
                       effect_analysis_allowed=False, batch_size=16)
            if output.exists(): raise ValueError('G1 must be registered before new outputs exist')
            jobs.append(job)
    clock = ext / 'plans/smol_date_context/plan.json'
    source_paths = [material_path, base / 'data/e4_agents.json', base / 'data/e4_agent_manifest.json',
                    base / 'code/inference_v3.py', base / 'code/loader_v3.py', base / 'setup/runtime_decision.json',
                    base / 'setup/run_e1_e2_queue.py', root / 'research_plan_v31_user.md', clock,
                    root / 'G0_guard_registration.json', ext / 'common.py', ext / 'checkpoint.py',
                    ext / 'panel_checkpoint.py', ext / 'evidence_delta.py', ext / 'dialogue_provenance.py']
    source_paths += [root / n for n in ('g1_plan.py', 'sealed_queue.py', 'score_entry.py', 'g0_boundary_guard.py', 'test_sealed_queue.py')]
    sources.update({str(p): sha(p) for p in source_paths})
    result = {'schema': 'voices_v31_G1_sealed_registration_v1', 'base': str(base), 'extension': str(ext),
              'models': models, 'jobs': jobs, 'rows': sum(j['rows'] for j in jobs), 'n_jobs': len(jobs),
              'batch_size': 16, 'attn_implementation': 'eager', 'precision': 'native_bfloat16',
              'source_sha256': closure(sources, base, ext), 'smol_clock_policy': str(clock),
              'model_order_reason': 'Finish resident Gemma G1 first to avoid re-downloading it; then remaining listed G1 models',
              'sealing': 'Only row identity, finite values, option mass, format, counts and SHA audit; no E4 effects before Gate1',
              'rotation': 'Only reproducible safetensors after current-model sealed evidence is independently backed up; preserve metadata, sources and all results',
              'G0_boundary_guard': str(root / 'G0_guard_registration.json'),
              'old_queues_remain_paused': True, 'E3_E5_skipped': True, 'G7_not_authorized_here': True}
    if len(jobs) != 30 or result['rows'] != 1470600: raise ValueError('G1 scope differs')
    write_new(target, result); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--base', required=True); p.add_argument('--extension', required=True)
    a = p.parse_args(); r = build(a.base, a.extension)
    print({'status': 'G1_registered_sealed', 'jobs': r['n_jobs'], 'rows': r['rows']})
