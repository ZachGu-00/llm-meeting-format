"""Synthetic E4 throughput benchmark only; run after inference is idle, inside screen.

Example (in the existing venv):
  python setup/benchmark_e4_gpu.py --output-dir provenance/e4_gpu_benchmark

Uses the real 80 STS topologies with seeded SYNTHETIC logits, never measured E4
cache results. Checks reference NumPy versus GPU using the same NumPy stream,
then measures one mapping: 80 designs x 3 methods x 5000 draws x 6 rounds.
No packages are installed and no inference code/runtime identity is changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from e4_graphs import PASCH_LEFT, PASCH_RIGHT, pair_projection, validate_sts
from e4_simulation import simulate as numpy_simulate
from e4_simulation_torch import simulate as torch_simulate

SECTIONS = ('outcomes', 'paired_difference_to_design0', 'paired_residual_U_measured_minus_other_methods')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def idle_guard():
    """Check both named inference controllers and active CUDA processes; kill nothing."""
    if not os.environ.get('STY'):
        raise RuntimeError('GPU benchmark must run inside GNU screen (STY)')
    if sys.platform != 'linux':
        raise RuntimeError('This GPU launch guard expects the Linux experiment server')
    listing = subprocess.run(['ps', '-eo', 'pid=,args='], text=True, capture_output=True, check=True).stdout
    pattern = re.compile(r'(?:^|[/\s])(?:inference_v\d+\.py|run_e0_v3\.py|run_e0_queue\.py|'
                         r'run_e1[^/\s]*\.py|vllm|text-generation-launcher)(?:\s|$)')
    offenders = []
    for line in listing.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2 or int(fields[0]) == os.getpid():
            continue
        match = pattern.search(fields[1])
        if match:
            offenders.append({'pid': int(fields[0]), 'matched_program': match.group(0).strip().split('/')[-1]})
    query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                            '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True).stdout
    for line in query.splitlines():
        fields = [x.strip() for x in line.split(',')]
        if fields and fields[0].isdigit() and int(fields[0]) != os.getpid():
            offenders.append({'pid': int(fields[0]), 'gpu_process_name': Path(fields[1]).name if len(fields) > 1 else None})
    if offenders:
        raise RuntimeError('Inference/other CUDA processes are active; benchmark deferred: ' + json.dumps(offenders))
    return {'screen': True, 'other_inference_controllers': 0, 'other_cuda_processes': 0,
            'checked_utc': datetime.now(timezone.utc).isoformat()}


def synthetic_inputs(seed, systems):
    rng = np.random.default_rng(seed)
    measured = rng.normal(0., 1.1, (15, 15, 15, 8))
    caches = {'measured': measured, 'P0': measured + rng.normal(0., .5, measured.shape),
              'PK': measured + rng.normal(0., .25, measured.shape)}
    designs = [s['blocks'] for s in systems]
    if len(designs) != 80:
        raise ValueError('Benchmark requires all 80 STS15')
    for design in designs:
        validate_sts(design)
    if pair_projection(PASCH_LEFT) != pair_projection(PASCH_RIGHT):
        raise ValueError('Pasch pair projection mismatch')
    return caches, designs, np.array([1] * 5 + [0] * 10, dtype=np.uint8)


def check_result(result):
    for section in SECTIONS:
        means = np.asarray(result[section]['means'])
        se = np.asarray(result[section]['Monte_Carlo_standard_errors'])
        if not np.isfinite(means).all() or not np.isfinite(se).all() or (se < 0).any():
            raise ValueError('Nonfinite simulation output or negative MCSE')
        if section == 'outcomes' and ((means < 0).any() or (means > 1).any()):
            raise ValueError('Outcome must be a fraction')
        if section != 'outcomes' and (np.any(means[:, 0] != 0) or np.any(se[:, 0] != 0)):
            raise ValueError('Design0 paired contrast and residual must be exactly zero')


def numerical_comparison(reference, candidate, tolerance=1e-12):
    errors = {}
    for section in SECTIONS:
        for key in ('means', 'Monte_Carlo_standard_errors'):
            left, right = np.asarray(reference[section][key]), np.asarray(candidate[section][key])
            if left.shape != right.shape:
                raise ValueError('CPU/GPU output shape differs')
            errors[section + '/' + key] = float(np.max(np.abs(left - right)))
    return {'absolute_tolerance': tolerance, 'max_absolute_errors': errors,
            'passed': all(value <= tolerance for value in errors.values()),
            'meaning': 'Same synthetic caches, seed, NumPy random stream and chunk; not an inference or measured-E4 validation.'}


def numerical_digest(result):
    value = {s: result[s] for s in SECTIONS}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def timed_gpu(torch, device, caches, designs, initial, **kwargs):
    idle_guard()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = torch_simulate(caches, designs, initial, device=str(device), **kwargs)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    check_result(result)
    return result, {'seconds_end_to_end_synchronized': elapsed,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
        'member_updates': len(caches) * len(designs) * kwargs['draws'] * kwargs['rounds'] * len(designs[0]) * 3,
        'timing_includes': 'CPU probability compilation, random generation, GPU transfers/simulation, reduction and result transfer; excludes JSON writing and idle guard'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--chunks', nargs='+', type=int, default=[512, 2048, 5000])
    p.add_argument('--draws', type=int, default=5000)
    p.add_argument('--rounds', type=int, default=6)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=20261031)
    p.add_argument('--max-timing-ratio', type=float, default=1.5)
    p.add_argument('--g6-full-max-seconds', type=float, default=120.)
    args = p.parse_args()
    if (args.draws < 2 or args.rounds < 1 or args.repeats < 2 or min(args.chunks) < 1 or
            len(set(args.chunks)) != len(args.chunks)):
        raise ValueError('Need draws>=2, rounds>=1, repeats>=2 and positive chunks')
    guard = idle_guard()
    import torch
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This benchmark requires a CUDA device')
    torch.set_num_threads(1)
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    report_path = out / 'benchmark_report.json'
    if report_path.exists() or (out / 'synthetic_inputs.npz').exists():
        raise FileExistsError('Use a fresh benchmark output directory; existing runs are retained')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(out / 'benchmark.log', encoding='utf-8')])
    sts_path = ROOT / 'data/e4_graphs/sts15_all80.json'
    systems = json.loads(sts_path.read_text(encoding='utf-8'))['systems']
    caches, designs, initial = synthetic_inputs(args.seed, systems)
    g6_caches = {name: value[:6, :6, :6, :].copy() for name, value in caches.items()}
    g6_initial = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)
    g6_designs = [PASCH_LEFT, PASCH_RIGHT]
    input_path = out / 'synthetic_inputs.npz'
    np.savez_compressed(input_path, **caches, designs=np.asarray(designs, dtype=np.int16), initial=initial,
        g6_initial=g6_initial, g6_designs=np.asarray(g6_designs, dtype=np.int16), seed=np.array(args.seed))
    decision = ROOT / 'setup/runtime_decision.json'
    report = {'schema': 'voices_v3_synthetic_E4_GPU_benchmark_v1', 'status': 'running',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'synthetic_only': True,
        'actual_experiment_completed': False, 'model_inference_performed': False,
        'input_seed': args.seed, 'synthetic_inputs_sha256': sha(input_path), 'initial_guard': guard,
        'configuration': vars(args) | {'output_dir': str(out)},
        'runtime': {'torch': torch.__version__, 'numpy': np.__version__, 'python': platform.python_version(),
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(device),
            'capability': list(torch.cuda.get_device_capability(device)), 'torch_cpu_threads': torch.get_num_threads(),
            'gpu_total_bytes': torch.cuda.get_device_properties(device).total_memory,
            'nvidia_gpu_query': subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
                '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout.strip()},
        'source_sha256': {str(path.relative_to(ROOT)): sha(path) for path in
            [Path(__file__), ROOT / 'code/e4_simulation.py', ROOT / 'code/e4_simulation_torch.py',
             ROOT / 'code/e4_graphs.py', sts_path]},
        'inference_runtime_decision_sha256_before': sha(decision) if decision.is_file() else None,
        'chunks': [], 'selection_rule': 'lowest median seconds among finite, exactly repeatable trials with max/min timing ratio <= configured limit',
        'scope': 'One STS mapping per call; later 20 independent mapping seeds x5000 draws =100000 per design/condition. G6 means normalize over six agents.'}
    write_json(report_path, report)
    logging.info('Synthetic benchmark started; no measured E4 claims')
    # CUDA warmup and all-80-design reference check use identical NumPy draws.
    tiny = dict(draws=31, rounds=2, chunk_size=16, seed=args.seed, tau=1)
    reference = numpy_simulate(caches, designs, initial, **tiny)
    idle_guard()
    torch_simulate(caches, designs, initial, device=str(device), rng_backend='numpy', **tiny)
    actual, tiny_timing = timed_gpu(torch, device, caches, designs, initial, rng_backend='numpy', **tiny)
    report['CPU_numpy_vs_GPU_same_stream'] = numerical_comparison(reference, actual)
    report['CPU_numpy_vs_GPU_same_stream']['GPU_timing'] = tiny_timing
    write_json(out / 'tiny_CPU_reference.json', reference); write_json(out / 'tiny_GPU_reference.json', actual)
    write_json(report_path, report)
    if not report['CPU_numpy_vs_GPU_same_stream']['passed']:
        report['status'] = 'numerical_check_failed'; write_json(report_path, report)
        raise RuntimeError('GPU numerical check failed; do not select a production chunk')
    logging.info('Same-stream CPU/GPU check passed')
    for chunk in args.chunks:
        entry = {'chunk_size': chunk, 'trials': [], 'status': 'running'}
        report['chunks'].append(entry)
        for repeat in range(args.repeats):
            logging.info('STS80 chunk=%s repeat=%s draws=%s rounds=%s', chunk, repeat + 1, args.draws, args.rounds)
            try:
                result, timing = timed_gpu(torch, device, caches, designs, initial, draws=args.draws,
                    rounds=args.rounds, chunk_size=chunk, seed=args.seed + 1, tau=1, rng_backend='torch')
            except torch.cuda.OutOfMemoryError as error:
                entry.update(status='cuda_out_of_memory', error=str(error)); torch.cuda.empty_cache()
                break
            except Exception as error:
                entry.update(status='execution_error', error=str(error))
                report['status'] = 'benchmark_failed'; write_json(report_path, report)
                raise
            artifact = out / f'sts_chunk{chunk}_repeat{repeat + 1}.json'
            write_json(artifact, result)
            entry['trials'].append(timing | {'repeat': repeat + 1, 'artifact': artifact.name,
                'artifact_sha256': sha(artifact), 'numerical_digest': numerical_digest(result)})
            write_json(report_path, report)
        if len(entry['trials']) == args.repeats:
            times = [t['seconds_end_to_end_synchronized'] for t in entry['trials']]
            entry.update(median_seconds=float(np.median(times)), timing_max_min_ratio=max(times) / min(times),
                exact_repeated_results=len({t['numerical_digest'] for t in entry['trials']}) == 1)
            entry['status'] = 'stable' if entry['exact_repeated_results'] and entry['timing_max_min_ratio'] <= args.max_timing_ratio else 'unstable'
        write_json(report_path, report)
    eligible = [e for e in report['chunks'] if e['status'] == 'stable']
    if not eligible:
        report['status'] = 'no_stable_chunk'; write_json(report_path, report)
        raise RuntimeError('No stable chunk; keep benchmark evidence and rerun in a fresh directory')
    chosen = min(eligible, key=lambda e: (e['median_seconds'], e['chunk_size']))
    report['selected_chunk_size'] = chosen['chunk_size']
    report['selected_rng_backend'] = 'torch'
    report['selection_uses_only_synthetic_runtime_and_consistency'] = True
    report['timing_extrapolations_not_experiments'] = {
        'one_STS_mapping_observed_seconds': chosen['median_seconds'],
        'twenty_STS_mappings_linear_estimate_seconds': chosen['median_seconds'] * 20,
        'six_conditions_six_topics_six_models_linear_estimate_seconds': chosen['median_seconds'] * 20 * 6 * 6 * 6,
        'scope': 'STS dynamic simulation only at benchmarked draws/rounds; excludes cache inference, G6, aggregation and runtime variation'}
    write_json(report_path, report)
    logging.info('Selected chunk=%s; STS single-map median %.3fs', chosen['chunk_size'], chosen['median_seconds'])
    # Six-agent Pasch timing is a separate size regime, with no frozen extra agents.
    result, timing = timed_gpu(torch, device, g6_caches, g6_designs, g6_initial, draws=10000,
        rounds=6, chunk_size=chosen['chunk_size'], seed=args.seed + 2, tau=1, rng_backend='torch')
    write_json(out / 'g6_10000draws.json', result)
    report['G6'] = {'n_agents': 6, 'n_designs': 2, 'rounds': 6, 'pilot_draws': 10000, 'pilot': timing,
        'linear_100000_draw_estimate_seconds': timing['seconds_end_to_end_synchronized'] * 10,
        'full_100000_executed': False}
    write_json(report_path, report)
    if report['G6']['linear_100000_draw_estimate_seconds'] <= args.g6_full_max_seconds:
        result, full_timing = timed_gpu(torch, device, g6_caches, g6_designs, g6_initial, draws=100000,
            rounds=6, chunk_size=chosen['chunk_size'], seed=args.seed + 2, tau=1, rng_backend='torch')
        write_json(out / 'g6_100000draws.json', result)
        report['G6'].update(full_100000_executed=True, full=full_timing)
    else:
        report['G6']['skip_reason'] = 'pilot linear estimate exceeds configured benchmark time budget; no full-run claim'
    report.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
        final_guard=idle_guard(), inference_runtime_decision_sha256_after=sha(decision) if decision.is_file() else None)
    report['inference_runtime_decision_unchanged'] = report['inference_runtime_decision_sha256_before'] == report['inference_runtime_decision_sha256_after']
    write_json(report_path, report)
    logging.info('Synthetic benchmark complete; selected chunk=%s, G6 full100k=%s', chosen['chunk_size'], report['G6']['full_100000_executed'])


if __name__ == '__main__':
    try:
        main()
    except Exception:
        logging.exception('Benchmark stopped; no synthetic runtime choice should be inferred from an unfinished run')
        raise
