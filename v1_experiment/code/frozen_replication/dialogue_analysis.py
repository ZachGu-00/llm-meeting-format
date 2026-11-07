"""Analysis of source-closed native dialogue bundles and paired topic resampling."""
from pathlib import Path
import numpy as np
from common import load, sha, verify_bindings, write_new
from dialogue_bundle import verify_bundle
from dialogue_core import seed


def interval(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all(): raise ValueError('Undefined bootstrap draws must be explicitly handled')
    return [float(x) for x in np.quantile(values, [.025, .975])]


def summarize_run(replayed):
    config = replayed['config']; events = replayed['events']; n = len(config['agent_ids']); tau = config['target_stance']
    per_method = {name: {metric: [] for metric in ('logit_squared_error', 'probability_squared_error', 'brier', 'log_loss')}
                  for name in ('P0', 'PK')}
    drift = {'P0': 0., 'PK': 0.}; rounds = []
    for i, event in enumerate(events):
        for update in event['member_updates']:
            actual = int(update['after'] == 1); before = int(update['before'] == 1)
            for name in per_method:
                logit = update['logits'][name]; probability = update['support_probabilities'][name]
                values = per_method[name]
                values['logit_squared_error'].append((logit - update['logits']['measured'])**2)
                values['probability_squared_error'].append((probability - update['support_probabilities']['measured'])**2)
                values['brier'].append((probability - actual)**2)
                # Stable Bernoulli negative log likelihood, including extreme logits.
                values['log_loss'].append(float(np.logaddexp(0., logit) - actual * logit))
                drift[name] += tau * (probability - before) / n
        if i == len(events) - 1 or events[i+1]['meeting']['round'] != event['meeting']['round']:
            initial = sum(x == tau for x in config['initial_state']) / n
            rounds.append({'round': event['meeting']['round'] + 1,
                'observed_target_adoption': sum(x == tau for x in event['after']) / n,
                'conditional_drift_compensator': {name: initial + value for name, value in drift.items()},
                'compensator_is_a_counterfactual_endpoint_probability': False})
    metrics = {name: {k: float(np.mean(v)) for k, v in values.items()} for name, values in per_method.items()}
    return {'run_id': config['run_id'], 'run_number': config['run_number'], 'topic_id': config['topic_id'],
        'family': config['family'], 'design_id': config['design_id'], 'design_ordinal': config['design_ordinal'],
        'target_stance': tau, 'minority_count': config['minority_count'], 'n_agents': n,
        'n_meetings': len(events), 'n_updates': len(events) * 3, 'metrics': metrics,
        'primary_P0_minus_PK_logit_squared_error': metrics['P0']['logit_squared_error'] - metrics['PK']['logit_squared_error'],
        'q_mean': float(np.mean([e['q_mean'] for e in events])),
        'q_below_point5_fraction': float(np.mean([e['q_below_point5_fraction'] for e in events])), 'rounds': rounds,
        'generation_quality': {'utterances': len(replayed['generation_quality']),
            'word_count_mean': float(np.mean([g['word_count'] for g in replayed['generation_quality']])),
            'word_count_min': min(g['word_count'] for g in replayed['generation_quality']),
            'word_count_max': max(g['word_count'] for g in replayed['generation_quality']),
            'token_count_mean': float(np.mean([g['generated_token_count'] for g in replayed['generation_quality']])),
            **{key + '_fraction': float(np.mean([g[key] for g in replayed['generation_quality']]))
               for key in ('empty_display_text', 'terminated_by_eos', 'reached_token_budget')}}}


def bootstrap_indices(topics, draws, random_seed):
    topics = np.array(topics); unique = sorted(set(topics)); rng = np.random.default_rng(random_seed)
    groups = {topic: np.flatnonzero(topics == topic) for topic in unique}
    for _ in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        yield np.concatenate([rng.choice(groups[t], size=len(groups[t]), replace=True) for t in sampled])


def family_analysis(rows, family, draws, random_seed):
    rows = [r for r in rows if r['family'] == family]; runs = sorted({r['run_number'] for r in rows})
    design_ids = [r['design_id'] for r in sorted([r for r in rows if r['run_number'] == runs[0]], key=lambda r: r['design_ordinal'])]
    cells = {(r['run_number'], r['design_id']): r for r in rows}
    if len(cells) != len(rows) or set(cells) != {(run, design) for run in runs for design in design_ids}:
        raise ValueError('Incomplete paired generated-dialogue design panel')
    topics = [cells[run, design_ids[0]]['topic_id'] for run in runs]
    if any(cells[run, design]['topic_id'] != topics[i] for i, run in enumerate(runs) for design in design_ids):
        raise ValueError('Competing designs have different topic contexts')
    improvement = np.array([[cells[run, design]['primary_P0_minus_PK_logit_squared_error'] for design in design_ids] for run in runs])
    adoption = np.array([[[round_['observed_target_adoption'] for round_ in cells[run, design]['rounds']]
                          for design in design_ids] for run in runs])
    estimate = float(improvement.mean()); boot = []; adoption_boot = []
    for sample in bootstrap_indices(topics, draws, random_seed):
        boot.append(float(improvement[sample].mean())); adoption_boot.append(adoption[sample].mean(axis=0))
    boot = np.asarray(boot); adoption_boot = np.asarray(adoption_boot); point = adoption.mean(axis=0)
    p = float((1 + np.count_nonzero(np.abs(boot - estimate) >= abs(estimate))) / (draws + 1))
    contrasts = []
    for a in range(len(design_ids)):
        for b in range(a + 1, len(design_ids)):
            contrasts.append({'design_a': design_ids[a], 'design_b': design_ids[b],
                'observed_target_adoption_difference_by_round': [float(x) for x in point[a] - point[b]],
                'interval95_by_round': [interval(adoption_boot[:, a, r] - adoption_boot[:, b, r]) for r in range(adoption.shape[2])],
                'interpretation': 'secondary paired observed-endpoint contrast; no individual multiplicity-adjusted claim'})
    result = {'family': family, 'runs_per_design': len(runs), 'design_ids': design_ids, 'topics': sorted(set(topics)),
        'primary_conditional_loss_improvement': {'estimate': estimate, 'interval95': interval(boot),
            'two_sided_recentered_bootstrap_p': p, 'positive_means': 'PK has smaller conditional squared-logit error'},
        'observed_adoption': [{'design_id': design, 'mean_by_round': [float(x) for x in point[j]],
            'interval95_by_round': [interval(adoption_boot[:, j, r]) for r in range(adoption.shape[2])]} for j, design in enumerate(design_ids)],
        'paired_endpoint_contrasts': contrasts,
        'secondary_losses': {name: {metric: float(np.mean([r['metrics'][name][metric] for r in rows]))
            for metric in rows[0]['metrics'][name]} for name in ('P0', 'PK')},
        'bootstrap': 'six topics, then paired run contexts within topic; all design members kept together',
        'topic_cluster_limitation': 'Only six original development topics; these are not thousands of independent agents'}
    return result, boot, adoption_boot


def analyze(plan_path, native_completion, output):
    plan = load(plan_path); verify_bindings(plan['source_sha256']); native = load(native_completion)
    if (native['status'] != 'complete_native_runs_backups_may_be_pending' or native['model_plan_sha256'] != sha(plan_path) or
        native['run_ids'] != [r['run_id'] for r in plan['runs']] or len(native['bundles']) != len(plan['runs'])):
        raise ValueError('Native dialogue panel is not exactly closed')
    result_rows = []; sources = {str(plan_path): sha(plan_path), str(native_completion): sha(native_completion)}
    bundle_hashes = {}
    for info in native['bundles']:
        _, replayed = verify_bundle(info['archive'], info, return_events=True)
        result_rows.append(summarize_run(replayed)); bundle_hashes[info['archive']] = info['archive_sha256']
        ack_path = Path(plan['extension']) / 'archives/dialogue_off_server_ack' / (info['archive_sha256'] + '.json')
        ack = load(ack_path)
        if (not ack.get('off_server_copy_verified') or not ack.get('state_and_memory_replayed') or
            ack['archive_sha256'] != info['archive_sha256'] or ack['model_plan_sha256'] != sha(plan_path)):
            raise ValueError('Independent native raw backup is not verified')
        sources[str(ack_path)] = sha(ack_path)
    analyses = {}; arrays = {}
    for family in ('G6', 'STS15'):
        analyses[family], boot, adoption = family_analysis(result_rows, family, plan['bootstrap_draws'], seed(plan['model']['repo_id'], family, 'bootstrap'))
        arrays[family + '_conditional_improvement'] = boot; arrays[family + '_observed_adoption'] = adoption
    artifact = Path(output).with_suffix('.npz'); artifact.parent.mkdir(parents=True, exist_ok=True)
    if artifact.exists():
        with np.load(artifact, allow_pickle=False) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                raise ValueError('Existing bootstrap artifact differs')
    else:
        with artifact.open('xb') as f: np.savez_compressed(f, **arrays)
    sources[str(artifact)] = sha(artifact)
    result = {'schema': 'voices_v3_dialogue_analysis_v1', 'status': 'complete_verified', 'experiment': plan['experiment'],
        'repo_id': plan['model']['repo_id'], 'model_plan_sha256': sha(plan_path), 'runs': result_rows,
        'families': analyses, 'source_sha256': sources, 'native_bundle_sha256': bundle_hashes,
        'native_bundle_scope': 'permanent compressed raw members independently verified and state-replayed, not replaced by summaries',
        'q_mean': float(np.mean([r['q_mean'] for r in result_rows])),
        'q_below_point5_fraction': float(np.mean([r['q_below_point5_fraction'] for r in result_rows])),
        'statistical_scope': 'conditional prediction on realized dialogue paths; endpoint contrasts separately descriptive',
        'per_family_p_values_not_yet_multiplicity_adjusted': True}
    write_new(output, result); return result


def report(registration, analyses, output):
    policy = load(registration); sources = {str(registration): sha(registration)}; models = []; tests = []
    for path in analyses:
        result = load(path); verify_bindings(result['source_sha256']); sources[str(path)] = sha(path)
        models.append({'repo_id': result['repo_id'], 'analysis': str(path), 'analysis_sha256': sha(path), 'families': result['families']})
        for family, data in result['families'].items():
            tests.append({'repo_id': result['repo_id'], 'family': family, **data['primary_conditional_loss_improvement']})
    if {m['repo_id'] for m in models} != {m['repo_id'] for m in policy['models']} or len(tests) != policy['primary_family_size']:
        raise ValueError('Incomplete E6 model or comparison family')
    ordered = sorted(range(len(tests)), key=lambda i: tests[i]['two_sided_recentered_bootstrap_p']); cumulative = 0.
    for rank, index in enumerate(ordered):
        cumulative = max(cumulative, (len(tests) - rank) * tests[index]['two_sided_recentered_bootstrap_p'])
        tests[index]['Holm_p'] = min(1., cumulative)
    result = {'schema': 'voices_v3_E6_report_v1', 'status': 'complete_verified_E6', 'models': models,
        'primary_comparisons': tests, 'source_sha256': sources, 'native_runs': 1800,
        'limitations': ['Only six original topics.', 'Generated prose stance is not independently validated.',
            'Conditional path predictions do not identify free-running counterfactual dialogue endpoints.',
            'The missing v2 section 3c is handled by the explicit registered v3 operationalization.']}
    write_new(output, result); return result
