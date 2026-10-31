"""H3 for the frozen six-model panel, using jointly resampled E1/E4 topics.

The primary U is E4b's ratio of equal-topic mean squared error to equal-topic
mean squared observed contrast. RMS is a sensitivity analysis, not another
primary hypothesis. No inference, fitting, model selection or filtering occurs.
"""
import argparse
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from analyze_v3 import read_json, sha
from summarize_select_e1_v3 import MANDATORY, expected_jobs, verify_e1
from write_e4_report_v3 import (Evidence, selection_receipt, validate_aggregate,
                                verify_topic_bindings)

ROOT = Path(__file__).resolve().parents[1]
ESTIMAND = 'Spearman(abs(E1_C), E4b_single_round_P0_unexplained_fraction_U)'
NOTE = 'provenance/E4_H3_operationalization_20261031.txt'


class SourceEvidence(Evidence):
    def __init__(self, root, source_roots):
        aliases = [source_roots] if isinstance(source_roots, str) else list(source_roots or [])
        self.aliases = sorted(set(str(p).replace('\\', '/').rstrip('/') for p in aliases), key=len, reverse=True)
        super().__init__(root, self.aliases[0] if self.aliases else '/workspace/voices_nonadditivity_v3')

    def resolve(self, value):
        text = str(value).replace('\\', '/')
        for alias in self.aliases:
            if text.startswith(alias + '/'):
                return self.root / text[len(alias) + 1:]
        return super().resolve(value)

    # The existing E1 verifier uses path(); the E4 verifier calls it resolve().
    def path(self, value):
        return self.resolve(value)


def number(value):
    return float(value) if np.isfinite(value) else None


def ratio(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan),
                     where=denominator > 0)


def rank_correlations(x, y):
    """Average-rank Spearman per row; never omit a model with invalid input."""
    x, y = np.atleast_2d(x).astype(float), np.atleast_2d(y).astype(float)
    if x.shape != y.shape or x.shape[1] < 2:
        raise ValueError('Rank correlation requires equally shaped rows with at least two models')
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    rx, ry = rankdata(x, axis=1, method='average'), rankdata(y, axis=1, method='average')
    rx -= rx.mean(axis=1, keepdims=True)
    ry -= ry.mean(axis=1, keepdims=True)
    sx, sy = np.sum(rx * rx, axis=1), np.sum(ry * ry, axis=1)
    constant_x, constant_y = finite & (sx == 0), finite & (sy == 0)
    denominator = np.sqrt(sx * sy)
    values = ratio(np.sum(rx * ry, axis=1), denominator)
    values[~finite] = np.nan
    values = np.clip(values, -1., 1.)
    return values, {'nonfinite_input': ~finite, 'constant_abs_C_rank': constant_x,
                    'constant_outcome_rank': constant_y}


def exact_positive_permutation(x, y):
    """Enumerate all 6! labeled permutations, retaining tie multiplicities."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != (6,) or y.shape != (6,):
        raise ValueError('The exact H3 permutation panel must contain six models')
    observed = rank_correlations(x, y)[0][0]
    permutations = np.array(list(itertools.permutations(range(6))))
    values = rank_correlations(np.broadcast_to(x, (720, 6)), y[permutations])[0]
    bad = int((~np.isfinite(values)).sum())
    count = None if bad or not np.isfinite(observed) else int(np.sum(values >= observed - 1e-12))
    return {'permutation_p_one_sided_positive': None if count is None else count / 720,
            'n_permutations': 720, 'permutation_exceedances_including_equal': count,
            'permutation_undefined': bad,
            'permutation_definition': 'Permute U/RMS labels over the six fixed models; average ranks for ties; include all 720 labeled assignments including identity; no Monte Carlo +1 correction.',
            'permutation_scope': 'Descriptive exchangeability calculation conditional on this selected panel; not a model-population or selection-adjusted p value.'}


def correlation_summary(c, outcome, c_draws, outcome_draws, repos, estimand):
    point, why = rank_correlations(np.abs(c), outcome)
    samples, failures = rank_correlations(np.abs(c_draws), outcome_draws)
    bad = int((~np.isfinite(samples)).sum())
    leave_out = []
    for i, repo in enumerate(repos):
        keep = np.arange(len(repos)) != i
        leave_out.append({'omitted_repo': repo,
                          'rho': number(rank_correlations(np.abs(c[keep]), outcome[keep])[0][0])})
    result = {'estimand': estimand, 'rho': number(point[0]),
              'point_undefined_reasons': [k for k, v in why.items() if v[0]],
              'interval95': None if bad else np.quantile(samples, [.025, .975]).tolist(),
              'undefined_draws': bad, 'defined_draws': len(samples) - bad,
              'undefined_draws_not_dropped': True,
              'undefined_reason_counts': {k: int(v.sum()) for k, v in failures.items()},
              'undefined_reason_counts_may_overlap': True,
              'interval_method': 'Pointwise percentile95 from shared-topic resampling; interval withheld if any draw is undefined.',
              'leave_one_model_out': leave_out,
              **exact_positive_permutation(np.abs(c), outcome)}
    return result, samples


def shared_topic_draws(e1, e4, draws, seed):
    """Ratios of weighted component sums, with one weight vector per topic set."""
    if draws < 2 or not e1 or len(e1) != len(e4):
        raise ValueError('At least two draws and matched nonempty model panels required')
    e1_topics = sorted({t for model in e1 for t in model['topics']})
    e4_topics = sorted(e4[0])
    if not e1_topics or not e4_topics or any(set(x) != set(e4_topics) for x in e4):
        raise ValueError('Shared bootstrap requires the same E4 topic panel for every model')
    w1 = np.random.default_rng(seed).multinomial(len(e1_topics),
        np.full(len(e1_topics), 1 / len(e1_topics)), size=draws)
    w4 = np.random.default_rng(seed + 1).multinomial(len(e4_topics),
        np.full(len(e4_topics), 1 / len(e4_topics)), size=draws)
    cn, cd = [], []
    for model in e1:
        labels = np.asarray(model['topics'])
        cn.append([np.asarray(model['numerators'])[labels == t].sum() for t in e1_topics])
        cd.append([np.asarray(model['denominators'])[labels == t].sum() for t in e1_topics])
    cn, cd = w1 @ np.asarray(cn).T, w1 @ np.asarray(cd).T
    observed = np.array([[m[t]['observed_mean_square_contrast'] for t in e4_topics] for m in e4])
    error = np.array([[m[t]['P0_mean_square_error'] for t in e4_topics] for m in e4])
    observed_mean, error_mean = w4 @ observed.T / len(e4_topics), w4 @ error.T / len(e4_topics)
    return {'E1_topic_weights': w1, 'E4_topic_weights': w4,
            'C_draws': ratio(cn, cd), 'U_draws': ratio(error_mean, observed_mean),
            'P0_residual_RMS_draws': np.sqrt(error_mean),
            'C_denominator_draws': cd, 'U_denominator_draws': observed_mean}, e1_topics, e4_topics


def agrees(computed, recorded, label):
    valid = np.isfinite(computed)
    if ((not valid and recorded is not None) or
        (valid and (recorded is None or not math.isclose(float(computed), float(recorded),
                                                       rel_tol=1e-11, abs_tol=1e-12)))):
        raise ValueError('Recomputed point differs from bound original analysis: ' + label)


def validated_e1(repo, row, aggregate, evidence):
    primary = row['primary']
    path = evidence.resolve(primary['path'])
    evidence.check(path, primary['sha256'])
    if primary['sha256'] not in aggregate['file_sha256'].values():
        raise ValueError('Aggregate does not bind the selected primary E1 analysis')
    jobs = [j for j in expected_jobs(evidence.root, repo, True) if j['role'] == 'primary']
    if len(jobs) != 1 or path.name != jobs[0]['name'] + '.json':
        raise ValueError('Wrong E1 primary partition; fresh180 is required for LFM and Qwen4')
    state_path = evidence.root / 'results' / repo.replace('/', '--') / 'e0/e0_driver_state.json'
    evidence.check(state_path, row['E0_state_sha256'])
    state = read_json(state_path)
    cal_path = evidence.resolve(state['selected_calibration'])
    evidence.check(cal_path, row['calibration_sha256'])
    calibration = {'value': read_json(cal_path), 'sha256': row['calibration_sha256']}
    result = verify_e1(path, jobs[0], repo, calibration, evidence)
    if (result['execution_identity']['critical_identity'] != aggregate['model_identity'] or
        aggregate['model_identity'] != primary['execution_identity']):
        raise ValueError('E1, aggregate and frozen selection model identities differ')
    expected_n = 180 if repo in MANDATORY else 200
    if result['n_items'] != expected_n:
        raise ValueError('Unexpected E1 primary sample size')
    items = {v['item_id']: v['topic'] for v in read_json(jobs[0]['items'])}
    ids, labels = result['item_ids'], result['topics']
    if labels != [items[item] for item in ids]:
        raise ValueError('E1 item/topic membership differs from the fixed inputs')
    numerator = np.asarray(result['item_C_numerators'], dtype=float)
    denominator = np.asarray(result['item_C_denominators'], dtype=float)
    if (numerator.shape != (expected_n,) or denominator.shape != numerator.shape or
        not np.isfinite(numerator).all() or not np.isfinite(denominator).all() or
        np.any(denominator < 0)):
        raise ValueError('Invalid E1 item-level C ratio components')
    c = ratio(numerator.sum(), denominator.sum()).item()
    agrees(c, result['point']['C'], repo + '/C')
    agrees(c, primary['C'], repo + '/selection C')
    return result, {'topics': labels, 'numerators': numerator, 'denominators': denominator}, c


def analyze(selection_path, aggregate_paths, root=ROOT,
            source_root='/workspace/voices_nonadditivity_v3', draws=10000, seed=20261031):
    root = Path(root).resolve()
    evidence = SourceEvidence(root, source_root)
    selection = selection_receipt(selection_path, evidence)
    if selection['status'] != 'verified':
        raise ValueError('Invalid/incomplete frozen selection: ' + str(selection))
    repos = selection['selected_repos']
    if len(aggregate_paths) != 6:
        raise ValueError('Exactly six aggregate inputs are required')
    frozen = read_json(selection_path)
    panel = read_json(evidence.resolve(frozen['panel_path']))
    panel_models = {r['repo_id']: r for r in panel['models']}
    if len(panel_models) != len(panel['models']):
        raise ValueError('Duplicate models in the frozen E1 panel')
    manifest_path = root / 'data/e4_cache_manifest.json'
    manifest = read_json(manifest_path)
    topic_names = {t['topic_id']: t['topic'] for t in manifest['topics']}
    if len(topic_names) != 6 or len(set(topic_names.values())) != 6:
        raise ValueError('The fixed E4 manifest must contain six distinct topics')
    manifest_sha = evidence.digest(manifest_path)
    evidence.check(manifest_path, manifest_sha)
    note_path = root / NOTE
    evidence.check(note_path, evidence.digest(note_path))
    aggregates = {}
    for path in aggregate_paths:
        value, integrity = evidence.package(path, 'aggregate')
        repo = value['model_identity']['repo']
        if repo not in repos or repo in aggregates:
            raise ValueError('Unexpected or duplicate aggregate model: ' + repo)
        if integrity['status'] != 'verified':
            raise ValueError('Unclosed aggregate sources: ' + str(integrity))
        if not validate_aggregate(value, topic_names):
            raise ValueError('Incomplete six-topic E4 aggregate')
        if manifest_sha not in value['file_sha256'].values():
            raise ValueError('Aggregate is not bound to the fixed E4 manifest')
        if value.get('topics') != [topic_names[t] for t in value['topic_ids']]:
            raise ValueError('Aggregate topic IDs/names differ from the fixed manifest')
        verify_topic_bindings(value, evidence)
        aggregates[repo] = (Path(path), value)
    if set(aggregates) != set(repos):
        raise ValueError('Aggregates do not cover exactly the selected model panel')
    models, e1, e4 = [], [], []
    for repo in repos:
        path, aggregate = aggregates[repo]
        kernel, components, c = validated_e1(repo, panel_models[repo], aggregate, evidence)
        if set(kernel['topics']) & set(topic_names.values()):
            raise ValueError('E1 and E4 topics must be disjoint')
        stats = aggregate['single_round']['E4b']['topic_sufficient_statistics']
        values = np.array([[stats[t][key] for t in sorted(topic_names)]
                          for key in ('observed_mean_square_contrast', 'P0_mean_square_error')])
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Invalid squared-contrast sufficient statistics')
        observed, error = values.mean(axis=1)
        u, rms = ratio(error, observed).item(), float(np.sqrt(error))
        metrics = aggregate['single_round']['E4b']['metrics']
        agrees(u, metrics['P0_unexplained_fraction_U']['estimate'], repo + '/U')
        agrees(rms, metrics['P0_residual_RMS']['estimate'], repo + '/P0 residual RMS')
        e1.append(components)
        e4.append(stats)
        models.append({'repo_id': repo, 'E1_analysis_sha256': panel_models[repo]['primary']['sha256'],
                       'aggregate_sha256': evidence.digest(path), 'aggregate_path': str(path.resolve()),
                       'C': number(c), 'absolute_C': number(abs(c)), 'U': number(u),
                       'P0_residual_RMS': rms, 'n_E1_items': kernel['n_items'],
                       'E1_topics': sorted(set(kernel['topics'])),
                       'E1_primary_partition': 'fresh180' if repo in MANDATORY else 'full200',
                       'P0_mean_square_error_mean': float(error),
                       'observed_mean_square_contrast_mean': float(observed),
                       'point_reproduced_from_components': True,
                       'original_C': kernel['point']['C'],
                       'original_U': metrics['P0_unexplained_fraction_U']['estimate'],
                       'original_P0_residual_RMS': metrics['P0_residual_RMS']['estimate']})
    arrays, e1_topics, e4_topics = shared_topic_draws(e1, e4, draws, seed)
    c = np.array([m['C'] for m in models], dtype=float)
    u = np.array([m['U'] for m in models], dtype=float)
    rms = np.array([m['P0_residual_RMS'] for m in models])
    primary, arrays['rho_draws'] = correlation_summary(c, u, arrays['C_draws'], arrays['U_draws'], repos, ESTIMAND)
    sensitivity, arrays['rho_RMS_draws'] = correlation_summary(c, rms, arrays['C_draws'],
        arrays['P0_residual_RMS_draws'], repos, 'Spearman(abs(E1_C), E4b_single_round_P0_residual_RMS)')
    for name in ('analyze_e4_cross_model_v3.py', 'analyze_v3.py', 'summarize_select_e1_v3.py',
                 'summarize_e0_panel_v3.py', 'write_e4_report_v3.py'):
        source = Path(__file__).with_name(name)
        evidence.check(source, evidence.digest(source))
    result = {'schema': 'voices_v3_E4_cross_model_H3_v1', 'status': 'complete_verified_for_selected_panel',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'selection_sha256': evidence.digest(selection_path), 'selected_repos': repos,
              'n_models': 6, 'topic_ids': e4_topics, 'models': models, 'primary': primary,
              'sensitivity': {'P0_residual_RMS': sensitivity},
              'bootstrap': {'draws': draws, 'seed_E1': seed, 'seed_E4': seed + 1, 'unit': 'topic',
                  'E1_topics': e1_topics, 'E4_topic_ids': e4_topics,
                  'E1_topic_weights_shared_across_models': True,
                  'E4_topic_weights_shared_across_models': True,
                  'E1_and_E4_topic_resampling_independent': True,
                  'existing_model_bootstrap_rows_concatenated': False, 'undefined_draws_not_dropped': True,
                  'E1_topic_support': 'Union of named E1 topics; each model contributes only its fixed primary items.',
                  'C_estimator': 'Ratio of sums of item numerator/denominator after shared topic weighting; abs taken after each ratio.',
                  'U_estimator': 'Ratio of equal-topic mean P0 squared error to equal-topic mean observed squared contrast; never mean of topic ratios.',
                  'RMS_estimator': 'Square root of equal-topic mean P0 squared error.',
                  'per_model_undefined_C_draws': dict(zip(repos, map(int, (~np.isfinite(arrays['C_draws'])).sum(axis=0)))),
                  'per_model_undefined_U_draws': dict(zip(repos, map(int, (~np.isfinite(arrays['U_draws'])).sum(axis=0))))},
              'scope': {'H3_primary_family': 'E4b single-round U only',
                  'RMS_is_sensitivity_not_an_additional_primary_test': True,
                  'H1_Holm_two_test_family_unchanged': True, 'new_acceptance_or_filtering_gate': False,
                  'conditional_on_frozen_selected_six_models': True, 'model_resampling_or_reselection': False,
                  'caveat': 'Six selected models and six E4 topics give limited independent information. Shared topic bootstrap is conditional on the selected panel, fixed prompts/readouts and deterministic caches; it is not model-population or selection-adjusted inference.'},
              'file_sha256': evidence.verified}
    return result, arrays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--aggregate', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--source-root', action='append', help='Recorded source-root alias; repeat for multiple synced roots')
    p.add_argument('--draws', type=int, default=10000)
    p.add_argument('--seed', type=int, default=20261031)
    args = p.parse_args()
    sidecar = args.output.with_name(args.output.stem + '_bootstrap.npz')
    if args.output.exists() or sidecar.exists():
        raise FileExistsError('Preserve existing H3 report and bootstrap artifact; choose new output paths')
    report, arrays = analyze(args.selection, args.aggregate, args.root, args.source_root, args.draws, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    report['bootstrap_artifact'] = {'filename': sidecar.name, 'sha256': sha(sidecar)}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'status': report['status'], 'rho': report['primary']['rho'],
                      'interval95': report['primary']['interval95'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
