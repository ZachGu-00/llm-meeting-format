"""CPU-only H3 arithmetic and real-schema synthetic receipt tests; no model runs."""
import copy
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy.stats import spearmanr

import analyze_v3 as base
import analyze_e4_cross_model_v3 as h3
from aggregate_e4_v3 import metric_values, statistics

REMOTE = '/workspace/voices_nonadditivity_v3'
TOPICS = [f'E4_{i}' for i in range(6)]
REPOS = [*h3.MANDATORY, *[f'fixture/model{i}' for i in range(4)]]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')
    return path


def fixture(root):
    """Actual analyzer/statistics output shapes; score receipts are synthetic.

    Tiny placeholder files deliberately avoid claiming a GPU inference or a
    token-level validation. The production verifier checks their exact bytes
    and their declared counts/identities, as it does for full closed runs.
    """
    remote = lambda p: REMOTE + '/' + p.relative_to(root).as_posix()
    manifest = write(root / 'data/e4_cache_manifest.json',
                     {'topics': [{'topic_id': t, 'topic': 'heldout ' + t} for t in TOPICS]})
    write(root / h3.NOTE, {'fixture': 'Synthetic H3 test operationalization'})
    items = [{'item_id': f'I{i:03d}', 'topic': f'E1_{i % 3}'} for i in range(200)]
    write(root / 'data/e1_holdout200_items.json', items)
    write(root / 'data/e1_remaining180_items.json', items[:180])
    prompt = write(root / 'data/e1_holdout200_prompts.jsonl', {'fixture': 'synthetic prompt receipt'})
    analyzer = root / 'code/analyze_v3.py'
    analyzer.parent.mkdir(parents=True, exist_ok=True)
    analyzer.write_bytes(Path(base.__file__).read_bytes())
    selected = {'schema': 'voices_v3_frozen_E4_model_selection_v1', 'status': 'frozen',
                'performed': True, 'all_requested_E1_complete': True, 'selected_repos': REPOS}
    panel = {'all_requested_E1_complete': True, 'E4_selection': copy.deepcopy(selected), 'models': []}
    aggregates = []
    for mi, repo in enumerate(REPOS):
        directory = root / 'results' / repo.replace('/', '--')
        identity = {'repo': repo, 'revision': 'synthetic', 'precision': 'fp32',
                    'remedy': 'original', 'model_kind': 'fixture', 'option_ids': {'A': 1, 'B': 2}}
        engine = {'parent_loader': {'repo_id': repo}, 'fixture': 'No actual inference'}
        cal = write(directory / 'e0/calibration.json',
                    {'engine': engine, 'option_ids': identity['option_ids'], 'batch_size': 16})
        state = write(directory / 'e0/e0_driver_state.json', {'selected_calibration': remote(cal)})
        raw = write(directory / 'e1/holdout200_R0.jsonl', {'fixture': 'synthetic score receipt', 'model': repo})
        run = {'engine': engine, 'option_ids': identity['option_ids'], 'batch_size': 16,
               'calibration_file_sha256': base.sha(cal), 'input_file_sha256': base.sha(prompt),
               'n_input': 124400, 'readout': 'R0'}
        run_path = write(Path(str(raw) + '.run.json'), run)
        write(Path(str(raw) + '.summary.json'), {**run, 'status': 'complete',
              'exact_id_and_count_equality': True, 'n_output': 124400,
              'output_bytes': raw.stat().st_size, 'output_sha256': base.sha(raw),
              'run_file_sha256': base.sha(run_path)})
        n = 180 if repo in h3.MANDATORY else 200
        used = items[:n]
        y = []
        for item in used:
            theta = (mi + 1) * (.8 + .2 * int(item['topic'][-1])) * (-1 if mi % 2 else 1)
            y.append([sum(1 if j < 3 else -1 for j in range(5) if mask >> j & 1)
                      + theta * sum(bool(mask >> a & 1 and mask >> b & 1)
                                    for a, b in itertools.combinations(range(3), 2))
                      for mask in range(32)])
        analysis = base.analyze({'y': np.array(y), 'masks': list(range(32)),
            'item_ids': [x['item_id'] for x in used], 'topics': [x['topic'] for x in used],
            'n_physical_rows': 622 * n, 'q_median': 1.}, draws=16)
        # The legacy interval payload is not used by H3; only this fixture's
        # completion receipt declares its planned production bootstrap count.
        analysis['bootstrap']['draws'] = 10000
        analysis.update(readout='R0', selection={'keep_items': None, 'exclude_items': None, 'filters': {}},
            execution_identity={'same_execution_policy': True, 'mixed_execution_explicitly_allowed': False,
                'critical_identity': identity, 'runs': [{'path': remote(raw), 'run_sha256': base.sha(run_path),
                    'batch_size': 16, 'calibration_sha256': base.sha(cal), 'engine': engine}]},
            file_sha256={remote(p): base.sha(p) for p in (raw, prompt, analyzer)})
        name = 'confirmatory180_R0_analysis.json' if n == 180 else 'holdout200_R0_analysis.json'
        e1 = write(directory / 'e1' / name, analysis)
        panel['models'].append({'repo_id': repo, 'E1_complete': True, 'E4_eligible': True,
            'E0_state_sha256': base.sha(state), 'calibration_sha256': base.sha(cal),
            'primary': {'path': remote(e1), 'sha256': base.sha(e1), 'C': analysis['point']['C'],
                        'execution_identity': identity}})
        hashes = {remote(p): base.sha(p) for p in (manifest, e1)}
        topic_stats = {}
        for ti, tid in enumerate(TOPICS):
            cache = directory / (tid + '_caches.npz')
            with cache.open('wb') as stream:
                np.savez(stream, fixture=np.array([mi, ti]))
            point = write(directory / (tid + '_E4_analysis.json'), {
                'topic_id': tid, 'execution_identity': {'critical_identity': identity},
                'cache_artifact': {'filename': cache.name, 'sha256': base.sha(cache)},
                'E4a': {'n_conditions': 70}, 'E4c': {'n_conditions': 192},
                'E4b': {'n_systems': 80, 'n_mappings': 20,
                    'configurations': [{'n_target': m, 'tau': t} for m in (3, 5, 7) for t in (-1, 1)]}})
            hashes.update({remote(p): base.sha(p) for p in (point, cache)})
            observed = np.broadcast_to(np.arange(80.) * (ti + 1) / 500, (6, 20, 80))
            p0 = (1 - np.sqrt((mi + 1) / 7)) * observed
            values = statistics('E4b', observed, p0, (.9 * observed)[..., None])
            topic_stats[tid] = {k: float(v[0]) for k, v in values.items()}
        means = {k: np.mean([v[k] for v in topic_stats.values()]) for k in topic_stats[TOPICS[0]]}
        metrics = {k: {'estimate': float(v)} for k, v in metric_values(means).items()}
        sidecar = directory / 'aggregate_bootstrap.npz'
        with sidecar.open('wb') as stream:
            np.savez(stream, fixture=np.array([mi]))
        aggregate = {'model_identity': identity, 'topic_ids': TOPICS,
            'topics': ['heldout ' + t for t in TOPICS], 'n_topics': 6,
            'planned_six_topics_complete': True, 'bootstrap': {'draws': 10000, 'unit': 'topic'},
            'single_round_E1_kernel_uncertainty_propagated': True,
            'single_round': {part: {'metrics': copy.deepcopy(metrics),
                'topic_sufficient_statistics': copy.deepcopy(topic_stats)} for part in ('E4a', 'E4b', 'E4c')},
            'file_sha256': hashes, 'bootstrap_artifact': {'filename': sidecar.name, 'sha256': base.sha(sidecar)}}
        aggregates.append(write(directory / 'aggregate.json', aggregate))
    panel_path = write(root / 'panel.json', panel)
    selected.update(panel_path=remote(panel_path), panel_sha256=base.sha(panel_path))
    return write(root / 'selection.json', selected), aggregates


class ArithmeticTests(unittest.TestCase):
    def test_known_ranks_ties_and_exact_permutation(self):
        x = np.arange(6.)
        self.assertAlmostEqual(h3.rank_correlations(x, x)[0][0], 1.)
        self.assertAlmostEqual(h3.rank_correlations(x, -x)[0][0], -1.)
        tied = np.array([0., 0., 2., 3., 3., 5.])
        self.assertAlmostEqual(h3.rank_correlations(x, tied)[0][0], spearmanr(x, tied).statistic)
        self.assertEqual(h3.exact_positive_permutation(x, x)['permutation_p_one_sided_positive'], 1 / 720)
        self.assertEqual(h3.exact_positive_permutation(tied, tied)['permutation_exceedances_including_equal'], 4)

    def test_constant_and_undefined_not_filtered(self):
        x = np.arange(6.)
        result, samples = h3.correlation_summary(x, np.ones(6),
            np.stack([x, x, x]), np.stack([x, np.ones(6), [np.nan, 1, 2, 3, 4, 5]]),
            REPOS, h3.ESTIMAND)
        self.assertIsNone(result['rho'])
        self.assertIsNone(result['permutation_p_one_sided_positive'])
        self.assertEqual(result['n_permutations'], 720)
        self.assertEqual(result['undefined_draws'], 2)
        self.assertIsNone(result['interval95'])
        self.assertEqual(result['undefined_reason_counts']['constant_outcome_rank'], 1)
        self.assertEqual(result['undefined_reason_counts']['nonfinite_input'], 1)
        self.assertTrue(np.isnan(samples[1:]).all())
        self.assertTrue(h3.rank_correlations(np.ones(6), x)[1]['constant_abs_C_rank'][0])

    def test_shared_topic_weights_and_ratio_reconstruction(self):
        e1 = [{'topics': ['x', 'x', 'y'], 'numerators': np.array([1., 2., 9.]) * k,
               'denominators': np.array([1., 1., 3.])} for k in (1, 2)]
        e4 = [{'a': {'observed_mean_square_contrast': 1., 'P0_mean_square_error': k},
               'b': {'observed_mean_square_contrast': 9., 'P0_mean_square_error': k}} for k in (1, 2)]
        arr, topics, tids = h3.shared_topic_draws(e1, e4, 200, 17)
        np.testing.assert_allclose(arr['C_draws'][:, 1], 2 * arr['C_draws'][:, 0])
        np.testing.assert_allclose(arr['U_draws'][:, 1], 2 * arr['U_draws'][:, 0])
        np.testing.assert_array_equal(arr['E1_topic_weights'].sum(axis=1), 2)
        np.testing.assert_array_equal(arr['E4_topic_weights'].sum(axis=1), 2)
        self.assertFalse(np.array_equal(arr['E1_topic_weights'], arr['E4_topic_weights']))
        equal = np.where(np.all(arr['E4_topic_weights'] == 1, axis=1))[0][0]
        self.assertAlmostEqual(arr['U_draws'][equal, 0], .2)
        self.assertNotAlmostEqual(arr['U_draws'][equal, 0], (1 + 1 / 9) / 2)

    def test_zero_denominators_remain_undefined(self):
        e1 = [{'topics': ['x', 'y'], 'numerators': [1., 1.], 'denominators': [0., 1.]}]
        e4 = [{'a': {'observed_mean_square_contrast': 0., 'P0_mean_square_error': 1.},
               'b': {'observed_mean_square_contrast': 1., 'P0_mean_square_error': 1.}}]
        arr, _, _ = h3.shared_topic_draws(e1, e4, 200, 31)
        self.assertGreater(np.isnan(arr['C_draws']).sum(), 0)
        self.assertGreater(np.isnan(arr['U_draws']).sum(), 0)
        self.assertTrue(np.isfinite(arr['P0_residual_RMS_draws']).all())

    def test_multiple_source_root_aliases(self):
        root = Path(tempfile.gettempdir()) / 'h3-map-test'
        e = h3.SourceEvidence(root, ['/old/execution', '/new/execution'])
        for value in ('/old/execution/results/a.json', '/new/execution/results/a.json'):
            self.assertEqual(e.resolve(value), root / 'results/a.json')
            self.assertEqual(e.path(value), root / 'results/a.json')


class ClosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = tempfile.TemporaryDirectory()
        cls.source_root = Path(cls.source.name)
        fixture(cls.source_root)

    @classmethod
    def tearDownClass(cls):
        cls.source.cleanup()

    def setUp(self):
        import shutil
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copytree(self.source_root, self.root, dirs_exist_ok=True)
        self.selection = self.root / 'selection.json'
        self.aggregates = [self.root / 'results' / repo.replace('/', '--') / 'aggregate.json' for repo in REPOS]

    def run_analysis(self, paths=None):
        return h3.analyze(self.selection, self.aggregates if paths is None else paths,
                          self.root, [REMOTE, '/another/recorded/root'], draws=128, seed=29)

    def rebind_e1(self, old_path, new_path):
        """Rebind deliberately edited test sources, to exercise semantic checks."""
        frozen = base.read_json(self.selection)
        panel_path = self.root / 'panel.json'
        panel = base.read_json(panel_path)
        panel['models'][0]['primary'].update(
            path=REMOTE + '/' + new_path.relative_to(self.root).as_posix(), sha256=base.sha(new_path))
        write(panel_path, panel)
        frozen['panel_sha256'] = base.sha(panel_path)
        write(self.selection, frozen)
        aggregate = base.read_json(self.aggregates[0])
        del aggregate['file_sha256'][REMOTE + '/' + old_path.relative_to(self.root).as_posix()]
        aggregate['file_sha256'][REMOTE + '/' + new_path.relative_to(self.root).as_posix()] = base.sha(new_path)
        write(self.aggregates[0], aggregate)

    def test_full_schema_fixture_reconstructs_original_points_and_all_sources(self):
        report, arrays = self.run_analysis()
        self.assertEqual(report['status'], 'complete_verified_for_selected_panel')
        self.assertEqual(report['n_models'], 6)
        self.assertAlmostEqual(report['primary']['rho'], 1.)
        self.assertEqual([m['n_E1_items'] for m in report['models']], [180, 180, 200, 200, 200, 200])
        self.assertTrue(all(m['point_reproduced_from_components'] for m in report['models']))
        self.assertEqual(arrays['C_draws'].shape, (128, 6))
        self.assertEqual(len(report['primary']['leave_one_model_out']), 6)
        self.assertIn(str((self.root / h3.NOTE).resolve()), report['file_sha256'])
        self.assertTrue(report['scope']['H1_Holm_two_test_family_unchanged'])
        json.dumps(report, allow_nan=False)

    def test_missing_duplicate_and_unselected_models(self):
        with self.assertRaisesRegex(ValueError, 'Exactly six'):
            self.run_analysis(self.aggregates[:5])
        with self.assertRaisesRegex(ValueError, 'duplicate aggregate'):
            self.run_analysis(self.aggregates[:5] + self.aggregates[:1])
        value = base.read_json(self.aggregates[0])
        value['model_identity']['repo'] = 'unselected/model'
        write(self.aggregates[0], value)
        with self.assertRaisesRegex(ValueError, 'Unexpected or duplicate'):
            self.run_analysis()

    def test_source_sha_sidecar_and_point_tampering(self):
        cache = self.aggregates[0].parent / 'E4_0_caches.npz'
        original = cache.read_bytes()
        cache.write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'Unclosed aggregate sources'):
            self.run_analysis()
        cache.write_bytes(original)
        value = base.read_json(self.aggregates[0])
        value['single_round']['E4b']['metrics']['P0_unexplained_fraction_U']['estimate'] = 12.
        write(self.aggregates[0], value)
        with self.assertRaisesRegex(ValueError, 'Recomputed point differs'):
            self.run_analysis()

    def test_wrong_topic_panel_or_wrong_frozen_selection(self):
        value = base.read_json(self.aggregates[0])
        value['topics'][0] = 'another topic'
        write(self.aggregates[0], value)
        with self.assertRaisesRegex(ValueError, 'topic IDs/names'):
            self.run_analysis()
        frozen = base.read_json(self.selection)
        frozen['selected_repos'][0] = 'not/selected'
        write(self.selection, frozen)
        with self.assertRaisesRegex(ValueError, 'Invalid/incomplete frozen selection'):
            self.run_analysis()

    def test_rebound_e1_component_must_reproduce_original_C(self):
        e1 = self.aggregates[0].parent / 'e1/confirmatory180_R0_analysis.json'
        value = base.read_json(e1)
        value['item_C_numerators'][0] += 50
        write(e1, value)
        self.rebind_e1(e1, e1)
        with self.assertRaisesRegex(ValueError, 'Recomputed point differs'):
            self.run_analysis()

    def test_full200_cannot_replace_mandatory_fresh180_source(self):
        e1 = self.aggregates[0].parent / 'e1/confirmatory180_R0_analysis.json'
        wrong = e1.with_name('holdout200_R0_analysis.json')
        wrong.write_bytes(e1.read_bytes())
        self.rebind_e1(e1, wrong)
        with self.assertRaisesRegex(ValueError, 'Wrong E1 primary partition'):
            self.run_analysis()

    def test_point_U_zero_denominator_is_retained_not_removed(self):
        aggregate = base.read_json(self.aggregates[0])
        for stats in aggregate['single_round']['E4b']['topic_sufficient_statistics'].values():
            stats['observed_mean_square_contrast'] = 0.
        aggregate['single_round']['E4b']['metrics']['P0_unexplained_fraction_U']['estimate'] = None
        write(self.aggregates[0], aggregate)
        report, arrays = self.run_analysis()
        self.assertEqual(report['status'], 'complete_verified_for_selected_panel')
        self.assertEqual(report['n_models'], 6)
        self.assertIsNone(report['models'][0]['U'])
        self.assertIsNone(report['primary']['rho'])
        self.assertEqual(report['primary']['undefined_draws'], 128)
        self.assertTrue(np.isnan(arrays['U_draws'][:, 0]).all())
        self.assertEqual(report['bootstrap']['per_model_undefined_U_draws'][REPOS[0]], 128)
        json.dumps(report, allow_nan=False)

    def test_cli_default10000_multiple_aliases_and_sidecar(self):
        output = self.root / 'reports/test_H3.json'
        command = [sys.executable, str(Path(h3.__file__).resolve()), '--root', str(self.root),
                   '--selection', str(self.selection), '--output', str(output),
                   '--source-root', '/irrelevant/alias', '--source-root', REMOTE]
        for path in self.aggregates:
            command.extend(['--aggregate', str(path)])
        done = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        report = base.read_json(output)
        self.assertEqual(report['bootstrap']['draws'], 10000)
        artifact = output.parent / report['bootstrap_artifact']['filename']
        self.assertEqual(base.sha(artifact), report['bootstrap_artifact']['sha256'])
        with np.load(artifact) as arrays:
            self.assertEqual(arrays['C_draws'].shape, (10000, 6))
            self.assertEqual(arrays['E4_topic_weights'].shape, (10000, 6))
        again = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(again.returncode, 0)
        self.assertIn('Preserve existing H3 report', again.stderr)


if __name__ == '__main__':
    unittest.main()
