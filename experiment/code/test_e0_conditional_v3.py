"""CPU-only tests of conditional policy, symmetric routing and evidence preservation."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import analyze_e0_v3_conditional as policy
import run_e0_v3_conditional as driver
import summarize_e0_panel_v3 as panel
import test_e0_taskpilot_v3 as taskpilot_tests


def source(q=.5982457, fmt=.98, r2=.96, n=100, opposite=False):
    gates = {key: {'value': value, 'status': 'failed' if key == 'option_mass_median' else 'pass'}
             for key, value in zip(policy.NORMAL, (q, fmt, r2))}
    return {'schema': 'voices_v3_E0_v1', 'n_items': n, 'bootstrap_draws': 10000,
        'operational_gates': gates, 'qualification': {'R1_candidate_scoring_valid': True, 'tier': 'R1_only'},
        'applicability_diagnostics': {'weight_directions': {'w_same': {'status': 'clear_opposite' if opposite else 'consistent'}}},
        'strict_original_gates': {'R0_R1_primary_agreement': {'value': {'w_same': .1, 'w_diff': .1}}},
        'file_sha256': {}}


class ConditionalE0Tests(unittest.TestCase):
    def test_union_boundaries_and_missing_values(self):
        cases = [(.70, .90, .85, 'normal', 'pass'), (.50, .95, .90, 'conditional', 'pass'),
                 (.5982457, .98, .96, 'conditional', 'pass'), (.534612, 1, .965, 'conditional', 'pass'),
                 (.499999, 1, 1, None, 'failed'), (.60, .949999, 1, None, 'failed'),
                 (.60, 1, .899999, None, 'failed'), (None, 1, 1, None, 'incomplete')]
        for q, fmt, r2, branch, status in cases:
            with self.subTest(q=q, fmt=fmt, r2=r2):
                result = policy.apply_policy(source(q, fmt, r2))
                self.assertEqual(result['operational_policy']['selected_branch'], branch)
                self.assertEqual(result['qualification']['R0_operational'], status)
                self.assertEqual(result['qualification']['conditional_option_mass_admission'], branch == 'conditional')

    def test_preserves_original_gates_and_opposite_sign_only_changes_readout(self):
        original = source(opposite=True); previous = copy.deepcopy(original)
        result = policy.apply_policy(original)
        self.assertEqual(original, previous)
        self.assertEqual(result['operational_gates'], original['operational_gates'])
        self.assertEqual(result['qualification']['tier'], 'R1_primary_R0_operational')
        self.assertEqual(result['qualification']['R0_operational'], 'pass')
        self.assertFalse(result['operational_policy']['scientific_effects_used_for_remedy_selection'])

    def run_scenario(self, root, measurements, no_legacy_skip=False):
        code, out, amendment, inputs, argv = taskpilot_tests.TaskPilotRecoveryTests().fixture(root)
        statepath = out / 'e0_driver_state.json'
        if no_legacy_skip:
            old = json.loads(statepath.read_text()); old['attempts'] = []; statepath.write_text(json.dumps(old))
        previous = statepath.read_bytes(); previous_hash = driver.sha(statepath)
        scored = []
        def fake_call(command, log):
            output = Path(command[command.index('--output') + 1]); output.write_text('{}')
            if '--mode' in command:
                self.assertEqual(command[command.index('--mode') + 1], 'score')
                scored.append((output.parent.name, output.name.split('_')[0]))
                Path(str(output) + '.summary.json').write_text('{}')
        def fake_analysis(ap, inp, raw, lock, output, amendment):
            stage = Path(ap).name.split('_')[0]; remedy = Path(ap).parent.name
            args = measurements.get((remedy, stage), (.1, .01, .01))
            result = policy.apply_policy(source(*args, n=100 if stage == 'development' else 12))
            Path(output).write_text(json.dumps(result)); return result
        with patch.object(driver, 'HERE', code), patch.object(driver.sys, 'argv', argv), \
                patch.dict(driver.os.environ, {'STY': 'CPU-mocked-no-GPU'}), \
                patch.object(driver, 'call', side_effect=fake_call), \
                patch.object(driver.conditional, 'ensure_analysis', side_effect=fake_analysis), \
                contextlib.redirect_stdout(io.StringIO()):
            driver.main()
        state = json.loads(statepath.read_text())
        self.assertEqual(state['schema'], driver.SCHEMA)
        self.assertEqual(state['previous_state_sha256'], previous_hash)
        self.assertEqual(Path(state['previous_state_snapshot']).read_bytes(), previous)
        self.assertEqual(state['amendment_sha256'], driver.sha(amendment))
        self.assertTrue(state['selected_analysis'].endswith('_E0_conditional_analysis.json'))
        self.assertFalse(state['calibration_used_for_admission'])
        return state, scored

    def test_actual_T2_conditional_pass_gets_full100_and_prevents_catastrophic_prefill(self):
        with tempfile.TemporaryDirectory() as folder:
            result, scored = self.run_scenario(Path(folder),
                {('t2', 'pilot'): (.5982457, .98, .96), ('t2', 'development'): (.58, .98, .95)})
            self.assertEqual(scored, [('original', 'pilot'), ('t2', 'pilot'), ('t2', 'development')])
            self.assertEqual(result['selected_remedy'], 't2')
            self.assertTrue(result['qualification']['conditional_option_mass_admission'])

    def test_models_without_calibration_skips_receive_same_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            result, scored = self.run_scenario(Path(folder),
                {('original', 'pilot'): (.99, 1, .99), ('original', 'development'): (.99, 1, .99)}, True)
            self.assertEqual(scored, [('original', 'pilot'), ('original', 'development')])
            self.assertEqual(result['selected_remedy'], 'original')
            self.assertFalse(result['qualification']['conditional_option_mass_admission'])

    def test_pilot_pass_full100_failure_advances_and_R1_fallback_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            result, scored = self.run_scenario(Path(folder), {('original', 'pilot'): (.6, .99, .99)})
            self.assertEqual(scored, [('original', 'pilot'), ('original', 'development'), ('t2', 'pilot'),
                                     ('prefill', 'pilot'), ('prefill', 'development')])
            self.assertEqual(result['selected_remedy'], 'prefill')
            self.assertEqual(result['final_qualification'], 'R1_only')

    def test_new_analysis_hash_binding_immutable_reuse_and_conditional_sidecar_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); original = root / 'development_E0_analysis.json'; original.write_text('original bytes')
            amendment = root / 'amendment.txt'; amendment.write_text('fixed union policy')
            output = root / 'development_E0_conditional_analysis.json'
            before = original.read_bytes()
            with patch.object(policy, 'validate_existing_analysis', return_value=source()):
                result = policy.ensure_analysis(original, root/'input', root/'raw', root/'lock', output, amendment)
                self.assertEqual(policy.ensure_analysis(original, root/'input', root/'raw', root/'lock', output, amendment), result)
                self.assertEqual(original.read_bytes(), before)
                self.assertEqual(result['file_sha256'][str(original.resolve())], driver.sha(original))
                self.assertEqual(result['file_sha256'][str(amendment.resolve())], driver.sha(amendment))
                amendment.write_text('changed policy')
                with self.assertRaisesRegex(ValueError, 'different bindings'):
                    policy.ensure_analysis(original, root/'input', root/'raw', root/'lock', output, amendment)
            raw = root / 'development_R0_R1.jsonl'
            rp = Path(str(raw) + '.run.json'); sp = Path(str(raw) + '.summary.json')
            rp.write_text('run'); sp.write_text('summary')
            result = {'file_sha256': {str(original): driver.sha(original)}, 'execution_integrity': {'R0': {
                'status': 'complete', 'run_sha256': driver.sha(rp), 'summary_sha256': driver.sha(sp)}}}
            self.assertEqual(panel.verify_analysis(output, result, root, '/workspace/test', {})['status'], 'verified_local')


if __name__ == '__main__':
    unittest.main()
