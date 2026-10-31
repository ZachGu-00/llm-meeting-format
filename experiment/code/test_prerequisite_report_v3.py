"""CPU checks for report completeness, provenance and explicit skipped scope."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import write_prerequisite_report_v3 as report


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='prerequisite_report_test_')
        self.addCleanup(self.temp.cleanup); self.root = Path(self.temp.name)

    def E0(self, verified):
        return {'models': [{'repo_id': 'fixture/model', 'closed_development100': True,
                            'integrity': {'status': 'verified_local' if verified else 'partial_local_sync'},
                            'status': 'complete', 'qualification': {'final_qualification': 'R0_primary'}}],
                'runtime_decision': {}, 'protocol_versions': {}}

    def test_missing_sources_never_close_and_E3_is_not_passed(self):
        with patch.object(report.e0, 'build', return_value=self.E0(False)), patch.object(
                report.e1, 'summarize_model', return_value={'repo_id': 'fixture/model', 'E1_complete': False, 'E1_status': 'pending_E0'}):
            value = report.build(self.root)
        self.assertEqual(value['status'], 'intermediate_pending')
        self.assertEqual(value['counts']['E0_verified'], 0)
        self.assertEqual(value['E3']['status'], 'skipped_by_user')
        self.assertFalse(value['E3']['counts_as_passed'])
        self.assertFalse(value['E4_completion_assessed'])
        self.assertIn('中间报告', report.markdown(value))

    def test_verified_primary_is_visible_while_other_E1_analyses_wait(self):
        primary = {'readout': 'R0', 'n_items': 200, 'C': -.4, 'C_interval95': [-.7, -.2],
                   'rules': {name: {'R2_within_item': -.2} for name in report.RULES}}
        row = {'repo_id': 'fixture/model', 'E1_complete': False, 'E1_status': 'incomplete_or_not_synced',
               'R0_operational': True, 'primary': primary}
        with patch.object(report.e0, 'build', return_value=self.E0(True)), patch.object(
                report.e1, 'summarize_model', return_value=row):
            value = report.build(self.root)
        text = report.markdown(value)
        self.assertIn('-0.4 [-0.7, -0.2]', text)
        self.assertIn('其他分析待闭合', text)
        self.assertIn('-0.2', text)
        self.assertEqual(value['status'], 'intermediate_pending')

    def test_source_mutation_is_rejected(self):
        path = self.root/'source.txt'; path.write_text('frozen')
        document = {'file_sha256': {str(path): report.e0.sha(path)}}
        verifier = report.e1.Verifier(self.root, '/workspace/voices_nonadditivity_v3')
        report.verify_sources(document, verifier, [path])
        path.write_text('changed')
        with self.assertRaises(ValueError): report.verify_sources(document, verifier, [path])

    def test_conditional_admission_is_explicit_and_does_not_close_E1(self):
        E0 = self.E0(True)
        E0['models'][0]['qualification'].update(conditional_option_mass_admission=True, operational_admission_branch='conditional')
        with patch.object(report.e0, 'build', return_value=E0), patch.object(
                report.e1, 'summarize_model', return_value={'repo_id': 'fixture/model', 'E1_complete': False, 'E1_status': 'pending'}):
            value = report.build(self.root)
        self.assertTrue(value['models'][0]['conditional_option_mass_admission'])
        self.assertIn('仅条件门通过', report.markdown(value))
        self.assertEqual(value['status'], 'intermediate_pending')

    def test_unknown_R0_identity_is_not_called_R1_only(self):
        verifier = report.e1.Verifier(self.root, '/workspace/voices_nonadditivity_v3')
        result = report.summarize_e2(self.root, 'fixture/model', {}, True, None, verifier)
        self.assertEqual(result['status'], 'pending_verified_E1_identity')
        result = report.summarize_e2(self.root, 'fixture/model', {'R0_operational': False}, True, None, verifier)
        self.assertEqual(result['status'], 'not_applicable_R1_only')

    def test_six_names_without_bound_panel_are_not_a_verified_selection(self):
        import json
        path = self.root/'selection.json'
        path.write_text(json.dumps({'schema': 'voices_v3_frozen_E4_model_selection_v1', 'status': 'frozen',
                                   'performed': True, 'selected_repos': [str(i) for i in range(6)],
                                   'panel_path': str(self.root/'missing_panel.json'), 'panel_sha256': 'missing'}))
        result = report.selection_summary(path, report.e1.Verifier(self.root, '/workspace/voices_nonadditivity_v3'), [])
        self.assertFalse(result['verified_frozen'])
        self.assertEqual(result['selected_repos'], [])

    def paired_cohort(self, strength=-2.4, t=None):
        def metric(value):
            return {'estimate': value, 'interval95': [value-.5, value+.5] if value is not None else None,
                    'undefined_draws': 0 if value is not None else 10000, 'undefined_draws_not_dropped': True}
        return {'paired_named_peer_differences': {
                    name: {'C': metric(0.), 'count_delta_1_0': metric(strength)} for name in (
                        'E2a_neutral0_minus_E1', 'E2a_neutral3_minus_neutral0', 'E2b_minus_E1', 'E2c_minus_E1')},
                'paired_count_grid_differences': {
                    name: {'count_grouping_contrast': metric(.1), 'power_t_common_1_to_3': metric(t)}
                    for name in ('E2d_minus_E1_matched_prefix', 'E2e_minus_E1_matched_prefix')},
                'count_grid_results': {'E2e': {'metrics': {
                    'power_t_common_1_to_3': metric(.3), 'power_t_native_range': metric(.4)}}}}

    def test_strength_summaries_copy_existing_paired_metrics_without_reestimating(self):
        cohort = self.paired_cohort()
        result = report.paired_summary(cohort)
        self.assertNotIn('empty10_minus_E1_C', result)
        self.assertIs(result['neutral0_minus_E1_C'],
                      cohort['paired_named_peer_differences']['E2a_neutral0_minus_E1']['C'])
        self.assertIs(result['neutral3_minus_neutral0_single_opponent_increment'],
                      cohort['paired_named_peer_differences']['E2a_neutral3_minus_neutral0']['count_delta_1_0'])
        self.assertIs(result['anonymous_minus_E1_single_opponent_increment'],
                      cohort['paired_named_peer_differences']['E2c_minus_E1']['count_delta_1_0'])
        self.assertIs(result['stance_only_minus_E1_power_t_common_1_to_3'],
                      cohort['paired_count_grid_differences']['E2d_minus_E1_matched_prefix']['power_t_common_1_to_3'])
        self.assertEqual(result['stance_only_minus_E1_power_t_common_1_to_3']['undefined_draws'], 10000)
        self.assertIsNone(result['stance_only_minus_E1_power_t_common_1_to_3']['interval95'])

    def test_markdown_shows_all_completed_models_and_fresh44_strength_even_when_C_crosses_zero(self):
        models = []
        for index, strength in enumerate((-2.4, 1.7)):
            main = report.paired_summary(self.paired_cohort(strength))
            fresh = report.paired_summary(self.paired_cohort(strength-.1)) if index == 0 else None
            models.append({'repo_id': 'fixture/model'+str(index), 'E0_verified': False, 'E0': {}, 'E1': {},
                           'E2': {'status': 'complete_verified', 'complete_verified': True,
                                  'major_comparisons': main, 'fresh44_sensitivity': fresh}})
        models.append({'repo_id': 'fixture/pending', 'E0_verified': False, 'E0': {}, 'E1': {},
                       'E2': {'status': 'stage_analyses_pending', 'complete_verified': False}})
        value = {'status': 'intermediate_pending', 'models': models,
                 'counts': {'E0_verified': 0, 'E1_verified': 0, 'E2_verified': 2,
                            'E2_not_applicable_R1_only': 0, 'requested_models': 3},
                 'E4_selection': {'verified_frozen': False, 'selected_repos': []}, 'limitations': []}
        text = report.markdown(value)
        table = text.split('| 模型 | 加3中性−0中性：单反对者增量差 |', 1)[1].split('单反对者增量指', 1)[0]
        self.assertIn('E2 配对诊断 2/3 个', text)
        self.assertIn('5槽无中性背景−E1 ΔC', text)
        self.assertNotIn('10槽空背景', text)
        self.assertIn('固定5个槽位、10种排列', text)
        self.assertIn('fixture/model0 | -2.4 [-2.9, -1.9]', table)
        self.assertIn('fixture/model1 | 1.7 [1.2, 2.2]', table)
        self.assertNotIn('fixture/pending', table)
        self.assertIn('C 近似不变不代表影响强度不变', text)
        self.assertIn('同一未见44条的影响强度：加3中性−0中性单反对者增量差=-2.5 [-3, -2]', text)
        self.assertIn('未重估或作多重比较校正', text)


if __name__ == '__main__':
    unittest.main()
