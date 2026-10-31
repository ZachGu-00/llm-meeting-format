"""Protect pilot/full-panel counting and separate runtime failures from readout failures."""
import json
from pathlib import Path
import tempfile
import unittest
from summarize_e0_panel_v3 import build, local_path, markdown, sha


class E0ProgressTests(unittest.TestCase):
    def make_tree(self, root, n=100, error=False):
        setup = root / 'setup'; setup.mkdir()
        (setup / 'download_queue.json').write_text(json.dumps([{'repo_id': 'org/model'}]))
        (setup / 'runtime_decision.json').write_text(json.dumps({'status': 'selected', 'batch_size': 16,
            'attn_implementation': 'eager', 'rationale': 'fixed pilot before E0', 'limitations': 'not exact equivalence'}))
        if error:
            (setup / 'e0_queue_status.json').write_text(json.dumps([{'model': 'org/model', 'exit_code': 1,
                'status': 'execution_error_not_model_disqualification'}]))
        folder = root / 'results/org--model/e0/original'; folder.mkdir(parents=True)
        raw = folder / 'development_R0_R1.jsonl'; raw.write_text('{}\n')
        run = Path(str(raw) + '.run.json'); run.write_text('{}')
        summary = Path(str(raw) + '.summary.json'); summary.write_text('{}')
        remote = '/workspace/voices_nonadditivity_v3/'
        q = {'tier': 'R0_primary', 'primary_readout': 'R0', 'development100_complete': n == 100,
             'final_qualification': 'R0_primary' if n == 100 else 'pilot_only'}
        gates = {k: {'value': v, 'status': 'pass'} for k, v in
                 [('option_mass_median', .8), ('first_token_format', .98), ('R2_concordance', .95)]}
        strict = {k: dict(v) for k, v in gates.items()}; strict['option_mass_median']['status'] = 'failed'
        analysis = folder / 'development_E0_analysis.json'
        analysis.write_text(json.dumps({'n_items': n, 'n_rows': n * 122, 'bootstrap_draws': 10000,
            'qualification': q, 'operational_gates': gates, 'strict_original_gates': strict,
            'applicability_diagnostics': {}, 'file_sha256': {remote + raw.relative_to(root).as_posix(): sha(raw)},
            'execution_integrity': {'R0': {'status': 'complete', 'run_sha256': sha(run), 'summary_sha256': sha(summary)}}}))
        (folder.parent / 'e0_driver_state.json').write_text(json.dumps({'repo_id': 'org/model', 'status': 'complete',
            'selected_remedy': 'original', 'selected_analysis': remote + analysis.relative_to(root).as_posix(), 'attempts': []}))
        return raw

    def test_remote_path_maps_to_local_without_modifying_receipt(self):
        root = Path('local-experiment')
        self.assertEqual(local_path('/workspace/voices_nonadditivity_v3/results/x', root), root / 'results/x')

    def test_closed_panel_keeps_strict_and_relaxed_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.make_tree(root); report = build(root)
            self.assertEqual(report['counts']['closed_development100_locally_verified'], 1)
            row = report['models'][0]
            self.assertEqual(row['strict_operational_status'], 'failed')
            self.assertEqual(row['relaxed_operational_status'], 'pass')
            self.assertIn('不代表 E0–E3', markdown(report))

    def test_pilot_and_runtime_error_are_not_full_panel_or_instruction_failure(self):
        for n, error in [(12, False), (100, True)]:
            with self.subTest(n=n, error=error), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); self.make_tree(root, n, error); report = build(root)
                self.assertEqual(report['counts']['closed_development100'], 0)
                self.assertEqual(report['counts']['closed_R0_operational_failed'], 0)
                self.assertEqual(report['counts']['runtime_errors'], int(error))
                if n == 12:
                    self.assertEqual(report['models'][0]['stage'], 'pilot_only')

    def test_missing_sync_and_changed_source_are_different(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = self.make_tree(root); raw.unlink()
            report = build(root)
            self.assertEqual(report['counts']['closed_development100'], 1)
            self.assertEqual(report['counts']['closed_development100_locally_verified'], 0)
            self.assertEqual(report['models'][0]['integrity']['status'], 'partial_local_sync')
            raw.write_text('changed')
            report = build(root)
            self.assertEqual(report['counts']['closed_development100'], 0)
            self.assertEqual(report['models'][0]['status'], 'integrity_error')

    def test_explicit_E0_protocol_archive_requires_matching_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.make_tree(root)
            (root / 'provenance').mkdir()
            archived = root / 'provenance/execution_protocol_E0_original.txt'; archived.write_text('original E0')
            (root / 'execution_protocol.txt').write_text('later pre-E1 clarification')
            path = root / 'results/org--model/e0/original/development_E0_analysis.json'
            analysis = json.loads(path.read_text())
            analysis['file_sha256']['/workspace/voices_nonadditivity_v3/execution_protocol.txt'] = sha(archived)
            path.write_text(json.dumps(analysis))
            report = build(root)
            self.assertEqual(report['counts']['closed_development100_locally_verified'], 1)
            self.assertEqual(len(report['models'][0]['integrity']['source_snapshot_substitutions']), 1)
            archived.write_text('wrong historical version')
            self.assertEqual(build(root)['counts']['closed_development100'], 0)


if __name__ == '__main__':
    unittest.main()
