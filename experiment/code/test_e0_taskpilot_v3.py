"""CPU-only routing and preservation tests; no actual model/subprocess execution."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_e0_v3_taskpilot as driver


class TaskPilotRecoveryTests(unittest.TestCase):
    def fixture(self, root):
        code=root/'code';code.mkdir();out=root/'results/test--model/e0';out.mkdir(parents=True)
        for name in ('inference_v3.py','loader_v3.py'):(code/name).write_text(name)
        (root/'execution_protocol.txt').write_text('unchanged E0 protocol')
        amendment=root/'amendment.txt';amendment.write_text('token IDs only; task quality gate unchanged')
        inputs={name:root/(name+'.jsonl') for name in ('calibration','pilot','development')}
        for name,path in inputs.items():path.write_text(name)
        old={'schema':'voices_v3_E0_driver_v1','repo_id':'test/model','revision':'rev','batch_size':16,
             'attn_implementation':'eager','status':'complete','selected_remedy':'prefill','final_qualification':'R1_only',
             'attempts':[{'remedy':m,'decision':'advance_due_to_calibration_format_or_q'} for m in ('original','t2')]}
        state=out/'e0_driver_state.json';state.write_text(json.dumps(old))
        for remedy,q,fmt in [('original',.154,0),('t2',.6716948,1),('prefill',.0109,0)]:
            folder=out/remedy;folder.mkdir()
            engine={'code_sha256':driver.sha(code/'inference_v3.py'),'native_loader_sha256':driver.sha(code/'loader_v3.py'),
                'parent_loader':{'repo_id':'test/model','revision':'rev','attn_implementation':'eager'},'remedy':remedy}
            lock={'input_file_sha256':driver.sha(inputs['calibration']),'engine':engine,'batch_size':16,
                  'median_q':q,'greedy_any_variant_fraction':fmt}
            lock['lock_content_sha256']=driver.canonical_hash(lock)
            (folder/'token_calibration.json').write_text(json.dumps(lock))
        argv=['test','--model-dir',str(root/'model'),'--repo-id','test/model','--revision','rev',
              '--calibration-input',str(inputs['calibration']),'--pilot-input',str(inputs['pilot']),
              '--development-input',str(inputs['development']),'--output-dir',str(out),'--batch-size','16',
              '--amendment-file',str(amendment)]
        return code,out,amendment,inputs,argv

    def run_scenario(self, root, operational, final_tier='R0_primary'):
        code,out,amendment,inputs,argv=self.fixture(root);state=out/'e0_driver_state.json'
        previous=state.read_bytes();old_hash=driver.sha(state)
        lock_hashes={m:driver.sha(out/m/'token_calibration.json') for m in driver.MODES};scored=[]
        def fake_call(command,log):
            output=Path(command[command.index('--output')+1]);output.write_text('{}')
            if '--mode' in command:
                self.assertEqual(command[command.index('--mode')+1],'score')
                scored.append((output.parent.name,output.name.split('_')[0]))
                Path(str(output)+'.summary.json').write_text('{}')
        def fake_analysis(path,*args):
            stage=Path(path).name.split('_')[0];remedy=Path(path).parent.name
            passed=operational.get((remedy,stage),False)
            return {'operational_gates':{'option_mass_median':{'status':'pass' if passed else 'failed'}},
                'qualification':{'R0_operational':'pass' if passed else 'failed','R1_candidate_scoring_valid':True,
                    'final_qualification':(final_tier if passed else 'R1_only') if stage=='development' else 'pilot_only',
                    'tier':final_tier if passed else 'R1_only'}}
        with patch.object(driver,'HERE',code),patch.object(driver.sys,'argv',argv),patch.dict(driver.os.environ,{'STY':'test-no-GPU'}),\
                patch.object(driver,'call',side_effect=fake_call),patch.object(driver,'validate_existing_analysis',side_effect=fake_analysis),\
                contextlib.redirect_stdout(io.StringIO()):
            driver.main()
        result=json.loads(state.read_text())
        self.assertEqual(Path(result['previous_state_snapshot']).read_bytes(),previous)
        self.assertEqual(result['previous_state_sha256'],old_hash)
        self.assertEqual(result['amendment_sha256'],driver.sha(amendment))
        self.assertFalse(result['calibration_used_for_admission'])
        for remedy,digest in lock_hashes.items():self.assertEqual(driver.sha(out/remedy/'token_calibration.json'),digest)
        return result,scored

    def test_low_unrelated_calibration_q_still_tests_T2_and_ignores_effect_tier(self):
        with tempfile.TemporaryDirectory() as folder:
            result,scored=self.run_scenario(Path(folder),{('t2','pilot'):True,('t2','development'):True},'R1_primary_R0_operational')
            self.assertEqual(scored,[('original','pilot'),('t2','pilot'),('t2','development')])
            self.assertEqual(result['selected_remedy'],'t2')
            self.assertEqual(result['final_qualification'],'R1_primary_R0_operational')

    def test_full_failure_advances_and_last_prefill_keeps_R1_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            result,scored=self.run_scenario(Path(folder),{('original','pilot'):True})
            self.assertEqual(scored,[('original','pilot'),('original','development'),('t2','pilot'),('prefill','pilot'),('prefill','development')])
            self.assertEqual(result['selected_remedy'],'prefill');self.assertEqual(result['final_qualification'],'R1_only')

    def test_first_full_task_pass_stops_without_later_remedies(self):
        with tempfile.TemporaryDirectory() as folder:
            result,scored=self.run_scenario(Path(folder),{('original','pilot'):True,('original','development'):True})
            self.assertEqual(scored,[('original','pilot'),('original','development')])
            self.assertEqual(result['selected_remedy'],'original')

    def test_unskipped_models_are_not_routed(self):
        self.assertFalse(driver.recovery_eligibility({'attempts':[{'remedy':'original','decision':'retain_earliest_operational_variant'}]})['eligible'])
        self.assertTrue(driver.recovery_eligibility({'schema':driver.SCHEMA,'original_calibration_only_skips':['t2']})['eligible'])

    def test_closed_score_reuse_requires_exact_inputs_calibration_and_sidecars(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=root/'pilot_R0_R1.jsonl';raw.write_text('{}\n')
            inp=root/'input.jsonl';inp.write_text('{}\n')
            lockpath=root/'token_calibration.json';lockpath.write_text(json.dumps({'engine':{'id':'fixed'},'batch_size':16}))
            run={'engine':{'id':'fixed'},'batch_size':16,'n_input':1,'readout':'R0_R1',
                 'input_file_sha256':driver.sha(inp),'calibration_file_sha256':driver.sha(lockpath)}
            runpath=Path(str(raw)+'.run.json');runpath.write_text(json.dumps(run))
            summary={**run,'status':'complete','n_output':1,'output_sha256':driver.sha(raw),'run_file_sha256':driver.sha(runpath)}
            summarypath=Path(str(raw)+'.summary.json');summarypath.write_text(json.dumps(summary))
            analysis={'n_items':12,'bootstrap_draws':10000,'file_sha256':{str(inp):driver.sha(inp),str(raw):driver.sha(raw)},
                'execution_integrity':{'R0':{'status':'complete','run_sha256':driver.sha(runpath),'summary_sha256':driver.sha(summarypath)}}}
            apath=root/'pilot_E0_analysis.json';apath.write_text(json.dumps(analysis))
            self.assertEqual(driver.validate_existing_analysis(apath,inp,raw,lockpath),analysis)
            summarypath.write_text(json.dumps(summary|{'n_output':2}))
            with self.assertRaisesRegex(ValueError,'bindings disagree'):driver.validate_existing_analysis(apath,inp,raw,lockpath)


if __name__=='__main__':unittest.main()
