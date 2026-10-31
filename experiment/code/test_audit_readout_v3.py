"""Synthetic CPU tests: streaming evidence integrity and bounded q diagnostics."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import audit_readout_v3 as audit


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def lines(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')


def fixture(root,stage='pilot',n=4,remedy='t2',qs=None):
    folder=root/'results/test--model/e0'/remedy;folder.mkdir(parents=True,exist_ok=True)
    calinput=root/'data/calibration_prompts.jsonl'
    if not calinput.exists():lines(calinput,[{'id':'cal0'}])
    lock={'input_file_sha256':audit.sha(calinput),'engine':{'remedy':remedy},'batch_size':16,
          'median_q':.3,'greedy_any_variant_fraction':1,'option_ids':{'A':1,'B':2}}
    lock['lock_content_sha256']=__import__('hashlib').sha256(audit.encoded(lock)).hexdigest()
    lockpath=folder/'token_calibration.json';write(lockpath,lock)
    inputs=[{'id':f'{stage}{i}','item_id':f'item{i}','subset_mask':0 if i%2==0 else 3,
             'label_map':i%2,'positive_letter':'A' if i%2==0 else 'B','permutation':i%2,
             'messages':[{'role':'user','content':f'synthetic {i}'}]} for i in range(n)]
    inp=root/'data'/(stage+'7_prompts.jsonl');lines(inp,inputs)
    normalized=[r|{'record_id':r['id']} for r in inputs]
    h=lambda value:__import__('hashlib').sha256(audit.encoded(value)).hexdigest()
    run={'n_input':n,'readout':'R0_R1','mode':'score',
        'input_file_sha256':audit.sha(inp),'input_record_ids_sha256':h([r['id'] for r in inputs]),
        'input_rows_sha256':h(normalized),'calibration_file_sha256':audit.sha(lockpath),
        'engine':lock['engine'],'option_ids':lock['option_ids'],'batch_size':16,
        'format_record_ids':[r['id'] for r in inputs],'r2_record_ids':[r['id'] for r in inputs]}
    raw=folder/(stage+'_R0_R1.jsonl');rp=Path(str(raw)+'.run.json');write(rp,run)
    values=qs or [.6]*n
    rows=[r|{'status':'ok','run_file_sha256':audit.sha(rp),'q':values[i],
        'R0_y_AB':1 if i%2==0 else -1,'greedy_argmax_any_variant_letter':'A' if i%2==0 else 'B',
        'diagnostic_format_selected':True,'diagnostic_r2_selected':True,
        'generated_first_token_id':1 if i%2==0 else 2,'generated_first_token_text':' A' if i%2==0 else 'B'}
        for i,r in enumerate(normalized)]
    lines(raw,rows)
    sp=Path(str(raw)+'.summary.json')
    summary=run|{'status':'complete','n_output':n,'exact_id_and_count_equality':True,
        'run_file_sha256':audit.sha(rp),'output_sha256':audit.sha(raw),'output_bytes':raw.stat().st_size}
    write(sp,summary)
    analysis={'n_items':n,'bootstrap_draws':10000,'qualification':{'tier':'R0_primary'},
        'operational_gates':{'option_mass_median':{'value':float(np.median(values))},
            'first_token_format':{'value':1.},'R2_concordance':{'value':1.}},
        'file_sha256':{str(inp):audit.sha(inp),str(raw):audit.sha(raw)},
        'execution_integrity':{'R0':{'status':'complete','run_sha256':audit.sha(rp),'summary_sha256':audit.sha(sp)}}}
    ap=folder/(stage+'_E0_analysis.json');write(ap,analysis)
    return raw,inp,lockpath,ap


class ReadoutAuditTests(unittest.TestCase):
    def test_exact_and_bounded_quantiles_with_strict_tail_counts(self):
        exact=audit.QStats(10);bounded=audit.QStats(2)
        for value in [0,.5,.7,.9,1]:exact.add(value);bounded.add(value)
        exact.add(None);exact.add(True)
        a,b=exact.report(),bounded.report()
        self.assertEqual(a['below_counts'],{'0.1':1,'0.5':1,'0.7':2,'0.9':3})
        self.assertEqual(a['quantiles']['0.5'],.7)
        self.assertEqual(a['invalid_or_missing_q_rows'],2)
        self.assertIsNone(bounded.values)
        self.assertLessEqual(b['quantile_intervals']['0.5'][0],.7)
        self.assertGreaterEqual(b['quantile_intervals']['0.5'][1],.7)

    def test_stream_grouped_q_and_both_effect_signs_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw,inp,lock,ap=fixture(root,qs=[.1,.5,.7,.9])
            result=audit.audit_closed_raw(raw,inp,lock,root)
            self.assertEqual(result['status'],'verified_closed');self.assertEqual(result['n_rows'],4)
            self.assertAlmostEqual(result['q']['quantiles']['0.5'],.6)
            self.assertEqual(result['q_by']['subset_size']['0']['valid_q_rows'],2)
            self.assertAlmostEqual(result['q_by']['positive_letter']['B']['mean'],.7)
            self.assertEqual(result['generated_diagnostics']['format']['fraction'],1)
            self.assertEqual(result['generated_diagnostics']['R2']['fraction'],1)

    def test_changed_raw_source_or_closed_sidecar_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw,inp,lock,ap=fixture(root)
            rows=[json.loads(s) for s in raw.read_text().splitlines()];rows[0]['subset_mask']=31;lines(raw,rows)
            self.assertEqual(audit.safe_raw(raw,inp,lock,root)['status'],'audit_error')
            raw,inp,lock,ap=fixture(root)
            sp=Path(str(raw)+'.summary.json');summary=audit.load(sp);summary['n_output']=5;write(sp,summary)
            self.assertEqual(audit.safe_raw(raw,inp,lock,root)['status'],'audit_error')

    def test_actual_generation_tie_and_missing_sample_are_not_silent_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw,inp,lock,ap=fixture(root)
            rows=[json.loads(s) for s in raw.read_text().splitlines()]
            rows[0]['R0_y_AB']=0;rows[1]['diagnostic_format_selected']=False;lines(raw,rows)
            sp=Path(str(raw)+'.summary.json');summary=audit.load(sp)
            summary.update(output_sha256=audit.sha(raw),output_bytes=raw.stat().st_size);write(sp,summary)
            result=audit.audit_closed_raw(raw,inp,lock,root)
            self.assertEqual(result['generated_diagnostics']['R2']['ties'],1)
            self.assertEqual(result['generated_diagnostics']['R2']['fraction'],.75)
            self.assertIsNone(result['generated_diagnostics']['format']['fraction'])

    def test_current_conditional_state_overrides_old_remedy_and_binds_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'results/test--model/e0'
            fixture(root,'pilot',12);_,_,lock,ap=fixture(root,'development',100)
            oldpath=folder/'provenance/old.json'
            write(oldpath,{'selected_remedy':'prefill','final_qualification':'R1_only',
                'attempts':[{'remedy':'t2','decision':'advance_due_to_calibration_format_or_q'}]})
            amendment=root/'provenance/amendment.txt';amendment.parent.mkdir(exist_ok=True);amendment.write_text('union')
            attempts={'remedy':'t2'}
            for stage in ('pilot','development'):
                source=folder/'t2'/(stage+'_E0_analysis.json');target=folder/'t2'/(stage+'_E0_conditional_analysis.json')
                value=audit.load(source);value['operational_policy']={'selected_branch':'conditional','conditional_only':True}
                value['file_sha256'][str(source)]=audit.sha(source);write(target,value)
                attempts[stage]={'analysis':str(target)}
            state={'schema':audit.CONDITIONAL,'status':'complete','selected_remedy':'t2','selected_calibration':str(lock),
                'selected_analysis':attempts['development']['analysis'],'previous_state_snapshot':str(oldpath),
                'previous_state_sha256':audit.sha(oldpath),'amendment_file':str(amendment),
                'amendment_sha256':audit.sha(amendment),'original_calibration_only_skips':['t2'],'attempts':[attempts]}
            write(folder/'e0_driver_state.json',state)
            result=audit.analyze_e0(folder,root,250000)
            self.assertEqual(result['status'],'selected_measurements_verified')
            self.assertEqual(result['selected_remedy'],'t2')
            self.assertEqual(result['selection_provenance'],'conditional_mass_reassessed')
            self.assertEqual(result['prior_state_chain'][0]['selected_remedy'],'prefill')
            self.assertFalse(result['initial_queue_status_used_to_select_model'])
            oldpath.write_text('{}')
            with self.assertRaisesRegex(ValueError,'snapshot hash'):audit.analyze_e0(folder,root,250000)

    def test_missing_raw_is_incomplete_sync_not_failed_quality(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw,inp,lock,ap=fixture(root);raw.unlink()
            self.assertEqual(audit.safe_raw(raw,inp,lock,root)['status'],'not_closed_or_not_synced')


if __name__=='__main__':unittest.main()
