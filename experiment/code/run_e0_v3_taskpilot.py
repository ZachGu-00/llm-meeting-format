"""Recover E0 calibration-only skips using actual-task pilots, without changing inference."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import analyze_e0_v3 as analysis
from inference_v3 import canonical_hash,sha

HERE=Path(__file__).resolve().parent
MODES=('original','t2','prefill')
SCHEMA='voices_v3_E0_taskpilot_driver_v1'
AMENDMENT=HERE.parent/'provenance/E0_taskpilot_amendment_20261031.txt'


def recovery_eligibility(state):
    skipped=[a['remedy'] for a in state.get('attempts',[])
             if a.get('decision')=='advance_due_to_calibration_format_or_q']
    if state.get('schema')==SCHEMA:
        skipped=state.get('original_calibration_only_skips',[])
    return {'eligible':bool(skipped),'calibration_only_skips':skipped,
            'reason':'earlier_remedies_skipped_without_actual_task_pilot' if skipped else 'no_calibration_only_skips'}


def archive_previous_state(statepath):
    statepath=Path(statepath);digest=sha(statepath)
    archive=statepath.parent/'provenance'/('e0_driver_state_before_taskpilot_'+digest+'.json')
    archive.parent.mkdir(parents=True,exist_ok=True)
    if archive.exists():
        if sha(archive)!=digest:raise ValueError('Existing prior-state snapshot changed')
    else:
        with archive.open('xb') as stream:
            stream.write(statepath.read_bytes());stream.flush();os.fsync(stream.fileno())
    return archive,digest


def write_state(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def call(command,log):
    with Path(log).open('a',encoding='utf-8') as stream:
        stream.write('\nInvocation UTC '+datetime.now(timezone.utc).isoformat()+'\n');stream.flush()
        rc=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=False).returncode
    if rc:raise RuntimeError('E0 subprocess failed; inspect '+str(log)+' (exit '+str(rc)+')')


def validate_existing_analysis(path,input_path,result_path,lockpath):
    d=json.loads(Path(path).read_text(encoding='utf-8'))
    for p,digest in d['file_sha256'].items():
        if sha(p)!=digest:raise ValueError('Existing E0 analysis source changed: '+p)
    runpath=Path(str(result_path)+'.run.json');summarypath=Path(str(result_path)+'.summary.json')
    run=json.loads(runpath.read_text(encoding='utf-8'));summary=json.loads(summarypath.read_text(encoding='utf-8'))
    receipt=d['execution_integrity']['R0']
    if (summary.get('status')!='complete' or summary.get('n_output')!=run['n_input'] or
        summary.get('output_sha256')!=sha(result_path) or summary.get('run_file_sha256')!=sha(runpath) or
        receipt.get('status')!='complete' or receipt.get('run_sha256')!=sha(runpath) or
        receipt.get('summary_sha256')!=sha(summarypath) or run['readout']!='R0_R1' or
        run['input_file_sha256']!=sha(input_path) or run['calibration_file_sha256']!=sha(lockpath) or
        any(summary.get(key)!=value for key,value in run.items())):
        raise ValueError('Existing closed E0 result/analysis bindings disagree')
    lock=json.loads(Path(lockpath).read_text(encoding='utf-8'))
    if run['engine']!=lock['engine'] or run['batch_size']!=lock['batch_size']:
        raise ValueError('Existing scores use another calibration engine/batch')
    expected_n=100 if Path(path).name.startswith('development_') else 12
    if d.get('n_items')!=expected_n or d.get('bootstrap_draws')!=10000:
        raise ValueError('Recovery requires the fixed task pilot12 or full development100 analysis')
    if expected_n==100 and not d['qualification'].get('development100_complete'):
        raise ValueError('Development analysis is not a complete100-item qualification')
    return d


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('model-dir','repo-id','revision','calibration-input','pilot-input','development-input','output-dir'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--batch-size',type=int,required=True)
    parser.add_argument('--model-kind',choices=['auto','chat','base'],default='auto')
    parser.add_argument('--attn-implementation',choices=['eager','sdpa','flash_attention_2'],default='eager')
    parser.add_argument('--max-length',type=int,default=8192)
    parser.add_argument('--stop-after',choices=['pilot','development'],default='development')
    parser.add_argument('--amendment-file',type=Path,default=AMENDMENT)
    parser.add_argument('--audit-only',action='store_true',help='Read existing routing and exit without writes or GPU work')
    a=parser.parse_args()
    if a.batch_size<1:raise ValueError('batch_size must be positive; this driver does not tune batching')
    out=Path(a.output_dir);statepath=out/'e0_driver_state.json'
    if not statepath.is_file():raise ValueError('Task-pilot recovery requires an existing E0 driver state')
    previous=json.loads(statepath.read_text(encoding='utf-8'))
    eligibility=recovery_eligibility(previous)
    if a.audit_only or not eligibility['eligible']:
        print(json.dumps({'repo_id':a.repo_id,**eligibility,'audit_only':a.audit_only}));return
    if not os.environ.get('STY'):raise RuntimeError('E0 driver and its GPU subprocesses must run in screen')
    if any(previous.get(key)!=value for key,value in [('repo_id',a.repo_id),('revision',a.revision),
        ('batch_size',a.batch_size),('attn_implementation',a.attn_implementation)]):
        raise ValueError('Recovery must retain the prior model/revision/batch/attention identity')
    amendment_hash=sha(a.amendment_file)
    archive,old_hash=archive_previous_state(statepath)
    runbase=[sys.executable,str(HERE/'inference_v3.py'),'--model-dir',a.model_dir,
        '--repo-id',a.repo_id,'--revision',a.revision,'--batch-size',str(a.batch_size),
        '--model-kind',a.model_kind,'--attn-implementation',a.attn_implementation,
        '--max-length',str(a.max_length)]
    state={'schema':SCHEMA,'repo_id':a.repo_id,'revision':a.revision,
        'batch_size':a.batch_size,'attn_implementation':a.attn_implementation,
        'selection_rule':'fixed_original_t2_prefill_taskpilot_then_first_full100_operational_pass',
        'amendment_reason':'Calibration only freezes token IDs; unrelated20 calibration quality must not skip actual-task pilots. Formal task q/format/R2 limits unchanged.',
        'amendment_file':str(a.amendment_file.resolve()),'amendment_sha256':amendment_hash,
        'previous_state_snapshot':str(archive.resolve()),'previous_state_sha256':old_hash,
        'original_calibration_only_skips':eligibility['calibration_only_skips'],
        'supersedes_selection':{k:previous.get(k) for k in ('selected_remedy','selected_analysis','final_qualification')},
        'calibration_used_for_admission':False,
        'calibration_screen_is_proxy_not_formal_format_diagnostic':True,
        'effect_directions_not_used_to_choose_template':True,
        'execution_protocol_sha256':sha(HERE.parent/'execution_protocol.txt'),
        'driver_sha256':sha(__file__),'started_utc':datetime.now(timezone.utc).isoformat(),
        'attempts':[],'status':'running'}
    write_state(statepath,state)
    for remedy in MODES:
        folder=out/remedy;folder.mkdir(exist_ok=True)
        lockpath=folder/'token_calibration.json'
        if not lockpath.exists():
            call(runbase+['--remedy',remedy,'--mode','calibrate','--input',a.calibration_input,
                '--output',str(lockpath)],folder/'calibration.log')
        lock=json.loads(lockpath.read_text(encoding='utf-8'))
        if (lock['input_file_sha256']!=sha(a.calibration_input) or
            lock['lock_content_sha256']!=canonical_hash({k:v for k,v in lock.items() if k!='lock_content_sha256'}) or
            lock['engine']['code_sha256']!=sha(HERE/'inference_v3.py') or
            lock['engine']['native_loader_sha256']!=sha(HERE/'loader_v3.py') or
            lock['engine']['parent_loader']['repo_id']!=a.repo_id or
            lock['engine']['parent_loader']['revision']!=a.revision or
            lock['engine']['parent_loader']['attn_implementation']!=a.attn_implementation or
            lock['engine']['remedy']!=remedy or lock['batch_size']!=a.batch_size):
            raise ValueError('Existing calibration identity differs; do not reuse another run')
        cal_ok=lock['median_q']>=.70 and lock['greedy_any_variant_fraction']>=.90
        attempt={'remedy':remedy,'calibration_file':str(lockpath.resolve()),
            'calibration_q_median':lock['median_q'],'calibration_argmax_format_proxy':lock['greedy_any_variant_fraction'],
            'calibration_screen_pass':cal_ok,'calibration_gate_used':False}
        state['attempts'].append(attempt);write_state(statepath,state)
        for stage,input_path in [('pilot',a.pilot_input),('development',a.development_input)]:
            result=folder/(stage+'_R0_R1.jsonl')
            if not Path(str(result)+'.summary.json').exists():
                call(runbase+['--remedy',remedy,'--mode','score','--readouts','R0_R1',
                    '--input',input_path,'--output',str(result),'--calibration',str(lockpath)],folder/(stage+'.log'))
            apath=folder/(stage+'_E0_analysis.json')
            if not apath.exists():
                call([sys.executable,str(HERE/'analyze_e0_v3.py'),'--inputs',input_path,
                    '--r0',str(result),'--output',str(apath)],folder/(stage+'_analysis.log'))
            metrics=validate_existing_analysis(apath,input_path,result,lockpath)
            if metrics['qualification']['R0_operational']=='incomplete' or not metrics['qualification']['R1_candidate_scoring_valid']:
                raise ValueError('Missing operational/R1 measurements; do not classify as instruction failure')
            attempt[stage]={'analysis':str(apath.resolve()),'operational_gates':metrics['operational_gates'],
                'qualification':metrics['qualification']}
            write_state(statepath,state)
            if metrics['qualification']['R0_operational']!='pass' and remedy!=MODES[-1]:
                attempt['decision']='advance_due_to_'+stage+'_format_q_R2';write_state(statepath,state);break
            if a.stop_after==stage or stage=='development':
                state.update(status='complete' if stage=='development' else 'pilot_selection_complete',
                    selected_remedy=remedy,selected_calibration=str(lockpath.resolve()),
                    final_qualification=metrics['qualification']['final_qualification'],
                    qualification=metrics['qualification'],selected_analysis=str(apath.resolve()))
                attempt['decision']='retain_earliest_operational_variant' if metrics['qualification']['R0_operational']=='pass' else 'retain_R1_only_last_rescue'
                break
        if state['status']!='running':break
    state['finished_utc']=datetime.now(timezone.utc).isoformat();write_state(statepath,state)
    print(json.dumps({k:state.get(k) for k in ['status','repo_id','selected_remedy','final_qualification']},ensure_ascii=False))


if __name__=='__main__':main()
