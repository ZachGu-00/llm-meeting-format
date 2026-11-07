"""Single-model E0 sequence; batching is supplied by the separately frozen pilot."""
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


def validate_existing_analysis(path):
    d=json.loads(Path(path).read_text(encoding='utf-8'))
    for p,digest in d['file_sha256'].items():
        if sha(p)!=digest:raise ValueError('Existing E0 analysis source changed: '+p)
    return d


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('model-dir','repo-id','revision','calibration-input','pilot-input','development-input','output-dir'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--batch-size',type=int,required=True)
    parser.add_argument('--model-kind',choices=['auto','chat','base'],default='auto')
    parser.add_argument('--attn-implementation',choices=['eager','sdpa','flash_attention_2'],default='eager')
    parser.add_argument('--max-length',type=int,default=8192)
    parser.add_argument('--stop-after',choices=['calibration','pilot','development'],default='development')
    a=parser.parse_args()
    if not os.environ.get('STY'):raise RuntimeError('E0 driver and its GPU subprocesses must run in screen')
    if a.batch_size<1:raise ValueError('batch_size must be positive; this driver does not tune batching')
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    runbase=[sys.executable,str(HERE/'inference_v3.py'),'--model-dir',a.model_dir,
        '--repo-id',a.repo_id,'--revision',a.revision,'--batch-size',str(a.batch_size),
        '--model-kind',a.model_kind,'--attn-implementation',a.attn_implementation,
        '--max-length',str(a.max_length)]
    state={'schema':'voices_v3_E0_driver_v1','repo_id':a.repo_id,'revision':a.revision,
        'batch_size':a.batch_size,'attn_implementation':a.attn_implementation,
        'selection_rule':'earliest_workable_format_q_variant_in_fixed_original_t2_prefill_sequence',
        'calibration_screen_is_proxy_not_formal_format_diagnostic':True,
        'effect_directions_not_used_to_choose_template':True,
        'execution_protocol_sha256':sha(HERE.parent/'execution_protocol.txt'),
        'driver_sha256':sha(__file__),'started_utc':datetime.now(timezone.utc).isoformat(),
        'attempts':[],'status':'running'}
    statepath=out/'e0_driver_state.json'
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
            'calibration_screen_pass':cal_ok}
        state['attempts'].append(attempt);write_state(statepath,state)
        if not cal_ok and remedy!=MODES[-1]:
            attempt['decision']='advance_due_to_calibration_format_or_q';write_state(statepath,state);continue
        if a.stop_after=='calibration':
            state.update(status='calibration_selection_complete',selected_remedy=remedy,
                         final_qualification='not_yet_measured');break
        for stage,input_path in [('pilot',a.pilot_input),('development',a.development_input)]:
            result=folder/(stage+'_R0_R1.jsonl')
            if not Path(str(result)+'.summary.json').exists():
                call(runbase+['--remedy',remedy,'--mode','score','--readouts','R0_R1',
                    '--input',input_path,'--output',str(result),'--calibration',str(lockpath)],folder/(stage+'.log'))
            apath=folder/(stage+'_E0_analysis.json')
            if not apath.exists():
                call([sys.executable,str(HERE/'analyze_e0_v3.py'),'--inputs',input_path,
                    '--r0',str(result),'--output',str(apath)],folder/(stage+'_analysis.log'))
            metrics=validate_existing_analysis(apath)
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
