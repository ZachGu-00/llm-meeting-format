"""Stage-major E1/E2 queue using completed E0 decisions. No downloads or tuning.

Default is E1 only, permitting the E4 model selection to be frozen before E2.
Use --stages E1 E2 for both, or --stages E2 to continue. Actual execution requires
screen; --dry-run writes commands/pending states without loading any model.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CORE_MODELS=(
 'LiquidAI/LFM2.5-1.2B-Instruct','Qwen/Qwen3.5-4B','Qwen/Qwen3.5-2B',
 'Qwen/Qwen3.5-0.8B','openbmb/MiniCPM5-2B','ibm-granite/granite-4.2-3b',
 'HuggingFaceTB/SmolLM3-3B','microsoft/Phi-4-mini-instruct','google/gemma-4-E2B-it',
 'mistralai/Ministral-3-3B-Instruct-2512-BF16')
PREVIOUS20_MODELS={'Qwen/Qwen3.5-4B','LiquidAI/LFM2.5-1.2B-Instruct'}
TIERS={'R0_primary','R1_primary_R0_operational','R1_only'}

def now():return datetime.now(timezone.utc).isoformat()
def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)
def event(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a',encoding='utf-8') as f:f.write(json.dumps({'utc':now(),**value},ensure_ascii=False,allow_nan=False)+'\n')

def verified_inputs(base):
 core=load(base/'data/core_data_manifest.json');e2=load(base/'data/e2_manifest.json');inputs={}
 for name,key in [('E1_R0','e1_holdout200_prompts.jsonl'),('E1_R1','e1_semantic64_prompts.jsonl')]:
  entry=core['outputs'][key];inputs[name]={'path':base/entry['path'],'n_rows':entry['n_rows'],'sha256':entry['sha256']}
 for name,entry in e2['outputs'].items():inputs[name]={'path':base/entry['path'],'n_rows':entry['n_rows'],'sha256':entry['sha256']}
 for spec in inputs.values():
  if sha(spec['path'])!=spec['sha256']:raise ValueError('Fixed input changed: '+str(spec['path']))
 parents={r['item_id'] for r in load(base/'data/e1_holdout200_items.json')}
 semantic={r['item_id'] for r in load(base/'data/e1_semantic64_items.json')}
 fresh={r['item_id'] for r in load(base/'data/e1_remaining180_items.json')}
 fresh_semantic={r['item_id'] for r in load(base/'data/e1_semantic_remaining44_items.json')}
 old=set(core['reused20_ids'])
 if not(len(parents)==200 and len(semantic)==64 and len(fresh)==180 and len(fresh_semantic)==44 and
        old<=semantic and parents-old==fresh and semantic-old==fresh_semantic and
        fresh==set(core['Qwen4_and_LFM_confirmatory_item_ids'])):
  raise ValueError('Prespecified200/64 versus fresh180/44 partitions do not agree')
 return inputs

def runtime(base):
 d=load(base/'setup/runtime_decision.json')
 if d.get('status')!='selected':raise ValueError('Runtime decision is not selected')
 for key,name in [('inference_sha256','inference_v3.py'),('loader_sha256','loader_v3.py')]:
  if d[key]!=sha(base/'code'/name):raise ValueError('Runtime code differs from frozen decision: '+name)
 if d['batch_size']<1:raise ValueError('Invalid frozen batch size')
 return d

def selected_model(base,item,decision):
 repo=item['repo_id'];slug=repo.replace('/','--');folder=base/'results'/slug/'e0'
 statepath=folder/'e0_driver_state.json'
 if not statepath.exists():return None,'E0_state_missing'
 state=load(statepath)
 if state.get('status')!='complete':return None,'E0_incomplete_not_model_disqualification'
 tier=state.get('final_qualification')
 if tier not in TIERS:return None,'E0_qualification_incomplete_not_model_disqualification'
 remedy=state['selected_remedy'];lockpath=folder/remedy/'token_calibration.json';analysispath=Path(state['selected_analysis']).resolve()
 if (Path(state['selected_calibration']).resolve()!=lockpath.resolve() or analysispath.parent!=(folder/remedy).resolve() or
     analysispath.name not in {'development_E0_analysis.json','development_E0_conditional_analysis.json'}):
  raise ValueError('Selected E0 files are not the named remedy development run')
 lock=load(lockpath);analysis=load(analysispath);q=analysis['qualification']
 if state['repo_id']!=repo or state['revision']!=item['revision'] or state['batch_size']!=decision['batch_size']:
  raise ValueError('E0 checkpoint or batch differs from queue/runtime')
 if q!=state['qualification'] or q['final_qualification']!=tier or not q['development100_complete'] or not q['R1_candidate_scoring_valid']:
  raise ValueError('Completed E0 qualification is inconsistent or lacks valid R1')
 protocol_archive=None
 for source,digest in analysis['file_sha256'].items():
  if sha(source)==digest:continue
  archive=base/'provenance/execution_protocol_E0_original.txt'
  if (Path(source).resolve()==(base/'execution_protocol.txt').resolve() and
      digest==state.get('execution_protocol_sha256') and archive.exists() and sha(archive)==digest):
   protocol_archive=str(archive)
  else:raise ValueError('Selected E0 analysis source changed: '+source)
 engine=lock['engine'];parent=engine['parent_loader']
 if (engine['code_sha256']!=decision['inference_sha256'] or engine['native_loader_sha256']!=decision['loader_sha256'] or
     engine['remedy']!=remedy or parent['repo_id']!=repo or parent['revision']!=item['revision'] or
     parent['attn_implementation']!=decision['attn_implementation'] or lock['batch_size']!=decision['batch_size']):
  raise ValueError('E0 calibration identity differs from frozen runtime/checkpoint')
 # Use the engine's own canonical convention; this matches inference_v3.canonical_hash.
 content={k:v for k,v in lock.items() if k!='lock_content_sha256'}
 actual=hashlib.sha256(json.dumps(content,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 if lock['lock_content_sha256']!=actual:raise ValueError('E0 calibration content changed')
 if sha(base/'data/calibration_prompts.jsonl')!=lock['input_file_sha256']:raise ValueError('Calibration source changed')
 model=base/'models'/slug
 if not(model/'snapshot_manifest.json').exists():return None,'checkpoint_unavailable_not_model_disqualification'
 return {'repo_id':repo,'revision':item['revision'],'slug':slug,'model_dir':str(model),'tier':tier,
  'R0_operational':q['R0_operational']=='pass','R1_candidate_scoring_valid':True,
  'E4_R0_eligible':q['R0_operational']=='pass','E4_readout_disagreement_flag':tier=='R1_primary_R0_operational',
  'E4_selection_not_performed_here':True,'core_E2_panel':repo in CORE_MODELS,
  'remedy':remedy,'model_kind':engine.get('model_kind','auto'),'max_length':parent.get('max_length',8192),
  'calibration':str(lockpath),'calibration_sha256':sha(lockpath),'option_ids':lock['option_ids'],
  'engine':engine,'E0_state_sha256':sha(statepath),'E0_analysis_sha256':sha(analysispath),'E0_analysis_path':str(analysispath),
  'E0_execution_protocol_sha256':state.get('execution_protocol_sha256'),
  'E1_execution_protocol_sha256':sha(base/'execution_protocol.txt'),
  'protocol_updated_since_E0':state.get('execution_protocol_sha256')!=sha(base/'execution_protocol.txt'),
  'E0_original_protocol_verified_archive':protocol_archive,
  'old20_excluded_from_confirmation':repo in PREVIOUS20_MODELS},None

def score_job(base,model,decision,spec,name,stage):
 readout='R1' if name=='E1_R1' else 'R0';folder=base/'results'/model['slug']/stage.lower()
 filename={'E1_R0':'holdout200_R0.jsonl','E1_R1':'semantic64_R1.jsonl'}.get(name,name.lower()+'_R0.jsonl')
 output=folder/filename
 command=[sys.executable,str(base/'code/inference_v3.py'),'--mode','score','--model-dir',model['model_dir'],
  '--repo-id',model['repo_id'],'--revision',model['revision'],'--input',str(spec['path']),'--output',str(output),
  '--readouts',readout,'--calibration',model['calibration'],'--remedy',model['remedy'],
  '--model-kind',model['model_kind'],'--batch-size',str(decision['batch_size']),
  '--attn-implementation',decision['attn_implementation'],'--max-length',str(model['max_length'])]
 return {'stage':stage,'name':name,'model':model['repo_id'],'input':str(spec['path']),'input_sha256':spec['sha256'],
  'rows':spec['n_rows'],'readout':readout,'output':str(output),'log':str(output)+'.log','command':command}

def score_closure(job,model,decision):
 output=Path(job['output']);sp=Path(str(output)+'.summary.json');rp=Path(str(output)+'.run.json')
 if not sp.exists():return None
 s=load(sp);r=load(rp)
 if not(s.get('status')=='complete' and s['n_input']==s['n_output']==job['rows'] and s['exact_id_and_count_equality']):
  raise ValueError('Existing score summary is not a complete planned run')
 if (s['output_sha256']!=sha(output) or s['output_bytes']!=output.stat().st_size or s['run_file_sha256']!=sha(rp) or
     r['input_file_sha256']!=job['input_sha256'] or r['n_input']!=job['rows'] or r['readout']!=job['readout'] or
     r['calibration_file_sha256']!=model['calibration_sha256'] or r['engine']!=model['engine'] or
     r['option_ids']!=model['option_ids'] or r['batch_size']!=decision['batch_size'] or any(s.get(k)!=v for k,v in r.items())):
  raise ValueError('Existing score closure/source/engine/calibration binding differs')
 return {'status':'complete','output_sha256':s['output_sha256'],'summary_sha256':sha(sp),'run_sha256':sha(rp),'n_output':s['n_output']}

def analysis_jobs(base,model,job):
 folder=Path(job['output']).parent;name=job['name'];items=[]
 def add(suffix,keep=None,filters=None,count=False,scope='descriptive'):
  output=folder/(suffix+'.json');script='analyze_v3_counts.py' if count else 'analyze_v3.py'
  cmd=[sys.executable,str(base/'code'/script),'--inputs',job['input'],'--results',job['output'],
       '--readout',job['readout'],'--draws','10000','--output',str(output)]
  if keep:cmd+=['--keep-items',str(base/'data'/keep)]
  if filters is not None:cmd+=['--filters',json.dumps(filters,sort_keys=True,separators=(',',':'))]
  items.append({'name':suffix,'output':str(output),'log':str(output)+'.log','command':cmd,'scope':scope})
 old=model['old20_excluded_from_confirmation']
 if name=='E1_R0':
  add('holdout200_R0_analysis',scope='descriptive_old20_included' if old else 'main200')
  if old:add('confirmatory180_R0_analysis','e1_remaining180_items.json',scope='confirmatory_fresh180')
  add('semantic64_R0_analysis','e1_semantic64_items.json',scope='descriptive_semantic64' if old else 'semantic64')
  if old:add('matched44_R0_analysis','e1_semantic_remaining44_items.json',scope='confirmatory_readout_matched44')
 elif name=='E1_R1':
  scope='R1_only_exploratory' if model['tier']=='R1_only' else 'descriptive_semantic64' if old else 'semantic64'
  add('semantic64_R1_analysis',scope=scope)
  if old:add('matched44_R1_analysis','e1_semantic_remaining44_items.json',scope='confirmatory_readout_matched44')
 elif name=='E2a':
  for n in range(4):add(f'e2a_neutral{n}_R0_analysis',filters={'neutral_count':n},scope='conditional_neutral_background')
 elif name in ('E2b','E2c'):add(name.lower()+'_R0_analysis',scope='named_role_prompt_ablation')
 else:add(name.lower()+'_R0_analysis',count=True,scope='count_grid_not_full_named_Mobius_kernel')
 return items

def build_plan(base,queue,decision,inputs,stages,models=None):
 wanted=set(models or []);known={x['repo_id'] for x in queue}
 if wanted-known:raise ValueError('Unknown --models: '+', '.join(sorted(wanted-known)))
 selected=[];pending=[]
 for item in queue:
  if wanted and item['repo_id'] not in wanted:continue
  try:model,reason=selected_model(base,item,decision)
  except Exception as ex:
   pending.append({'model':item['repo_id'],'status':'E0_identity_error_not_model_disqualification','error':str(ex)});continue
  if reason:pending.append({'model':item['repo_id'],'status':reason});continue
  selected.append(model)
 jobs=[]
 for stage in ('E1','E2'):
  if stage not in stages:continue
  for model in selected:
   names=(['E1_R0','E1_R1'] if model['R0_operational'] else ['E1_R1']) if stage=='E1' else (
      ['E2a','E2b','E2c','E2d','E2e'] if model['R0_operational'] and model['core_E2_panel'] else [])
   for name in names:
    job=score_job(base,model,decision,inputs[name],name,stage);job['analyses']=analysis_jobs(base,model,job);jobs.append(job)
 return selected,pending,jobs

def run_command(command,log):
 Path(log).parent.mkdir(parents=True,exist_ok=True)
 with Path(log).open('a',encoding='utf-8') as f:
  f.write('\nInvocation '+now()+'\n');f.flush()
  return subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=False).returncode

def analysis_closed(job):
 p=Path(job['output'])
 if not p.exists():return None
 d=load(p)
 if not d.get('file_sha256'):raise ValueError('Analysis lacks source bindings')
 for path,digest in d['file_sha256'].items():
  if sha(path)!=digest:raise ValueError('Existing analysis binding changed: '+path)
 return {'status':'complete','output_sha256':sha(p),'n_items':d['n_items'],'scope':job['scope']}

def existing_model_gpu_work(base):
 # Read only relevant process arguments; no process environments or credentials.
 if not Path('/proc').exists():return []
 me=os.getpid();matches=[]
 for p in Path('/proc').glob('[0-9]*'):
  if int(p.name)==me:continue
  try:args=(p/'cmdline').read_bytes().split(b'\x00')
  except (OSError,PermissionError):continue
  if any(arg.decode(errors='ignore') in {str(base/'code/inference_v3.py'),str(base/'code/run_e0_v3.py'),str(base/'setup/run_e0_queue.py')} for arg in args):matches.append(int(p.name))
 return matches

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
 p.add_argument('--stages',nargs='+',choices=['E1','E2'],default=['E1']);p.add_argument('--models',nargs='+')
 p.add_argument('--dry-run',action='store_true');p.add_argument('--analysis-mode',choices=['execute','defer'],default='execute')
 a=p.parse_args();base=a.root.resolve();decision=runtime(base);inputs=verified_inputs(base);queue=load(base/'setup/download_queue.json')
 if not queue:raise ValueError('Empty checkpoint panel cannot count as a completed experiment')
 selected,pending,jobs=build_plan(base,queue,decision,inputs,a.stages,a.models)
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');history=base/'setup/e1_e2_queue_history'/stamp
 protocol_sha=sha(base/'execution_protocol.txt');runtime_sha=sha(base/'setup/runtime_decision.json')
 plan={'schema':'voices_v3_E1_E2_queue_v1','created_utc':now(),'stages':a.stages,'dry_run':a.dry_run,'analysis_mode':a.analysis_mode,
  'order':'all selected E1 models before any E2 models','queue_sha256':sha(base/'setup/download_queue.json'),
  'execution_protocol_sha256':protocol_sha,'runtime_decision_sha256':runtime_sha,'driver_sha256':sha(__file__),
  'inputs':{k:{**v,'path':str(v['path'])} for k,v in inputs.items()},'selected_models':selected,'pending_models':pending,
  'jobs':jobs,'total_planned_score_rows':sum(j['rows'] for j in jobs),
  'initial_core_panel':list(CORE_MODELS),'prior20_rescored_not_pooled_from_v2':True,
  'confirmatory_selection_for_Qwen4_LFM':'fresh180 R0 and matched fresh44 R0/R1; all200/64 descriptive',
  'R1_only_models_not_E4_candidates':True,'E4_model_selection_performed_by_root_after_E1':True}
 write(history/'plan.json',plan)
 if a.dry_run:
  print(json.dumps({'status':'dry_run_only','selected':len(selected),'pending':pending,'jobs':len(jobs),'rows':plan['total_planned_score_rows'],'plan':str(history/'plan.json')},ensure_ascii=False));return
 if not os.environ.get('STY'):raise RuntimeError('E1/E2 queue and GPU subprocesses must run inside screen')
 competing=existing_model_gpu_work(base)
 if competing:raise RuntimeError('Another E0/inference process is active; do not overlap GPU queues: '+str(competing))
 lease=base/'setup/e1_e2_queue.lock'
 if lease.exists():
  owner=load(lease)
  try:os.kill(int(owner['pid']),0)
  except ProcessLookupError:lease.replace(history/'recovered_stale_lock.json')
  else:raise RuntimeError('Another E1/E2 queue lease is active: '+str(owner['pid']))
 with lease.open('x',encoding='utf-8') as f:json.dump({'pid':os.getpid(),'started_utc':now(),'history':str(history)},f)
 state={'started_utc':now(),'plan':str(history/'plan.json'),'pending_models':pending,'jobs':[],'status':'running'}
 statepath=base/'setup/e1_e2_queue_status.json';events=history/'events.jsonl';models={m['repo_id']:m for m in selected}
 failed=set()
 try:
  write(statepath,state)
  for job in jobs:
   model=models[job['model']];entry={'model':job['model'],'stage':job['stage'],'name':job['name'],'output':job['output'],'started_utc':now()}
   state['current_job']=entry.copy();write(statepath,state)
   if job['model'] in failed:
    entry.update(status='deferred_after_execution_error_not_disqualified');state['jobs'].append(entry);write(statepath,state);continue
   try:
    if sha(base/'setup/runtime_decision.json')!=runtime_sha or sha(base/'execution_protocol.txt')!=protocol_sha:raise ValueError('Frozen execution decisions changed while queue was running')
    if job['stage']=='E2':
     for dependency in ('E1_R0','E1_R1'):
      dep=score_job(base,model,decision,inputs[dependency],dependency,'E1')
      if score_closure(dep,model,decision) is None:raise ValueError('E2 waits for completed E1 R0 and R1 for this model')
    closure=score_closure(job,model,decision)
    if closure is None:
     raw=Path(job['output']);entry['resume_existing_bytes']=raw.stat().st_size if raw.exists() else 0
     event(events,{'event':'score_start','model':job['model'],'name':job['name'],'command':job['command'],'resume_existing_bytes':entry['resume_existing_bytes']})
     print('START '+job['model']+' '+job['name'],flush=True)
     rc=run_command(job['command'],job['log']);entry['exit_code']=rc
     if rc:raise RuntimeError('Scoring exited '+str(rc)+'; partial raw/log/run identity retained')
     closure=score_closure(job,model,decision)
     if closure is None:raise ValueError('Scoring returned without a closed summary')
    entry.update(closure);entry['analyses']=[]
    for ajob in job['analyses']:
     ast={'name':ajob['name'],'output':ajob['output'],'scope':ajob['scope'],'command':ajob['command']}
     try:
      complete=analysis_closed(ajob)
      if complete is None and a.analysis_mode=='execute':
       rc=run_command(ajob['command'],ajob['log'])
       if rc:raise RuntimeError('Analysis exited '+str(rc)+'; retain logs')
       complete=analysis_closed(ajob)
      ast.update(complete or {'status':'analysis_pending_command_saved'})
     except Exception as ex:ast.update(status='analysis_error_not_model_disqualification',error=str(ex))
     entry['analyses'].append(ast)
   except Exception as ex:
    entry.update(status='execution_error_not_model_disqualification',error=str(ex));failed.add(job['model'])
   entry['finished_utc']=now();state['jobs'].append(entry);event(events,{'event':'job_finished',**entry});write(statepath,state)
   print(json.dumps({k:entry[k] for k in ('model','stage','name','status')},ensure_ascii=False),flush=True)
  state.update(status='finished',finished_utc=now(),current_job=None,all_requested_scores_complete=not pending and all(j['status']=='complete' for j in state['jobs']),
     all_requested_analyses_complete=not pending and all(j['status']=='complete' and all(x['status']=='complete' for x in j.get('analyses',[])) for j in state['jobs']),
     no_model_disqualification_from_execution_errors=True)
  write(statepath,state);write(history/'final_status.json',state)
 finally:
  if lease.exists() and load(lease).get('pid')==os.getpid():lease.unlink()
 print(json.dumps({'status':state['status'],'all_scores_complete':state.get('all_requested_scores_complete'),'all_analyses_complete':state.get('all_requested_analyses_complete'),'failed_execution_models':sorted(failed),'state':str(statepath)},ensure_ascii=False))

if __name__=='__main__':main()
