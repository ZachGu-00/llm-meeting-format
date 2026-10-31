"""Local read-only SSH monitor for closed E9/E4 outputs, with verified backups.

Remote writes are only new tar archives. Routing follows the exact bridge and
phase receipts, never a newest-directory search. No weights or credentials.
"""
import argparse
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import subprocess
import time

import backup_monitor_v3 as legacy

BASE=legacy.BASE
REMOTE_BASE=legacy.REMOTE_BASE

REMOTE=r'''
import io,json,re,sys,tarfile
from pathlib import Path
from datetime import datetime,timezone
base=Path(PAYLOAD['base']).resolve()
sys.path.insert(0,str(base/'setup'));sys.path.insert(0,str(base/'code'))
import run_e1_e2_queue as q
import run_e4_analysis_watcher_v3 as w
import run_e4_dynamics_queue_v3 as d
import run_after_prerequisite_v3 as after
import write_e4_report_v3 as reporter

def safe(value):
 p=Path(value)
 if p.is_symlink():raise ValueError('Symlink sources are not archived')
 p=p.resolve();parts=p.relative_to(base).parts
 if not parts or parts[0] not in ('results','reports','data','code','setup','provenance') and parts!=('execution_protocol.txt',):
  raise ValueError('Outside allowed evidence routes')
 if any(x.casefold() in ('models','.ssh','.hf_home','credentials') for x in parts):raise ValueError('Model/credential routes forbidden')
 if p.name.casefold() in ('api.txt','hug.txt','token','id_rsa','id_ed25519') or p.suffix.casefold() in ('.pem','.key'):
  raise ValueError('Credential file forbidden')
 if p.name.endswith('_status.json') and p.name!='final_status.json':raise ValueError('Mutable status files are not archived')
 return p

def exact_history(value,name):
 p=Path(value).resolve()
 if p.parent!=base/'setup'/name:raise ValueError('Unexpected explicit history route')
 return p

def context():
 bp=base/'setup/after_prerequisite_status.json'
 if not bp.exists():return {'bridge':{'status':'not_started'},'phase':None}
 b=q.load(bp);ctx={'bridge':b,'phase':None}
 if not b.get('E4_history'):return ctx
 if b.get('E9_complete') is not True:raise ValueError('E4 history supplied before closed E9 stage')
 history=exact_history(b['E4_history'],'e4_phase_history');p=q.load(history/'status.json')
 if (p.get('pid')!=b.get('E4_pid') or p.get('history')!=str(history) or
     p.get('selection_sha256')!=b.get('selection_sha256') or p.get('prerequisite_report_sha256')!=b.get('prerequisite_report_sha256')):
  raise ValueError('Bridge and exact E4 phase identities differ')
 ctx.update(phase=p,phase_history=history)
 if p.get('score_plan'):
  path=safe(p['score_plan']);exact_history(path.parent,'e4_score_queue_history');plan=q.load(path)
  if path.name!='plan.json' or q.sha(path)!=p['score_plan_sha256'] or plan.get('schema')!='voices_v3_E4_score_queue_v1' or plan.get('dry_run') is not False:
   raise ValueError('E4 scoring plan differs from exact phase receipt')
  ctx.update(score_path=path,score=plan)
  wp=path.parent/'analysis_watcher_status.json'
  if wp.exists():
   ws=q.load(wp)
   if ws.get('plan')!=str(path) or ws.get('plan_sha256')!=q.sha(path):raise ValueError('Watcher belongs to another score plan')
   ctx['watcher']=ws
 if p.get('dynamics_plan'):
  path=safe(p['dynamics_plan']);exact_history(path.parent,'e4_dynamics_queue_history');plan=q.load(path)
  if q.sha(path)!=p['dynamics_plan_sha256'] or plan.get('schema')!='voices_v3_E4_dynamics_queue_v1':raise ValueError('Dynamic plan identity differs')
  if plan['score_plan']!=str(ctx.get('score_path')) or plan['score_plan_sha256']!=q.sha(ctx['score_path']):raise ValueError('Dynamic/score plan binding differs')
  ctx.update(dynamic_path=path,dynamic=plan)
  candidates=[path.parent/'final_status.json',base/'setup/e4_dynamics_queue_status.json']
  for sp in candidates:
   if sp.exists():
    ds=q.load(sp)
    if ds.get('plan')==str(path) and ds.get('plan_sha256')==q.sha(path):ctx['dynamic_status']=ds;break
 return ctx

def catalog(ctx):
 b=ctx['bridge'];p=ctx.get('phase');items=[];required=[]
 if b.get('E9_complete'):
  path=safe(b['E9_plan']);key='E9|'+q.sha(path);required.append(key)
  if q.sha(path)!=b['E9_plan_sha256']:raise ValueError('Bridge E9 plan changed')
  items.append({'kind':'E9_all10','key':key})
 if 'score' not in ctx:return items,required
 score=ctx['score'];digest=q.sha(ctx['score_path']);ws=ctx.get('watcher',{})
 done={(j['model'],j['topic_id']):j for j in ws.get('jobs',[]) if j.get('status')=='complete'}
 for job in score['jobs']:
  key='E4topic|'+digest+'|'+job['model']+'|'+job['topic_id'];required.append(key)
  if (job['model'],job['topic_id']) in done:items.append({'kind':'E4_topic','key':key,'repo':job['model'],'topic_id':job['topic_id']})
 complete_models={r['model'] for r in ws.get('model_summaries',[]) if r.get('status')=='complete'}
 for repo in score['selected_repos']:
  key='E4summaries|'+digest+'|'+repo;required.append(key)
  if repo in complete_models:items.append({'kind':'E4_model_summaries','key':key,'repo':repo})
 if 'dynamic' in ctx:
  plan=ctx['dynamic'];digest=q.sha(ctx['dynamic_path'])
  completed={(j['model'],j['topic_id']) for j in ctx.get('dynamic_status',{}).get('jobs',[]) if j.get('status')=='complete'}
  aggregated={r['model']:r for r in p.get('aggregation_results',[]) if r.get('status')=='complete'}
  for repo in plan['selected_repos']:
   jobs=[j for j in plan['jobs'] if j['model']==repo];key='E4dynamics|'+digest+'|'+repo;required.append(key)
   if len(jobs)==6 and all((repo,j['topic_id']) in completed for j in jobs):items.append({'kind':'E4_model_dynamics','key':key,'repo':repo})
   key='E4aggregate|'+digest+'|'+repo;required.append(key)
   if repo in aggregated:items.append({'kind':'E4_dynamic_aggregate','key':key,'repo':repo})
 if p.get('status')=='complete_E4_scope' and p.get('E4_complete'):
  items.append({'kind':'E4_final_report','key':'E4report|'+p['report_sha256'],'requires_backup_keys':list(required)})
 return items,required

def collect(ctx,item):
 files=set();dependencies={};bound={};detail={'kind':item['kind'],'key':item['key']}
 def add(path,digest=None):
  p=safe(path)
  if not p.is_file():raise ValueError('Evidence source is not a regular file')
  digest=q.sha(p) if digest is None else digest
  if q.sha(p)!=digest:raise ValueError('Bound evidence SHA changed: '+str(p))
  bound[str(p)]=digest
  # Already delivered E1 archives contain these large originals; do not duplicate them.
  parts=p.relative_to(base).parts
  if len(parts)>2 and parts[0]=='results' and parts[2]=='e1':dependencies[str(p)]=digest
  else:files.add(str(p))
 def bindings(value):
  if not value:raise ValueError('Missing bound source hashes')
  for path,digest in value.items():add(path,digest)
 def sidecar(path,value,key):
  artifact=value[key];p=Path(path).parent/artifact['filename']
  if p.resolve().parent!=Path(path).parent.resolve():raise ValueError('Sidecar escapes its result directory')
  add(p,artifact['sha256'])
 kind=item['kind'];b=ctx['bridge']
 if kind=='E9_all10':
  pp=safe(b['E9_plan']);plan,tasks=after.closed_e9_plan(pp,b['E9_plan_sha256'],base,Path(q.load(pp)['e1_plan']),Path(b['prerequisite_report']))
  rp=safe(b['E9_report_json']);report=after.e9_report_closed(rp,b['E9_analysis_results'],base)
  if q.sha(rp)!=b['E9_report_sha256']:raise ValueError('Bridge E9 report SHA differs')
  add(pp);add(pp.parent/'deferred_analysis_commands.json');add(rp);add(rp.with_suffix('.md'));bindings(report['verified_source_sha256'])
  for job in plan['jobs']:
   for suffix in ('','.run.json','.summary.json'):add(job['output']+suffix)
  for task in tasks:
   path=Path(task['output_prefix']+'.json');value=q.load(path);add(path);add(path.with_suffix('.md'));sidecar(path,value,'bootstrap_artifact')
  for model in plan['selected_models']:add(model['calibration'],model['calibration_sha256'])
  detail.update(report_json=str(rp),report_markdown=str(rp.with_suffix('.md')),n_models=10,
                e9_plan=str(pp),prior_E1_originals_are_dependencies_not_duplicated=True)
 elif kind in ('E4_topic','E4_model_summaries'):
  plan=ctx['score'];models={m['repo_id']:m for m in plan['selected_models']};model=models[item['repo']]
  jobs=[j for j in plan['jobs'] if j['model']==item['repo']];add(ctx['score_path']);add(model['calibration'],model['calibration_sha256'])
  if kind=='E4_topic':
   job=next(j for j in jobs if j['topic_id']==item['topic_id']);closed=w.topic_closure(job,model,plan['runtime_decision'])
   if closed is None:raise ValueError('Topic point/raw is not closed')
   point=Path(closed['analysis_path']);value=q.load(point)
   for suffix in ('','.run.json','.summary.json'):add(job['output']+suffix)
   add(point,closed['analysis_sha256']);add(closed['cache_path'],closed['cache_sha256']);bindings(value['input_hashes'])
   detail.update(repo=item['repo'],topic_id=item['topic_id'],closure=closed,point_analysis=str(point))
  else:
   tasks=w.model_tasks(base,ctx['score_path'].parent,model,jobs)
   for task in tasks:
    closed=w.model_closure(task,model)
    if closed is None:raise ValueError('Six-topic CPU model summary not closed')
    path=Path(task['output']);value=q.load(path);add(path);bindings(value['file_sha256'])
    sidecar(path,value,'bootstrap_artifact' if task['kind']=='aggregate' else 'outcomes_and_bootstrap_artifact')
    if task['kind']=='supplement':add(path.with_suffix('.md'))
   detail.update(repo=item['repo'],tasks=tasks)
 elif kind in ('E4_model_dynamics','E4_dynamic_aggregate'):
  plan=ctx['dynamic'];jobs=[j for j in plan['jobs'] if j['model']==item['repo']];add(ctx['dynamic_path'])
  if len(jobs)!=6:raise ValueError('Dynamic model lacks all six topics')
  if kind=='E4_model_dynamics':
   receipts=[]
   for job in jobs:
    closed=d.closed(job)
    if closed is None or closed['n_closed_parts']!=190:raise ValueError('Dynamics lacks complete 190-part closure')
    add(job['output'],closed['output_sha256']);add(job['topic_analysis']);add(job['kernel_analysis'])
    receipts.append({'topic_id':job['topic_id'],**closed})
   detail.update(repo=item['repo'],topic_closures=receipts,part_files_archived=False,
                 scope='Complete combined topic outputs saved; intermediate resumable part files are not separately archived.')
  else:
   task=next(t for t in d.aggregation_commands(base,plan,ctx['dynamic_path'].parent) if t['model']==item['repo'])
   path=Path(task['command'][task['command'].index('--output')+1]);value=q.load(path)
   state=next(r for r in ctx['phase']['aggregation_results'] if r['model']==item['repo'])
   if state['output']!=str(path) or q.sha(path)!=state['output_sha256']:raise ValueError('Dynamic aggregate differs from phase receipt')
   if value['model_identity']['repo']!=item['repo'] or value['n_topics']!=6 or value['bootstrap']['draws']!=10000 or not reporter.dynamics_summary(value)['planned_complete']:
    raise ValueError('Dynamic aggregate is partial')
   add(path);bindings(value['file_sha256']);sidecar(path,value,'bootstrap_artifact');detail.update(repo=item['repo'],aggregate=str(path))
 elif kind=='E4_final_report':
  phase=after.e4_child_closed(ctx['phase_history'],b['E4_pid'],Path(b['prerequisite_report']),Path(b['selection']))
  path=safe(phase['report_json']);value=q.load(path);add(path,phase['report_sha256']);add(phase['report_markdown'])
  bindings(value['verified_source_sha256']);detail.update(report_json=str(path),report_markdown=phase['report_markdown'],counts=value['counts'],phase_history=str(ctx['phase_history']))
 else:raise ValueError('Unknown backup kind')
 detail.update(source_sha256=bound,prior_local_dependencies_sha256=dependencies)
 return files,detail

def archive(files,detail):
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
 label=re.sub('[^A-Za-z0-9_.-]','_',detail['kind']+'_'+detail.get('repo','all')+'_'+detail.get('topic_id',''))
 folder=base/'setup/backup_archives';folder.mkdir(exist_ok=True)
 out=folder/(stamp+'_'+label+'.tar.gz');paths=sorted({safe(p) for p in files})
 bindings={p.relative_to(base).as_posix():{'sha256':q.sha(p),'bytes':p.stat().st_size} for p in paths}
 stamps={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in paths}
 receipt={**detail,'schema':'voices_v3_closed_E9_E4_backup_receipt_v1','created_utc':stamp,'files':bindings}
 member='setup/backup_archives/'+stamp+'_'+label+'_receipt.json';encoded=(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n').encode()
 with tarfile.open(out,'x:gz',compresslevel=1) as tf:
  for p in paths:tf.add(p,arcname=p.relative_to(base).as_posix(),recursive=False)
  info=tarfile.TarInfo(member);info.size=len(encoded);tf.addfile(info,io.BytesIO(encoded))
 if stamps!={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in paths}:raise ValueError('Evidence changed during archive creation')
 return {'archive':str(out),'archive_sha256':q.sha(out),'archive_bytes':out.stat().st_size,'receipt_member':member,'receipt':receipt}

def dispatch(payload):
 ctx=context();items,required=catalog(ctx)
 if payload['mode']=='poll':
  return {'bridge_status':ctx['bridge'].get('status'),'phase_status':(ctx.get('phase') or {}).get('status'),
          'exact_phase_history':str(ctx.get('phase_history')) if ctx.get('phase_history') else None,
          'ready':items,'required_current_scope_keys':required}
 if payload['mode']=='archive':
  matches=[item for item in items if item['key']==payload['key']]
  if len(matches)!=1:raise ValueError('Requested scope is not currently closed under the exact bridge/phase')
  files,detail=collect(ctx,matches[0]);return archive(files,detail)
 raise ValueError('Only read-only poll or closed-evidence archive operations are supported')
'''


def remote(payload):
    payload={'base':REMOTE_BASE,**payload}
    code='PAYLOAD = '+repr(payload)+'\n'+REMOTE+'\nprint(json.dumps(dispatch(PAYLOAD),ensure_ascii=False))\n'
    result=subprocess.run(legacy.SSH+['/venv/main/bin/python','-'],input=code,capture_output=True,text=True,encoding='utf-8',timeout=600)
    if result.returncode:raise RuntimeError('E9/E4 backup check failed: '+result.stderr[-4000:])
    return json.loads(result.stdout)


def allowed_archive_info(info):
    """Additional allowlist before the existing immutable safe extractor runs."""
    receipt=info['receipt']
    if receipt.get('schema')!='voices_v3_closed_E9_E4_backup_receipt_v1':raise ValueError('Unexpected E9/E4 backup receipt')
    archive_path=PurePosixPath(info['archive']);receipt_path=PurePosixPath(info['receipt_member'])
    if (archive_path.parent!=PurePosixPath(REMOTE_BASE)/'setup/backup_archives' or archive_path.suffix!='.gz' or
        '\\' in info['archive'] or receipt_path.parent!=PurePosixPath('setup/backup_archives') or receipt_path.suffix!='.json'):
        raise ValueError('Unexpected archive transport route')
    for relative in receipt['files']:
        parts=relative.replace('\\','/').split('/')
        if (relative.startswith('/') or '..' in parts or '\\' in relative or
            parts[0] not in ('results','reports','data','code','setup','provenance') and relative!='execution_protocol.txt' or
            any(p.casefold() in ('models','.ssh','.hf_home','credentials') for p in parts) or
            parts[-1].casefold() in ('api.txt','hug.txt','token','id_rsa','id_ed25519') or Path(parts[-1]).suffix.casefold() in ('.key','.pem')):
            raise ValueError('Unsafe/model/credential backup member')


def validate_local(receipt):
    if receipt.get('schema')!='voices_v3_closed_E9_E4_backup_receipt_v1':raise ValueError('Wrong receipt schema')
    for source,digest in receipt['source_sha256'].items():
        if legacy.sha(legacy.local_path(source))!=digest:raise ValueError('Local bound source differs: '+source)
    if receipt['kind'] in ('E9_all10','E4_final_report'):
        report=legacy.load(legacy.local_path(receipt['report_json']))
        for source,digest in report['verified_source_sha256'].items():
            if legacy.sha(legacy.local_path(source))!=digest:raise ValueError('Local report source binding differs: '+source)
        if receipt['kind']=='E9_all10':
            if report.get('status')!='complete_requested_diagnostic_panel' or report.get('n_verified_models')!=10:raise ValueError('Local E9 report incomplete')
        elif (report.get('status')!='complete_E4_main_and_supplement' or any(report['counts'].get(k)!=6 for k in
            ('single_round_complete_models','supplement_complete_models','dynamics_complete_models'))):raise ValueError('Local E4 report incomplete')
    return {'status':'verified_local_source_hashes','n_verified_sources':len(receipt['source_sha256']),
        'prior_E1_dependencies_checked':len(receipt.get('prior_local_dependencies_sha256',{}))}


def copy_final_report(receipt):
    if receipt['kind']!='E4_final_report':raise ValueError('Only the verified final E4 report is copied to the project root')
    copies={}
    for suffix,key in [('.json','report_json'),('.md','report_markdown')]:
        source=legacy.local_path(receipt[key]);destination=BASE.parents[1]/('voices_v3_E4_阶段报告'+suffix)
        if destination.exists() and legacy.sha(destination)!=legacy.sha(source):raise ValueError('Different project-root report preserved; inspect before replacing')
        if not destination.exists():shutil.copyfile(source,destination)
        copies[str(destination)]=legacy.sha(destination)
    return copies


def backup_one(state,item,save):
    key=item['key']
    if key not in state['pending_archives']:
        info=remote({'mode':'archive','key':key});allowed_archive_info(info)
        state['pending_archives'][key]=info
        state['archive_history'].append({k:info[k] for k in ('archive','archive_sha256','archive_bytes')});save()
    info=state['pending_archives'][key];allowed_archive_info(info)
    target,receipt=legacy.download(info);validation=validate_local(receipt)
    if receipt.get('key')!=key or receipt.get('kind')!=item['kind']:raise ValueError('Archive receipt differs from requested closed scope')
    entry={'status':'backed_up_verified','completed_utc':legacy.now(),'archive':str(target),
        'archive_sha256':info['archive_sha256'],'receipt':str(BASE/info['receipt_member']),'validation':validation,'kind':receipt['kind']}
    if receipt['kind']=='E4_final_report':
        entry['project_report_copies']=copy_final_report(receipt)
        state.update(status='complete_E9_E4_backup',report_copies=entry['project_report_copies'],completed_utc=legacy.now())
    state['backups'][key]=entry;del state['pending_archives'][key];save()
    print(json.dumps({'event':'closed_backup_complete','key':key,'bytes':info['archive_bytes']},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--poll-seconds',type=float,default=40)
    parser.add_argument('--once',action='store_true');parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if not 30<=args.poll_seconds<=45:raise ValueError('Use 30–45 second polling')
    if args.dry_run:
        print(json.dumps({'status':'readonly_remote_poll_only','poll':remote({'mode':'poll'}),'no_archives_downloads_or_status_writes':True},ensure_ascii=False));return
    status_path=BASE/'setup/backup_e4_monitor_status.json'
    state=legacy.load(status_path) if status_path.exists() else {'schema':'voices_v3_local_E9_E4_backup_monitor_v1',
        'started_utc':legacy.now(),'backups':{},'pending_archives':{},'archive_history':[],'errors':[]}
    state.update(pid=os.getpid(),status='monitoring',helper_sha256=legacy.sha(__file__))
    def save():
        state['updated_utc']=legacy.now();temp=status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(status_path)
    def record_error(error,key=None):
        state['errors'].append({'utc':legacy.now(),'type':type(error).__name__,'error':str(error),'key':key})
        state['status']='monitoring_with_backup_error';save()
        print(json.dumps({'event':'backup_error','error':str(error),'key':key},ensure_ascii=False),flush=True)
    save()
    while True:
        try:
            poll=remote({'mode':'poll'});state['remote_status']=poll;save()
            for item in poll['ready']:
                key=item['key']
                if key in state['backups']:continue
                if any(required not in state['backups'] for required in item.get('requires_backup_keys',[])):continue
                try:backup_one(state,item,save)
                except Exception as error:record_error(error,key)
            if state['status']=='complete_E9_E4_backup':return
            if poll['bridge_status']=='terminal_error_not_model_rejection' or poll['phase_status']=='terminal_error_not_model_rejection':
                state['status']='waiting_for_root_repair';save()
        except Exception as error:
            record_error(error)
        if args.once:
            if state['status']=='monitoring_with_backup_error':raise SystemExit(2)
            return
        time.sleep(args.poll_seconds)


if __name__=='__main__':main()
