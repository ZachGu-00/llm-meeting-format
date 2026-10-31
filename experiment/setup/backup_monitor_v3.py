"""Local-only SSH monitor and verified backup; never launches experiment jobs.

Remote writes are confined to setup/backup_archives/*.tar.gz. Existing results,
plans, code, model files and protocols are read-only. Local extraction refuses
changed existing files and all paths outside this experiment directory.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time

BASE = Path(__file__).resolve().parents[1]
REMOTE_BASE = '/workspace/voices_nonadditivity_v3'
E1_PLAN = REMOTE_BASE + '/setup/e1_e2_queue_history/20261031T053546212590Z/plan.json'
HOST = 'root@157.107.19.62'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15',
       '-o', 'ServerAliveCountMax=3', '-p', '16675', HOST]
REMOTE = r'''
import io,json,sys,tarfile,hashlib
from datetime import datetime,timezone
from pathlib import Path
base=Path('/workspace/voices_nonadditivity_v3')
sys.path.insert(0,str(base/'setup'))
import run_e1_e2_queue as q
import run_analysis_watcher_v3 as w
coordinator=q.load(base/'setup/continue_after_e1_status.json')
assert coordinator['e1_plan']==PAYLOAD['e1_plan']
def safe(path):
 p=Path(path).resolve();p.relative_to(base);return p
def summarize(path):
 a=q.load(path)
 if 'point' not in a:return {'n_items':a.get('n_items'),'status':'closed_counts_analysis'}
 return {'n_items':a['n_items'],'readout':a.get('readout'),'C':a['point'].get('C'),
  'C_interval95':a.get('bootstrap',{}).get('intervals',{}).get('C',{}).get('interval95'),
  'rules_R2':{k:v.get('R2_within_item') for k,v in a['point'].get('rules',{}).items()}}
def read_plan(path):
 path=safe(path);p=q.load(path)
 assert p['schema']=='voices_v3_E1_E2_queue_v1' and not p.get('dry_run')
 return path,p
def stage_ready(path,plan,repo):
 wp=path.parent/'analysis_watcher_status.json'
 if not wp.exists():return False
 ws=q.load(wp);assert ws['plan']==str(path) and ws['plan_sha256']==q.sha(path)
 done={(x['model'],x['name']):x for x in ws.get('jobs',[])}
 jobs=[x for x in plan['jobs'] if x['model']==repo]
 return bool(jobs) and all(done.get((repo,j['name']),{}).get('status')=='complete' and
  all(any(a['name']==t['name'] and a.get('status')=='complete' for a in done[(repo,j['name'])].get('analyses',[]))
      for t in j['analyses']) for j in jobs)
def closure(path,plan,repo):
 assert stage_ready(path,plan,repo),'Model has incomplete watcher jobs'
 model=next(x for x in plan['selected_models'] if x['repo_id']==repo)
 decision=q.runtime(base);files={str(path)};jobs=[];summaries={}
 assert q.sha(model['calibration'])==model['calibration_sha256']
 for job in plan['jobs']:
  if job['model']!=repo:continue
  assert q.sha(job['input'])==job['input_sha256']
  closed=q.score_closure(job,model,decision);assert closed is not None
  files.update([job['output'],job['output']+'.run.json',job['output']+'.summary.json'])
  if Path(job['log']).exists():files.add(job['log'])
  analyses=[]
  for task in job['analyses']:
   ac=w.analysis_closure(task);assert ac is not None
   files.add(task['output'])
   if Path(task['log']).exists():files.add(task['log'])
   summaries[task['name']]=summarize(task['output']);analyses.append({'name':task['name'],**ac})
  jobs.append({'name':job['name'],**closed,'analyses':analyses})
 return model,files,jobs,summaries
def archive(files,receipt,label):
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
 folder=base/'setup/backup_archives';folder.mkdir(exist_ok=True)
 out=folder/(stamp+'_'+label+'.tar.gz')
 paths=sorted({safe(x) for x in files})
 bindings={str(p.relative_to(base)):{'sha256':q.sha(p),'bytes':p.stat().st_size} for p in paths}
 stamps={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in paths}
 receipt.update(schema='voices_v3_closed_backup_receipt_v1',created_utc=stamp,files=bindings)
 member='setup/backup_archives/'+stamp+'_'+label+'_receipt.json'
 encoded=(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n').encode()
 with tarfile.open(out,'x:gz',compresslevel=1) as tf:
  for p in paths:tf.add(p,arcname=str(p.relative_to(base)),recursive=False)
  info=tarfile.TarInfo(member);info.size=len(encoded);tf.addfile(info,io.BytesIO(encoded))
 assert stamps=={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in paths},'Source changed during archive'
 return {'archive':str(out),'archive_sha256':q.sha(out),'archive_bytes':out.stat().st_size,
  'receipt_member':member,'receipt':receipt}
if PAYLOAD['mode']=='poll':
 result={'coordinator':{k:coordinator.get(k) for k in ('status','pid','updated_utc','E2_plan','E2_plan_sha256','error','selection_file','report_json')},'stages':[]}
 for stage,path in [('E1',PAYLOAD['e1_plan']),('E2',coordinator.get('E2_plan'))]:
  if not path:continue
  path,plan=read_plan(path)
  if stage=='E2':assert q.sha(path)==coordinator['E2_plan_sha256']
  ws=q.load(path.parent/'analysis_watcher_status.json') if (path.parent/'analysis_watcher_status.json').exists() else {}
  final=q.load(path.parent/'final_status.json') if (path.parent/'final_status.json').exists() else {}
  entry={'stage':stage,'plan':str(path),'plan_sha256':q.sha(path),'watcher_status':ws.get('status'),
         'final_status':final.get('status'),'ready_models':[],'errors':[]}
  for m in plan['selected_models']:
   if stage_ready(path,plan,m['repo_id']):entry['ready_models'].append(m['repo_id'])
  for x in ws.get('jobs',[]):
   if x.get('status') not in ('complete','running','pending','waiting_for_score') and ('error' in x or 'error' in x.get('status','')):
    entry['errors'].append({k:x.get(k) for k in ('model','name','status','error')})
  result['stages'].append(entry)
 qp=base/'setup/e1_e2_queue_status.json'
 if qp.exists():
  qs=q.load(qp);result['current_job']=qs.get('current_job');result['queue_status']=qs.get('status')
 print(json.dumps(result,ensure_ascii=False))
elif PAYLOAD['mode']=='model_archive':
 path,plan=read_plan(PAYLOAD['plan']);stage=PAYLOAD['stage'];repo=PAYLOAD['repo']
 assert plan['stages']==[stage]
 assert str(path)==(PAYLOAD['e1_plan'] if stage=='E1' else coordinator['E2_plan'])
 model,files,jobs,summaries=closure(path,plan,repo)
 print(json.dumps(archive(files,{'kind':'model_stage','stage':stage,'repo':repo,'model':model,'plan':str(path),
  'plan_sha256':q.sha(path),'jobs':jobs,'summaries':summaries},stage+'_'+model['slug']),ensure_ascii=False))
elif PAYLOAD['mode']=='report_archive':
 assert coordinator['status']=='complete_prerequisite_scope'
 rp=safe(coordinator['report_json']);report=q.load(rp)
 assert report['status']=='complete_verified_for_requested_scope'
 for source,digest in report['verified_source_sha256'].items():assert q.sha(safe(source))==digest,source
 files={str(rp),coordinator['report_markdown'],coordinator['selection_file']}
 for suffix in ('.json','.md'):files.add(coordinator['selection_prefix']+suffix)
 for job in coordinator['paired_jobs']:
  assert job['exit_code']==0
  files.add(job['output']);p=Path(job['output']).with_suffix('.md')
  if p.exists():files.add(str(p))
 for pp in (PAYLOAD['e1_plan'],coordinator['E2_plan']):
  path,plan=read_plan(pp)
  files.update(str(path.parent/x) for x in ('plan.json','final_status.json','analysis_watcher_status.json'))
 print(json.dumps(archive(files,{'kind':'prerequisite_report','coordinator':coordinator,
  'report_json':str(rp),'report_markdown':coordinator['report_markdown'],'counts':report['counts']},'prerequisite_report'),ensure_ascii=False))
else:raise ValueError('Unknown read-only backup operation')
'''


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def remote(payload):
    payload = {'e1_plan': E1_PLAN, **payload}
    code = 'PAYLOAD = ' + repr(payload) + '\n' + REMOTE
    result = subprocess.run(SSH + ['/venv/main/bin/python', '-'], input=code,
                            capture_output=True, text=True, encoding='utf-8', timeout=600)
    if result.returncode:
        raise RuntimeError('Remote backup/check failed: ' + result.stderr[-4000:])
    return json.loads(result.stdout)


def contained(path, base=None):
    base = BASE if base is None else Path(base)
    resolved = Path(path).resolve()
    resolved.relative_to(base.resolve())
    return resolved


def local_path(value):
    if not str(value).startswith(REMOTE_BASE + '/'):
        raise ValueError('Unexpected remote path outside experiment root')
    return contained(BASE / str(value)[len(REMOTE_BASE)+1:])


def extract_verified(archive, info):
    if sha(archive) != info['archive_sha256']:
        raise ValueError('Downloaded archive SHA mismatch')
    expected = info['receipt']['files']
    with tarfile.open(archive, 'r:gz') as tf:
        members = tf.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or set(names) != set(expected) | {info['receipt_member']}:
            raise ValueError('Unexpected/duplicate archive members')
        for member in members:
            posix = PurePosixPath(member.name)
            if not member.isfile() or posix.is_absolute() or '..' in posix.parts or '\\' in member.name:
                raise ValueError('Unsafe archive entry')
            destination = contained(BASE.joinpath(*posix.parts))
            if member.name not in expected:
                continue
            if destination.exists() and sha(destination) != expected[member.name]['sha256']:
                raise ValueError('Refusing to overwrite differing local artifact: ' + str(destination))
        for member in members:
            destination = contained(BASE.joinpath(*PurePosixPath(member.name).parts))
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(member) as source, destination.open('xb') as output:
                shutil.copyfileobj(source, output, length=1048576)
            if member.name in expected and sha(destination) != expected[member.name]['sha256']:
                raise ValueError('Extracted member SHA mismatch: ' + str(destination))
    receipt = load(BASE / info['receipt_member'])
    if receipt != info['receipt']:
        raise ValueError('Embedded archive receipt differs')
    return receipt


def download(info):
    target = contained(BASE / 'setup/backup_archives' / Path(info['archive']).name)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(target) != info['archive_sha256']:
            raise ValueError('Existing local archive differs; preserved for inspection')
    else:
        result = subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                                 '-P', '16675', HOST + ':' + info['archive'], str(target)],
                                capture_output=True, text=True, encoding='utf-8', timeout=600)
        if result.returncode:
            raise RuntimeError('Archive download failed: ' + result.stderr[-2000:])
    return target, extract_verified(target, info)


def validate_model(receipt):
    sys.path.insert(0, str(BASE / 'code'))
    import summarize_select_e1_v3 as e1
    import compare_e2_v3 as e2
    verifier = e1.Verifier(BASE, REMOTE_BASE)
    model = receipt['model']; repo = model['repo_id']
    calibration_path = local_path(model['calibration'])
    verifier.check(calibration_path, model['calibration_sha256'])
    calibration = {'value': load(calibration_path), 'sha256': model['calibration_sha256']}
    if receipt['stage'] == 'E1':
        for job in e1.expected_jobs(BASE, repo, model['R0_operational']):
            path = BASE / 'results' / model['slug'] / 'e1' / (job['name'] + '.json')
            e1.verify_e1(path, job, repo, calibration, verifier)
    else:
        plan = load(local_path(receipt['plan']))
        for job in plan['jobs']:
            if job['model'] != repo:
                continue
            for task in job['analyses']:
                _, identity = e2.closed_analysis(local_path(task['output']), verifier,
                                                 local_path(job['input']), job['rows'])
                if identity != (model['engine'], calibration['value']['batch_size'],
                                model['calibration_sha256'], model['option_ids']):
                    raise ValueError('Local E2 calibration engine differs')
    return {'status': 'verified_local_sources_and_sidecars', 'verified_file_count': len(verifier.verified)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poll-seconds', type=float, default=40)
    args = parser.parse_args()
    if not 30 <= args.poll_seconds <= 45:
        raise ValueError('Use 30–45 second polling')
    status_path = BASE / 'setup/backup_monitor_status.json'
    state = load(status_path) if status_path.exists() else {
        'schema': 'voices_v3_local_backup_monitor_v1', 'started_utc': now(),
        'e1_exact_plan': E1_PLAN, 'backups': {}, 'errors': [], 'archive_history': []}
    state.update(pid=os.getpid(), status='monitoring', helper_sha256=sha(__file__))
    def save(event=None):
        state['updated_utc'] = now()
        temp = status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temp.replace(status_path)
        if event:
            print(json.dumps({'utc': now(), **event}, ensure_ascii=False), flush=True)
    audit = BASE / 'provenance/audit_LFM_E1_independent_20261031.json'
    key = 'E1|LiquidAI/LFM2.5-1.2B-Instruct'
    if key not in state['backups']:
        record = load(audit)
        if record['status'] != 'pass':
            raise ValueError('Existing LFM local audit did not pass')
        for source, digest in record['source_sha256'].items():
            if sha(source) != digest:
                raise ValueError('Existing LFM audit source changed')
        state['backups'][key] = {'status': 'already_local_independently_verified',
                                 'audit': str(audit), 'audit_sha256': sha(audit)}
    save({'event': 'monitor_started', 'pid': os.getpid()})
    while True:
        try:
            poll = remote({'mode': 'poll'})
            prior = state.get('remote_status')
            state['remote_status'] = poll
            save({'event': 'remote_progress', **poll} if poll != prior else None)
            for stage in poll['stages']:
                for repo in stage['ready_models']:
                    key = stage['stage'] + '|' + repo
                    if key in state['backups']:
                        continue
                    pending = state.setdefault('pending_archives', {})
                    if key not in pending:
                        pending[key] = remote({'mode': 'model_archive', 'stage': stage['stage'], 'repo': repo,
                                               'plan': stage['plan']})
                        state['archive_history'].append({k: pending[key][k] for k in ('archive', 'archive_sha256', 'archive_bytes')})
                    info = pending[key]
                    save({'event': 'archive_created', 'model': repo, 'stage': stage['stage'],
                          'archive_bytes': info['archive_bytes']})
                    target, receipt = download(info)
                    validation = validate_model(receipt)
                    state['backups'][key] = {'status': 'backed_up_verified', 'completed_utc': now(),
                        'archive': str(target), 'archive_sha256': info['archive_sha256'],
                        'receipt': str(BASE / info['receipt_member']), 'validation': validation,
                        'summaries': receipt['summaries']}
                    del pending[key]
                    save({'event': 'model_backup_complete', 'model': repo, 'stage': stage['stage'],
                          'summaries': receipt['summaries']})
            if poll['coordinator']['status'] == 'complete_prerequisite_scope':
                if len(state['backups']) != 19:
                    raise ValueError('Final report ready before all ten E1/nine E2 backups verified')
                info = remote({'mode': 'report_archive'})
                state['archive_history'].append({k: info[k] for k in ('archive', 'archive_sha256', 'archive_bytes')})
                target, receipt = download(info)
                report = load(local_path(receipt['report_json']))
                for source, digest in report['verified_source_sha256'].items():
                    if sha(local_path(source)) != digest:
                        raise ValueError('Local prerequisite report source SHA differs: ' + source)
                copies = {}
                for extension, source in (('.md', receipt['report_markdown']), ('.json', receipt['report_json'])):
                    destination = BASE.parents[1] / ('voices_v3_E0_E3_阶段报告' + extension)
                    original = local_path(source)
                    if destination.exists() and sha(destination) != sha(original):
                        raise ValueError('User-facing report path exists with different content')
                    if not destination.exists():
                        shutil.copyfile(original, destination)
                    copies[str(destination)] = sha(destination)
                state.update(status='complete_prerequisite_backup', report_copies=copies,
                             report_archive=str(target), report_counts=report['counts'], completed_utc=now())
                save({'event': 'prerequisite_report_ready', 'copies': copies, 'counts': report['counts']})
                return
            if poll['coordinator']['status'] == 'terminal_error_not_model_rejection':
                state['status'] = 'waiting_for_root_repair'
                save({'event': 'coordinator_terminal_error', 'detail': poll['coordinator']})
        except Exception as error:
            entry = {'utc': now(), 'type': type(error).__name__, 'error': str(error)}
            state['errors'].append(entry)
            state['status'] = 'monitoring_with_backup_error'
            save({'event': 'backup_error', **entry})
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
