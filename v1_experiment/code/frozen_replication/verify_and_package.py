"""Perform one full-panel replay, export tables and make a final portable package."""
from datetime import datetime,timezone
from pathlib import Path
import csv,hashlib,json,subprocess,sys,zipfile
ROOT=Path('/workspace/voices_backup_recheck_20261106');RUN=ROOT/'LFM_rerun_3090'
sys.path[:0]=[str(ROOT/'runtime/ext'),str(ROOT/'runtime/base/code')]
from dialogue_bundle import verify_bundle
from dialogue_core import meeting_schedule
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    plan=json.loads((RUN/'plan.json').read_text())
    reports=[json.loads(p.read_text()) for p in sorted((RUN/'run_status').glob('*.json'))]
    if len(reports)!=288 or {r['run_id'] for r in reports}!={r['run_id'] for r in plan['runs']}:raise ValueError('Incomplete fixed panel')
    expected=sum(len(meeting_schedule(c))*3 for c in plan['runs'])
    proof_path=RUN/'full_panel_verification.json'
    previous=json.loads(proof_path.read_text()) if proof_path.is_file() else {}
    hashes={r['run_id']:r['bundle']['archive_sha256'] for r in reports}
    reusable=(previous.get('verified_runs')==288 and previous.get('source_hashes_tokens_stances_and_memory_replayed') is True
              and previous.get('expected_utterances')==expected and previous.get('run_archive_sha256')==hashes)
    if reusable:
        for r in reports:
            info=r['bundle'];path=Path(info['archive'])
            if path.stat().st_size!=info['archive_bytes'] or sha(path)!=info['archive_sha256']:
                raise ValueError('Previously verified bundle changed '+r['run_id'])
        print('Reused completed full-panel replay after rechecking all bundle hashes',flush=True)
    else:
        acks=[]
        for r in reports:
            info=r['bundle'];acks.append(verify_bundle(info['archive'],info))
        proof={'utc':datetime.now(timezone.utc).isoformat(),'verified_runs':len(acks),
               'source_hashes_tokens_stances_and_memory_replayed':True,'expected_utterances':expected,
               'run_archive_sha256':hashes}
        proof_path.write_text(json.dumps(proof,indent=2)+'\n')
    subprocess.run([sys.executable,str(ROOT/'code/finalize_rerun.py')],check=True)
    result=json.loads((RUN/'export/replication_report.json').read_text())
    if result['counts']!={'updates':expected,'utterances':expected,'readouts':expected*14}:raise ValueError('Full export count mismatch')
    files=[p for d in ('bundles','run_status','export') for p in (RUN/d).rglob('*') if p.is_file()]
    files+=[RUN/n for n in ['plan.json','calibration.json','platform_validation.json','pilot.json','full_panel_verification.json']]
    extra=[p for p in (ROOT/'code').glob('*.py')]
    extra+=[ROOT/n for n in ['rerun_metadata.zip','restored_runtime.csv','search_summary.json','exact_archive_matches.csv',
                             'LFM_run_fragments.csv','archive_links.csv','diagnostic_status_snapshots.csv','JOB_CONTROL.json']]
    extra+=[p for p in (ROOT/'复查与重跑说明.md',ROOT/'ERRORS.jsonl') if p.exists()]
    inventory=[]
    for p in files+extra:inventory.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)})
    manifest=ROOT/'LFM_FINAL_FILE_INDEX.json';manifest.write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'files':inventory},indent=2)+'\n')
    final=ROOT/'LFM_3090_replication_20261106.zip';temp=final.with_suffix('.zip.part')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4,allowZip64=True) as z:
        for p in files+extra+[manifest]:
            z.write(p,str(p.relative_to(ROOT)),compress_type=zipfile.ZIP_STORED if p.suffix in ('.gz','.zip') else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(temp) as z:
        if z.testzip() is not None:raise ValueError('Package CRC failed')
    temp.replace(final);value=sha(final);(ROOT/'LFM_3090_replication_20261106.zip.sha256').write_text(value+'  '+final.name+'\n')
    print(json.dumps({'status':'verified_and_packaged','runs':288,'counts':result['counts'],'bytes':final.stat().st_size,'sha256':value}),flush=True)
if __name__=='__main__':main()

