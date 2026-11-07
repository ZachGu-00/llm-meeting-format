"""Download the exact registered checkpoint and hash it against the old snapshot."""
from pathlib import Path
import hashlib,json,os,shutil
ROOT=Path('/workspace/voices_backup_recheck_20261106')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    from huggingface_hub import snapshot_download
    plan=json.loads((ROOT/'original_LFM_plan.json').read_text());model=plan['model']
    original=ROOT/'runtime/original_snapshot_manifest.json';manifest=json.loads(original.read_text());dest=ROOT/'model'
    os.environ.pop('HF_HUB_OFFLINE',None);os.environ.pop('TRANSFORMERS_OFFLINE',None)
    snapshot_download(repo_id=model['repo_id'],revision=model['revision'],local_dir=str(dest),allow_patterns=list(manifest['files']),max_workers=4)
    for name,meta in manifest['files'].items():
        p=dest/name
        if p.stat().st_size!=meta['bytes'] or sha(p)!=meta['sha256']:raise ValueError('Checkpoint mismatch '+name)
    shutil.copyfile(original,dest/'snapshot_manifest.json')
    (dest/'RESEARCH_REVISION.txt').write_text(model['revision']+'\n')
    (ROOT/'model_ready.json').write_text(json.dumps({'model':model['repo_id'],'revision':model['revision'],'registered_files_verified':len(manifest['files']),'bytes':sum(v['bytes'] for v in manifest['files'].values()),'snapshot_sha256':sha(original)},indent=2))
    print('MODEL_READY',flush=True)
if __name__=='__main__':main()

