"""Small SSH client for the same GPU server; status is read-only."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['status', 'resume', 'fetch-reports'])
    parser.add_argument('--confirm-resume', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts')
    args = parser.parse_args()
    if args.action == 'resume' and not args.confirm_resume:
        parser.error('Resume requires explicit user authorization and --confirm-resume.')
    state = json.loads((ROOT / 'handoff/STATE.json').read_text(encoding='utf-8'))
    server = state['server']
    key = Path(os.environ.get('VOICES_SSH_KEY_FILE', str(Path.home() / '.ssh/voices_v3_ed25519')))
    if not key.is_file():
        raise SystemExit('SSH key absent. Configure the environment Secret and run bash cloud/setup.sh.')
    common = ['-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'ConnectTimeout=15',
              '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
              '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=' + str(ROOT / 'cloud/known_hosts'),
              '-i', str(key)]
    host = server['user'] + '@' + server['host']
    payload = {'action': args.action, 'state': state}
    program = 'PAYLOAD = ' + repr(payload) + '\n' + (ROOT / 'cloud/remote_ops.py').read_text(encoding='utf-8')
    result = subprocess.run(['ssh', '-T', *common, '-p', str(server['port']), host, server['python'], '-'],
                            input=program, text=True, encoding='utf-8', capture_output=True, timeout=600)
    if result.returncode:
        raise SystemExit('SSH/remote operation failed; no automatic queue restart.\n' + result.stderr[-4000:])
    response = json.loads(result.stdout)
    if args.action == 'fetch-reports':
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for item in response['files']:
            remote = PurePosixPath(item['path'])
            remote.relative_to(PurePosixPath(server['root']))
            target = args.output_dir / remote.name
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == item['sha256']:
                continue
            if target.exists():
                target = args.output_dir / (remote.stem + '_' + item['sha256'][:12] + remote.suffix)
            temporary = target.with_suffix(target.suffix + '.part')
            subprocess.run(['scp', *common, '-P', str(server['port']), host + ':' + str(remote), str(temporary)], check=True)
            if temporary.stat().st_size != item['bytes'] or hashlib.sha256(temporary.read_bytes()).hexdigest() != item['sha256']:
                raise SystemExit('Downloaded artifact failed size/SHA verification: ' + remote.name)
            temporary.replace(target)
        (args.output_dir / 'report_download_receipt.json').write_text(json.dumps(response, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

