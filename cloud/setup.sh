#!/usr/bin/env bash
set -euo pipefail
# Put VOICES_SSH_PRIVATE_KEY in the Codex environment's Secrets, never in Git.
command -v python3 >/dev/null
command -v ssh >/dev/null
python3 - <<'PY'
from pathlib import Path
import os
key = os.environ.get('VOICES_SSH_PRIVATE_KEY', '')
if not key.strip():
    raise SystemExit('Missing environment Secret VOICES_SSH_PRIVATE_KEY; setup did not configure SSH.')
target = Path.home() / '.ssh' / 'voices_v3_ed25519'
target.parent.mkdir(mode=0o700, exist_ok=True)
target.write_text(key.rstrip() + '\n', encoding='utf-8')
target.chmod(0o600)
print('SSH credential installed outside the repository. No experiment was resumed.')
PY

