from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write_new(path, value):
    path = Path(path)
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                       allow_nan=False) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Different existing artifact preserved: ' + str(path))
        return
    with path.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

def atomic_status(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')
    temporary.replace(path)

def verify_bindings(bindings):
    for path, digest in bindings.items():
        if sha(path) != digest:
            raise ValueError('Frozen source changed: ' + str(path))
