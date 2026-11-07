"""Shared immutable dependency closure and verified-file cache for many runs."""
from pathlib import Path
from common import load, sha

KEYS = {'source_sha256', 'file_sha256', 'input_hashes', 'frozen_file_sha256', 'verified_source_sha256'}
OLD_PROTOCOL = 'b1e681999a4c8a7115cd077121879109ea91b2162c68a2ce6a898ac0e59b554a'


def historical(path, value, base):
    if path == str(Path(base) / 'execution_protocol.txt') and value == OLD_PROTOCOL:
        return str(Path(base) / 'provenance/execution_protocol_E0_original.txt')
    return path


def closure(initial, base, extension):
    roots = [Path(p).resolve() for p in (base, extension)]; bindings = {}; todo = []
    def add(path, value):
        path = historical(path, value, base); p = Path(path)
        if not isinstance(value, str) or len(value) != 64 or not p.is_absolute():
            raise ValueError('Non-SHA or nonabsolute evidence dependency')
        if not any(p.resolve().is_relative_to(root) for root in roots):
            raise ValueError('Dependency is outside the registered project roots')
        if path in bindings and bindings[path] != value:
            raise ValueError('Conflicting evidence versions need explicit preserved-source resolution')
        if path not in bindings: bindings[path] = value; todo.append(path)
    for path, value in initial.items(): add(path, value)
    while todo:
        path = todo.pop()
        if sha(path) != bindings[path]: raise ValueError('Changed source: ' + path)
        if Path(path).suffix != '.json': continue
        def visit(value):
            if isinstance(value, list):
                for item in value: visit(item)
            elif isinstance(value, dict):
                for key, item in value.items():
                    if key in KEYS and isinstance(item, dict):
                        for name, digest in item.items():
                            if isinstance(name, str) and Path(name).is_absolute(): add(name, digest)
                    else: visit(item)
        visit(load(path))
    return bindings


class VerifiedSources:
    """Hash each frozen byte version, then recheck inode/size/mtime/ctime per use."""
    def __init__(self, resolver=lambda p, value: Path(p)):
        self.resolver = resolver; self.cache = {}

    def __call__(self, original, expected):
        path = Path(self.resolver(original, expected)); stat = path.stat()
        stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        previous = self.cache.get(str(path))
        if previous is None or previous[:2] != (stamp, expected):
            if sha(path) != expected: raise ValueError('Frozen external source changed: ' + str(path))
            content = path.read_bytes() if path.suffix == '.json' else b''
            self.cache[str(path)] = (stamp, expected, content)
        return self.cache[str(path)][2]
