"""Run the paper analysis against an external paper_data directory."""
import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SCRIPTS = {
    "verify": "validate_mainline_evidence.py",
    "tables": "build_writing_tables.py",
    "evidence": "build_mainline_evidence.py",
    "validate-data": "validate_dataset.py",
    "inventory": "build_file_inventory.py",
}


def verify_sources():
    manifest = json.loads((ROOT / "source_manifest.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        path = ROOT / record["file"]
        assert path.is_file(), record["file"]
        assert path.stat().st_size == record["bytes"], record["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], record["file"]
    scripts = list(ROOT.rglob("*.py"))
    for path in scripts:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    print(json.dumps({"status": "PASS", "source_files": len(manifest["files"]),
                      "python_files_parsed": len(scripts)}))


def load_script(filename, data_root):
    path = ROOT / "code" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ROOT = data_root
    if filename in {"build_mainline_evidence.py", "validate_mainline_evidence.py"}:
        module.OUT = data_root / "mainline"
        module.DATA = module.OUT / "figure_data"
    elif filename == "build_file_inventory.py":
        module.OUTPUT = data_root / "provenance" / "data_inventory.csv"
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["sources", *SCRIPTS, "group", "summarize"])
    parser.add_argument("--data-root", type=Path, help="Path to the complete paper_data directory")
    args = parser.parse_args()
    if args.command == "sources":
        verify_sources()
        return
    if args.data_root is None:
        parser.error("--data-root is required for this command")
    data_root = args.data_root.expanduser().resolve()
    if not (data_root / "coverage.csv").is_file():
        parser.error("--data-root must contain paper_data/coverage.csv")
    os.environ["VOICES_OUT"] = str(data_root)
    os.environ["VOICES_BACKUP"] = str(data_root / "historical")
    if args.command in SCRIPTS:
        load_script(SCRIPTS[args.command], data_root).main()
    else:
        sys.path.insert(0, str(ROOT / "code" / "frozen_analysis"))
        module = importlib.import_module("group_analysis" if args.command == "group" else "summarize")
        if args.command == "group":
            module.main()
        else:
            module.stats()


if __name__ == "__main__":
    main()
