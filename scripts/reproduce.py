"""Reproduce the current manuscript tables from frozen measurements."""
import argparse
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm_meeting_format.analysis import run
from llm_meeting_format.data import conditions
from llm_meeting_format.design import check_design
from llm_meeting_format.paper_tables import write
from llm_meeting_format.validation import check_caches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path, help="External frozen paper_data directory")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--manuscript", type=Path, help="Optional manuscript source directory for exact table checks")
    parser.add_argument("--check-caches", action="store_true", help="Reconstruct all 162 cache jobs and test the Jensen bound")
    args = parser.parse_args()
    root, output = args.data_root.resolve(), args.output.resolve()
    tables, verification = run(root, output)
    verification.update(check_design())
    verification["manuscript_tables_matched"] = write(tables, output, args.manuscript)
    if args.check_caches:
        verification.update(check_caches(root, conditions(root)))
    verification["status"] = "PASS"
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
