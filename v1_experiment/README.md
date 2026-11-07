# v1_experiment

Code for **Same Contacts, Different Rooms: Meeting Format Shapes LLM Group Influence Beyond Pairwise Contact Counts**.

## Files

- `code/`: analysis, table builders, and data validators.
- `code/frozen_analysis/`: cached-measurement analysis and transferred weight fitting.
- `code/frozen_replication/`: the original LFM replication scripts.
- `frozen/runtime/`: archived measurement and dialogue dependencies, in their original layout.
- `reference/`: small result summaries and the data manifest.
- `source_manifest.json`: source locations and SHA-256 checksums.

The copied scripts are unchanged. `run.py` sets portable data paths for the offline analysis.

## Run

Use Python 3.11 or later. The offline requirements below match the checked local environment.

```bash
python -m pip install -r v1_experiment/requirements.txt
python v1_experiment/run.py sources
python v1_experiment/run.py verify --data-root /path/to/paper_data
```

The full `paper_data` directory is external to Git. Preserve its `analysis/`, `cache/`, `inputs/`, `mainline/`, `micro/`, `tables/`, `protocols/`, `dialogue/`, `historical/`, and `provenance/` layout. Raw outputs, model weights, and credentials are not included here. A clean clone alone cannot reproduce data-dependent results.

With the full data directory available:

```bash
python v1_experiment/run.py tables --data-root /path/to/paper_data
python v1_experiment/run.py evidence --data-root /path/to/paper_data
python v1_experiment/run.py verify --data-root /path/to/paper_data
```

`tables` rebuilds the writing summaries. `evidence` rebuilds the figure-source tables and numerical checks; it does not render the final manuscript figures. Both write generated outputs within the supplied data directory. `verify` checks the exported data, 311,040 matched configurations, 15,552 auxiliary conditions, and the main predictor scores.

To rebuild the group results from cached measurements, run `group`, then `summarize`, before rebuilding the tables and evidence. `summarize` refreshes the group score tables and reconciles them against the original E4 report. Existing micro and dialogue exports must remain available. These CPU commands can take substantially longer. `validate-data` also needs all raw and native dialogue archives; `inventory` refreshes the data inventory.

The scripts in `frozen/` and `code/frozen_replication/` preserve historical execution paths and runtime assumptions. They are source snapshots, not portable GPU launch commands. The two-package offline environment does not supply their model-inference dependencies. No command above starts model inference or contacts a GPU server.

## Interpretation

The current paper analyses saved measurements. Topic bootstrap intervals use 20,000 draws and seed 20261105, with models treated as fixed. The meeting-format analysis is exploratory. The frozen confirmatory tests retain their original scope and outcomes. The generated-dialogue E1 relation kernel differs from the paper's transferred E2 size weights.

This directory does not change the manuscript or its authors.
