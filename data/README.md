# Frozen measurement inputs

Pass the original `paper_data` directory to `scripts/reproduce.py --data-root`.

| Input | Purpose |
|---|---|
| `mainline/figure_data/F1_matched_conditions.csv.gz` | Observed and predicted schedule contrasts for 311,040 configurations |
| `mainline/figure_data/F2_E4c_jensen_conditions.csv` | Auxiliary five-person versus triad conditions |
| `inputs/e1_semantic64_items.json` | Micro item-to-topic mapping for held-out-topic fitting |
| `inputs/*__e2e_R0_analysis.json` | Item-level opponent/ally count responses and archived power fits |
| `micro/count_curves_by_item.csv` | Count-response measurements |
| `micro/neutral_summary.csv` | Archived neutral-statement estimates and intervals |
| `micro/fixed_share.csv` | Fixed opponent:ally share estimates and intervals |
| `tables/09_model_versions.csv` | Pinned model download revisions |
| `inputs/M1_manifest.json`, `inputs/e4_cache_manifest.json` | Topic and five-member-core definitions |
| `cache/*.npz` | Baseline, pair, triad, and five-person logit tensors for optional reconstruction |

Historical identifiers are translated by the input adapter into `main`, `additional_topics`, `additional_models`, and `answer_likelihood`. Public result tables use the paper's terms: `resize`, `regroup`, `sum`, `mean`, `size`, `stance`, and `network`.

`manifest.json` records checksums of the inputs used for the verified reproduction. It is a provenance record, not a data download. Large measurement files, prompts, raw model outputs, and model weights are not included in Git.
