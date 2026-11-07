# LLM Meeting Format

Code and results for **Same Contacts, Different Rooms: Meeting Format Shapes LLM Group Influence Beyond Pairwise Contact Counts**.

The paper tests what changes when the same members and pairwise contact counts are organised into different meetings. **Regroup** changes who meets in triads. **Resize** merges triads into five-person meetings. Both interventions move the same 60 dyadic exposures.

Across six fixed models and 14 topics, resize changes one-step group support more than regroup: **4.36 versus 2.59 percentage points RMS**, a difference of **1.77 [1.50, 2.03]**, with resize larger in **81 of 84** model-topic cells. Resize also amplifies the initial core majority by **1.72 [1.41, 2.02] percentage points**.

Five pairwise predictors separate the form of influence weights from their aggregation placement: **Sum, Mean, Size, Stance, and Static network**. Their accuracy differs across models; negative scores are retained. With identical Size weights, meeting-wise computation improves resize R² over the static network by **0.51 for Granite4.2-3B** and **0.58 for LFM2.5-1.2B**. The placement difference is a sigmoid Jensen gap bounded by within-member logit variance.

## Repository

```text
src/llm_meeting_format/
  design.py          Contact-matched schedules
  predictors.py      Weight fitting and aggregation
  statistics.py      Topic bootstrap and contrast scores
  analysis.py        Paper estimates and robustness checks
  paper_tables.py    Manuscript table export and comparison
  validation.py      Reconstruction from cached measurements
scripts/reproduce.py Analysis entry point
results/             Current estimates, tables, and verification
data/                External measurement requirements and checksums
tests/               Scientific invariant checks
```

## Reproduce

Use Python 3.11 or later. The pinned packages match the checked analysis environment.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python scripts/reproduce.py --data-root /path/to/paper_data --output results
```

The frozen measurements are external to this repository; see [data requirements](data/README.md). A clean clone provides the analysis code and result tables, but does not include the full measurement dataset or model checkpoints.

For reconstruction from all cached tensors and an exact comparison with the manuscript tables:

```bash
python scripts/reproduce.py --data-root /path/to/paper_data --output results \
  --check-caches --manuscript /path/to/manuscript_source
```

The script writes only to the output directory. It does not run LLM inference, call an API, or modify the frozen inputs. The optional cache check reconstructs 311,040 configurations across 162 model-topic-readout jobs. The main analysis uses 161,280 configurations; the remainder support the scope checks. The auxiliary averaging analysis covers 15,552 conditions.

## Scope

This is an exploratory analysis of saved measurements. Percentile 95% intervals use 20,000 topic-cluster bootstrap draws and seed 20261105. Models are fixed, and pooled intervals resample topics jointly across models. Two-topic full-answer likelihood checks are reported without a cluster interval.

The outcomes are one-step responses from fixed initial stances. Meeting count and token cost are not held fixed. These are LLM collectives, not human groups. The repository follows the current paper's meeting-format analysis; historical experiments address separate hypotheses and are not presented as confirmations of these results.

The main effects and predictor statistics are recomputed from configuration-level measurements. Size and Stance weights are refitted from the micro-experiment with the target topic excluded. Archived micro power-fit and neutral-effect intervals are carried forward from their frozen estimates. The result files describe this scope explicitly.
