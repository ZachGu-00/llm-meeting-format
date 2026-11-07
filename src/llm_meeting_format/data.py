"""Read the frozen measurement export without changing source files."""
import json
from pathlib import Path
import pandas as pd


MODELS = ["Granite4.2-3B", "LFM2.5-1.2B", "Gemma4-E2B", "MiniCPM5-2B", "Phi4-mini", "Qwen3.5-4B"]
LABELS = {
    "ibm-granite--granite-4.2-3b": "Granite4.2-3B",
    "LiquidAI--LFM2.5-1.2B-Instruct": "LFM2.5-1.2B",
    "google--gemma-4-E2B-it": "Gemma4-E2B", "openbmb--MiniCPM5-2B": "MiniCPM5-2B",
    "microsoft--Phi-4-mini-instruct": "Phi4-mini", "Qwen--Qwen3.5-4B": "Qwen3.5-4B",
    "google--gemma-4-E4B-it": "Gemma4-E4B", "Qwen--Qwen3.5-2B": "Qwen3.5-2B",
    "HuggingFaceTB--SmolLM3-3B": "SmolLM3-3B",
    "mistralai--Ministral-3-3B-Instruct-2512-BF16": "Ministral3-3B",
    "nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16": "Nemotron3-Nano-4B",
}
SCOPES = {"G2": "main", "E4": "additional_topics", "G1": "additional_models", "G3": "answer_likelihood"}
COLUMNS = {"observed": "observed", "sum": "P0", "mean": "present_mean",
           "size": "size_E2", "stance": "state_E2", "network": "static_projection_E2"}
DISPLAY = {"sum": "Sum", "mean": "Mean", "size": "Size", "stance": "Stance", "network": "Static network"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def conditions(root):
    frame = pd.read_csv(root / "mainline/figure_data/F1_matched_conditions.csv.gz",
                        dtype={"support_bits": str})
    assert len(frame) == 311040 and frame.support_bits.str.len().eq(6).all()
    assert frame.groupby(["stage", "model", "topic_id", "readout"]).size().eq(1920).all()
    frame["scope"] = frame.stage.map(SCOPES)
    frame["model_label"] = frame.model.map(LABELS)
    frame["core_supporters"] = frame.support_bits.str[:5].str.count("1")
    frame["majority_sign"] = 2 * frame.core_supporters.ge(3).astype(int) - 1
    return frame


def topic_metadata(root):
    records = []
    for filename in ("e4_cache_manifest.json", "M1_manifest.json"):
        records.extend(read_json(root / "inputs" / filename)["topics"])
    return {row["topic_id"]: row for row in records}
