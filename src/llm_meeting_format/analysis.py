"""Recompute the paper's estimates, intervals, and predictor comparisons."""
import json
import numpy as np
import pandas as pd
from .data import MODELS, LABELS, COLUMNS, DISPLAY, conditions, read_json, topic_metadata
from .predictors import fit_weights
from .statistics import bootstrap_weights, bootstrap_indices, interval, rms_comparison, contrast_score, paired_gain


def format_effects(frame):
    records, pooled = [], []
    for scope, subset in frame.groupby("scope", sort=True):
        for weighting in ("person", "incidence"):
            resize_col, regroup_col = (("size_observed", "seat_observed") if weighting == "person" else
                                      ("delta_resize_incidence", "delta_regroup_incidence"))
            topics = subset.assign(resize_ms=subset[resize_col] ** 2, regroup_ms=subset[regroup_col] ** 2)
            topics = topics.groupby(["model_label", "topic_id"], sort=True)[["resize_ms", "regroup_ms"]].mean()
            for model, group in topics.groupby(level="model_label", sort=True):
                values = rms_comparison(group.resize_ms.to_numpy(), group.regroup_ms.to_numpy())
                records.append({"scope": scope, "weighting": weighting, "model": model, **values})
            resize = topics.resize_ms.unstack("topic_id").sort_index(axis=1).to_numpy()
            regroup = topics.regroup_ms.unstack("topic_id").sort_index(axis=1).to_numpy()
            draws = bootstrap_weights(resize.shape[1])
            low, high = interval(100 * (np.sqrt(draws @ resize.T) - np.sqrt(draws @ regroup.T)).mean(axis=1))
            pooled.append({"scope": scope, "weighting": weighting, "n_models": len(resize),
                           "n_topics": resize.shape[1], "resize_rms_pp": 100 * np.sqrt(resize.mean(axis=1)).mean(),
                           "regroup_rms_pp": 100 * np.sqrt(regroup.mean(axis=1)).mean(),
                           "difference_pp": 100 * (np.sqrt(resize.mean(axis=1)) - np.sqrt(regroup.mean(axis=1))).mean(),
                           "ci_low": low if resize.shape[1] > 2 else np.nan,
                           "ci_high": high if resize.shape[1] > 2 else np.nan,
                           "resize_larger_cells": int((resize > regroup).sum()),
                           "total_cells": int(resize.size),
                           "positive_models": int((resize.mean(axis=1) > regroup.mean(axis=1)).sum())})
    return pd.DataFrame(records), pd.DataFrame(pooled)


def prediction_scores(frame):
    topic_rows = []
    main = frame[frame.scope == "main"]
    for (model, topic), group in main.groupby(["model_label", "topic_id"], sort=True):
        for intervention, prefix in (("resize", "size"), ("regroup", "seat")):
            observed = group[prefix + "_observed"].to_numpy()
            for method, suffix in COLUMNS.items():
                if method == "observed":
                    continue
                predicted = group[prefix + "_" + suffix].to_numpy()
                topic_rows.append({"model": model, "topic": topic.rsplit("_", 1)[-1],
                                   "intervention": intervention, "method": method,
                                   "observed_ms": np.mean(observed ** 2),
                                   "error_ms": np.mean((observed - predicted) ** 2)})
    topic = pd.DataFrame(topic_rows)
    scores, comparisons = [], []
    for (model, intervention), group in topic.groupby(["model", "intervention"], sort=True):
        errors = group.pivot(index="topic", columns="method", values="error_ms").sort_index()
        observed = group.groupby("topic", sort=True).observed_ms.first().to_numpy()
        for method in DISPLAY:
            scores.append({"model": model, "intervention": intervention, "method": method,
                           **contrast_score(observed, errors[method].to_numpy())})
        for method, reference in (("size", "mean"), ("stance", "size"), ("size", "network")):
            comparisons.append({"model": model, "intervention": intervention, "method": method,
                                "reference": reference, **paired_gain(observed, errors[method], errors[reference])})
    return topic, pd.DataFrame(scores), pd.DataFrame(comparisons)


def majority_shifts(frame):
    main = frame[frame.scope == "main"]
    records, topic_values = [], {}
    for model in MODELS:
        group = main[main.model_label == model]
        columns = {method: 100 * group.majority_sign * group["size_" + suffix]
                   for method, suffix in COLUMNS.items()}
        columns["regroup_observed"] = 100 * group.majority_sign * group.seat_observed
        table = pd.DataFrame(columns).assign(topic_id=group.topic_id).groupby("topic_id", sort=True).mean()
        topic_values[model] = table
        low, high = interval(table.observed.to_numpy()[bootstrap_indices(len(table))].mean(axis=1))
        records.append({"model": model, **table.mean().to_dict(), "ci_low": low, "ci_high": high})
    values = np.stack([topic_values[model].to_numpy() for model in MODELS])
    mean_topics = values.mean(axis=0)
    low, high = interval(mean_topics[:, 0][bootstrap_indices(len(mean_topics))].mean(axis=1))
    records.append({"model": "All six", **dict(zip(table.columns, mean_topics.mean(axis=0))),
                    "ci_low": low, "ci_high": high})
    composition = main.groupby(["model_label", "core_supporters"], sort=True).size_observed.mean().reset_index()
    composition = composition.rename(columns={"model_label": "model", "size_observed": "resize_signed_pp"})
    composition.resize_signed_pp *= 100
    return pd.DataFrame(records), composition


def transferred_weights(root, frame):
    metadata = topic_metadata(root)
    keys = frame[frame.scope == "main"][["model", "model_label", "topic_id"]].drop_duplicates()
    records = []
    for row in keys.itertuples():
        size, _ = fit_weights(root, row.model, metadata[row.topic_id]["topic"])
        records.append({"model": row.model_label, "topic": row.topic_id.rsplit("_", 1)[-1],
                        "two_peers": size[2], "four_peers": size[4]})
    topic = pd.DataFrame(records)
    means = topic.groupby("model", sort=True)[["two_peers", "four_peers"]].mean().reset_index()
    means["total_influence_ratio"] = 4 * means.four_peers / (2 * means.two_peers)
    means["power_exponent"] = np.log(means.total_influence_ratio) / np.log(2)
    return means, topic


def auxiliary_averaging(root):
    frame = pd.read_csv(root / "mainline/figure_data/F2_E4c_jensen_conditions.csv",
                        dtype={"support_bits": str})
    assert len(frame) == 15552
    identity_error = np.abs(frame.delta_local - frame.delta_static + frame.jensen_gap_triangles).max()
    static_mean_error = np.abs(frame.simple_mean_static_delta).max()
    assert identity_error < 1e-12 and static_mean_error < 1e-12
    main = frame[frame.stage == "G2"]
    records, dispersion = [], []
    for model, group in main.groupby("model_label", sort=True):
        records.append({"model": model, "observed_rms_pp": 100 * np.sqrt(np.mean(group.delta_observed ** 2)),
                        "mean_meeting_rms_pp": 100 * np.sqrt(np.mean(group.simple_mean_local_delta ** 2)),
                        "mean_network_max_abs_pp": 100 * np.abs(group.simple_mean_static_delta).max()})
        for topic, part in group.groupby("topic_id", sort=True):
            dispersion.append({"model": model, "topic": topic.rsplit("_", 1)[-1],
                               "logit_variance": part.mean_within_person_event_logit_variance.mean(),
                               "jensen_rms_pp": 100 * np.sqrt(np.mean(part.jensen_gap_triangles ** 2))})
    dispersion = pd.DataFrame(dispersion)
    correlation = np.corrcoef(np.log(dispersion.logit_variance), np.log(dispersion.jensen_rms_pp))[0, 1]
    return pd.DataFrame(records), dispersion, {"auxiliary_conditions": len(frame),
        "max_jensen_identity_error": float(identity_error), "max_mean_network_contrast": float(static_mean_error),
        "dispersion_gap_log_log_correlation": float(correlation)}


def micro_summary(root):
    counts = pd.read_csv(root / "micro/count_curves_by_item.csv")
    counts = counts[counts.condition == "e2e"]
    neutral = pd.read_csv(root / "micro/neutral_summary.csv").set_index("model")
    fixed = pd.read_csv(root / "micro/fixed_share.csv")
    fixed = fixed[(fixed.cohort == "full64") & (fixed.ratio == "2:1")].set_index("model")
    records = []
    for model, group in counts.groupby("model", sort=True):
        fits = read_json(root / "inputs" / (model + "__e2e_R0_analysis.json"))
        ci = fits["bootstrap"]["intervals"]["power_t_no_allies"]["interval95"]
        six = group[(group.n_opponents == 6) & (group.n_allies == 0)].delta_opposition_logit.mean()
        n, f = neutral.loc[model], fixed.loc[model]
        records.append({"model": LABELS[model], "six_opponents_logit": six,
                        "power_exponent": fits["point"]["power_t_no_allies"],
                        "power_ci_low": ci[0] if ci else np.nan, "power_ci_high": ci[1] if ci else np.nan,
                        "neutral_effect": n.paired_logit_difference, "neutral_ci_low": n.ci_lo,
                        "neutral_ci_high": n.ci_hi, "fixed_share_effect": f.difference_logit,
                        "fixed_share_ci_low": f.ci_lo, "fixed_share_ci_high": f.ci_hi})
    curves = counts.groupby(["model", "n_opponents", "n_allies"], sort=True).delta_opposition_logit.mean().reset_index()
    curves.model = curves.model.map(LABELS)
    return pd.DataFrame(records).sort_values("six_opponents_logit", ascending=False), curves


def model_metadata(root):
    source = pd.read_csv(root / "tables/09_model_versions.csv")
    source["model_label"] = source.model.map(LABELS)
    source["role"] = np.where(source.model_label.isin(MODELS), "main + micro", "micro; earlier topics")
    return source[["model_label", "repo_id", "archived_download_revision", "role"]].rename(
        columns={"model_label": "model", "repo_id": "repository", "archived_download_revision": "revision"})


def run(root, output):
    output.mkdir(parents=True, exist_ok=True)
    frame = conditions(root)
    effects, scopes = format_effects(frame)
    topic, scores, comparisons = prediction_scores(frame)
    majority, composition = majority_shifts(frame)
    weights, weight_topics = transferred_weights(root, frame)
    averaging, dispersion, diagnostics = auxiliary_averaging(root)
    micro, curves = micro_summary(root)
    tables = {"format_effects": effects[effects.scope == "main"], "scope_checks": scopes,
              "prediction_topic": topic, "predictor_scores": scores, "predictor_gains": comparisons,
              "majority_shifts": majority, "majority_by_composition": composition,
              "size_weights": weights, "size_weights_by_topic": weight_topics,
              "auxiliary_averaging": averaging, "logit_dispersion": dispersion,
              "micro_summary": micro, "micro_count_curves": curves, "model_revisions": model_metadata(root)}
    for name, table in tables.items():
        table.to_csv(output / (name + ".csv"), index=False, float_format="%.17g")
    person = scopes[(scopes.scope == "main") & (scopes.weighting == "person")].iloc[0]
    diagnostics.update({"matched_configurations": len(frame), "main_configurations": int((frame.scope == "main").sum()),
                        "resize_minus_regroup_pp": float(person.difference_pp),
                        "resize_minus_regroup_ci": [float(person.ci_low), float(person.ci_high)],
                        "majority_amplification_pp": float(majority.iloc[-1].observed),
                        "main_predictor_scores": len(scores), "new_model_inference": False,
                        "analysis": "Exploratory; 20000 topic-cluster draws; six fixed main models."})
    (output / "verification.json").write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    return tables, diagnostics
