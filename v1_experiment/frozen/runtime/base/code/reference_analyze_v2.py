"""V2 sections5/9 candidate analysis. No old probability-calibration gates."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np

SCREEN = (0, 1, 2, 4, 3, 5, 6)
FACTORS = ("item_id", "topic", "subset_mask", "permutation", "label_map", "positive_letter")
ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAPS = 10000
SEED = 20261030


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    if Path(path).suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    value = json.loads(text)
    return value if isinstance(value, list) else value["rows"]


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_scored(path, input_path):
    """Independently bind a closed run to exact input, model and option IDs."""
    inputs = read_rows(input_path)
    normalized = [{**row, "record_id": row_id(row)} for row in inputs]
    rows = read_rows(path)
    run_path = Path(str(path) + ".run.json")
    summary_path = Path(str(path) + ".summary.json")
    if not run_path.exists() or not summary_path.exists():
        return rows, {"status": "incomplete", "reason": "run_or_completion_summary_missing"}
    run = json.loads(run_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if run["input_file_sha256"] != sha(input_path) or run["input_rows_sha256"] != canonical(normalized):
        raise ValueError("Run input hash mismatch")
    if run["n_input"] != len(inputs) or run["input_record_ids_sha256"] != canonical([row_id(row) for row in inputs]):
        raise ValueError("Run planned IDs mismatch")
    if summary.get("status") != "complete":
        return rows, {"status": "incomplete", "reason": "generation_summary_not_complete"}
    if any(summary.get(k) != v for k, v in run.items()) or summary["output_sha256"] != sha(path) or summary["output_bytes"] != Path(path).stat().st_size:
        raise ValueError("Closed run output/identity mismatch")
    if [row_id(row) for row in rows] != [row_id(row) for row in inputs] or summary["n_output"] != len(inputs):
        raise ValueError("Closed run count/order differs")
    if run["engine"].get("precision") != "native_bfloat16":
        raise ValueError("V2 requires unquantized native BF16 execution")
    for source, row in zip(normalized, rows):
        if row.get("run_file_sha256") != sha(run_path) or any(row.get(k) != v for k, v in source.items()):
            raise ValueError("Row input/run binding mismatch")
        if row.get("status") != "ok" or row.get("option_ids") != run["option_ids"]:
            raise ValueError("Invalid row or option IDs drift")
    return rows, {"status": "complete", "run": run, "run_sha256": sha(run_path), "summary_sha256": sha(summary_path)}


def row_id(row):
    value = row.get("record_id", row.get("id"))
    if not isinstance(value, str) or ("id" in row and row["id"] != value):
        raise ValueError("Record identity missing or conflicting")
    return value


def mobius(ybar, n=5):
    if len(ybar) != 2**n:
        raise ValueError("Full subset lattice required for Mobius inversion")
    result = np.asarray(ybar, dtype=float).copy()
    for bit in range(n):
        for mask in range(1 << n):
            if mask & (1 << bit):
                result[mask] -= result[mask ^ (1 << bit)]
    return result


def zeta(coefficients, n=5):
    result = np.asarray(coefficients, dtype=float).copy()
    if len(result) != 2**n:
        raise ValueError("Full lattice required")
    for bit in range(n):
        for mask in range(1 << n):
            if mask & (1 << bit):
                result[mask] += result[mask ^ (1 << bit)]
    return result


def design_index(inputs):
    ids = {row_id(row): row for row in inputs}
    if len(ids) != len(inputs):
        raise ValueError("Duplicate planned IDs")
    groups = defaultdict(dict)
    for row in inputs:
        if row.get("split") == "token_calibration":
            raise ValueError("Token calibration controls cannot enter scientific estimates")
        item = groups[row["item_id"]]
        key = (row["subset_mask"], row["permutation"], row["label_map"])
        if key in item or row["positive_letter"] not in ("A", "B"):
            raise ValueError("Duplicate cell or invalid positive letter")
        if row["subset_mask"] == 0:
            if row["permutation"] != 0 or row.get("empty_slot_replication") is not True:
                raise ValueError("Empty must be two explicitly deduplicated source prompts")
            if row.get("replicate_slot_indices", list(range(10))) != list(range(10)):
                raise ValueError("Empty replication must cover the prescribed ten slots")
        item[key] = row
    mask_sets = {tuple(sorted({key[0] for key in item})) for item in groups.values()}
    if len(mask_sets) != 1:
        raise ValueError("Items have different prescribed subset panels")
    masks = next(iter(mask_sets)) if mask_sets else ()
    if set(masks) not in (set(SCREEN), set(range(32))):
        raise ValueError("Require the seven-mask screen or complete32 design")
    expected = {(mask, slot, label) for mask in masks for slot in ([0] if mask == 0 else range(10)) for label in (0, 1)}
    for item in groups.values():
        if set(item) != expected or len({r["topic"] for r in item.values()}) != 1:
            raise ValueError("Incomplete planned factorial item or changing topic")
        for mask in masks:
            for slot in ([0] if mask == 0 else range(10)):
                if item[(mask, slot, 0)]["positive_letter"] == item[(mask, slot, 1)]["positive_letter"]:
                    raise ValueError("Labels do not reverse positive direction")
        for label in (0, 1):
            if len({row["positive_letter"] for key, row in item.items() if key[2] == label}) != 1:
                raise ValueError("Positive direction changes across subsets or slots")
    return ids, groups, masks


def prepare(inputs, rows, readout="R0"):
    planned, design, masks = design_index(inputs)
    actual = {row_id(row): row for row in rows}
    if len(actual) != len(rows) or not set(actual) <= set(planned):
        raise ValueError("Duplicate/unplanned output IDs")
    missing = sorted(set(planned) - set(actual))
    if missing:
        return {"status": "incomplete", "missing_record_ids": missing, "n_missing": len(missing)}
    item_ids = sorted(design)
    oriented = np.empty((len(item_ids), len(masks), 10, 2))
    raw = np.empty_like(oriented)
    field = "R0_y_AB" if readout == "R0" else "R1_y_AB"
    missing_readout = []
    for i, item_id in enumerate(item_ids):
        for m, mask in enumerate(masks):
            for slot, label in itertools.product(range(10), (0, 1)):
                source = design[item_id][(mask, 0 if mask == 0 else slot, label)]
                result = actual[row_id(source)]
                for key in FACTORS:
                    if result.get(key) != source[key]:
                        raise ValueError("Output metadata changed: " + key)
                for key in ("messages", "candidates", "template_variant", "split"):
                    if key in source and result.get(key) != source[key]:
                        raise ValueError("Output input binding differs: " + key)
                value = result.get(field, result.get("y_AB") if readout == "R0" else None)
                if value is None:
                    missing_readout.append(row_id(source))
                    continue
                if not isinstance(value, (float, int)) or not math.isfinite(value):
                    raise ValueError("Nonfinite model readout")
                if readout == "R0":
                    q = result.get("q")
                    if not isinstance(q, (float, int)) or not math.isfinite(q) or not 0 <= q <= 1 + 1e-6:
                        raise ValueError("Missing/invalid full-vocabulary option mass")
                    if all(key in result for key in ("logprob_A", "logprob_B")):
                        expected_q = math.exp(result["logprob_A"]) + math.exp(result["logprob_B"])
                        if abs(q - expected_q) > 1e-6:
                            raise ValueError("q is not the full-vocabulary candidate mass")
                raw[i, m, slot, label] = value
                oriented[i, m, slot, label] = value * (1 if source["positive_letter"] == "A" else -1)
    if missing_readout:
        return {"status": "incomplete", "missing_readout_ids": sorted(set(missing_readout))}
    return {"status": "complete", "item_ids": item_ids, "topics": [next(iter(design[i].values()))["topic"] for i in item_ids],
        "masks": list(masks), "oriented": oriented, "raw": raw, "physical_rows": list(actual.values()),
        "item_sources": [next(iter(design[i].values())) for i in item_ids]}


def components(panel):
    slots = panel["oriented"].mean(axis=3)
    means = slots.mean(axis=2)
    bymask = {mask: means[:, j] for j, mask in enumerate(panel["masks"])}
    singles = np.stack([bymask[mask] - bymask[0] for mask in (1, 2, 4)], axis=1)
    m = np.stack([bymask[a | b] - bymask[a] - bymask[b] + bymask[0] for a, b in ((1, 2), (1, 4), (2, 4))], axis=1)
    den = np.stack([(singles[:, a] + singles[:, b]) / 2 for a, b in ((0, 1), (0, 2), (1, 2))], axis=1)
    return slots, means, singles, m, den


def ratio_summary(numerator, denominator, weights=None):
    weights = np.ones(len(numerator)) if weights is None else np.asarray(weights)
    num = float(weights @ np.asarray(numerator))
    den = float(weights @ np.asarray(denominator))
    cancellation = abs(den) <= 1e-12 * float(weights @ np.abs(denominator)) if np.any(denominator) else True
    return {"numerator_sum": num, "denominator_sum": den,
        "denominator_positive": den > 0, "numerical_cancellation": cancellation,
        "ratio": num / den if den > 0 and not cancellation else None}


def primary_point(singles, interactions, denominators, weights=None):
    ratios = [ratio_summary(interactions[:, j], denominators[:, j], weights) for j in range(3)]
    w = [1 + r["ratio"] if r["ratio"] is not None else None for r in ratios]
    valid = all(value is not None for value in w)
    return {"w_same": w[0], "w_diff": (w[1] + w[2]) / 2 if valid else None,
        "ASI": (w[1] + w[2]) / 2 - w[0] if valid else None,
        "pair_weights": dict(zip(("A_Aprime", "A_C", "Aprime_C"), w)),
        "pair_ratio_diagnostics": dict(zip(("A_Aprime", "A_C", "Aprime_C"), ratios)),
        "vbar": float(singles.mean()), "all_primary_denominators_positive_stable": valid,
        "nonpositive_opponent_singleton_fraction": float(np.mean(singles <= 0))}


def cluster_weights(topics, draws=BOOTSTRAPS, seed=SEED):
    unique = sorted(set(topics))
    if len(unique) < 2:
        return None
    rng = np.random.default_rng(seed)
    sampled = rng.multinomial(len(unique), np.repeat(1 / len(unique), len(unique)), size=draws)
    mapping = {topic: j for j, topic in enumerate(unique)}
    return sampled[:, [mapping[topic] for topic in topics]]


def bootstrap_primary(singles, interactions, denominators, topics, draws=BOOTSTRAPS, seed=SEED):
    weights = cluster_weights(topics, draws, seed)
    if weights is None:
        return {"status": "incomplete", "reason": "fewer_than_two_topic_clusters", "draws": draws}
    numerators, dens = weights @ interactions, weights @ denominators
    absdens = weights @ np.abs(denominators)
    bad = (dens <= 0) | (np.abs(dens) <= 1e-12 * absdens)
    if np.any(bad):
        return {"status": "unstable_denominator", "draws": draws, "n_topic_clusters": len(set(topics)),
            "invalid_replicate_count": int(np.any(bad, axis=1).sum()),
            "denominator_95_intervals": np.percentile(dens, [2.5, 97.5], axis=0).T.tolist(),
            "no_invalid_bootstrap_replicates_dropped": True, "ASI_95_interval": None, "ASI_95_halfwidth": None}
    w = 1 + numerators / dens
    values = np.stack([w[:, 0], (w[:, 1] + w[:, 2]) / 2, (w[:, 1] + w[:, 2]) / 2 - w[:, 0]], axis=1)
    intervals = np.percentile(values, [2.5, 97.5], axis=0).T
    return {"status": "complete", "draws": draws, "seed": seed, "n_topic_clusters": len(set(topics)),
        "method": "resample_topics_with_replacement_keep_all_items_recompute_ratio_of_sums",
        "intervals": dict(zip(("w_same", "w_diff", "ASI"), intervals.tolist())),
        "ASI_95_interval": intervals[2].tolist(), "ASI_95_halfwidth": float((intervals[2, 1] - intervals[2, 0]) / 2)}


def crossfit_item_weights(panel):
    slots, _, _, _, _ = components(panel)
    mask_index = {mask: j for j, mask in enumerate(panel["masks"])}
    halves = [slots[:, :, :5].mean(axis=2), slots[:, :, 5:].mean(axis=2)]
    pairs = ((1, 2), (1, 4), (2, 4))
    output = []
    for item_index, item_id in enumerate(panel["item_ids"]):
        record = {"item_id": item_id, "pairs": {}}
        for a, b in pairs:
            folds = []
            for selection_half, evaluation_half in ((0, 1), (1, 0)):
                train, test = halves[selection_half], halves[evaluation_half]
                all_v = np.stack([train[:, mask_index[s]] - train[:, mask_index[0]] for s in (1, 2, 4)], axis=1)
                vbar = float(all_v.mean())
                select_den = float((train[item_index, mask_index[a]] + train[item_index, mask_index[b]]) / 2 - train[item_index, mask_index[0]])
                selected = vbar > 0 and select_den >= .25 * vbar
                denominator = float((test[item_index, mask_index[a]] + test[item_index, mask_index[b]]) / 2 - test[item_index, mask_index[0]])
                interaction = float(test[item_index, mask_index[a | b]] - test[item_index, mask_index[a]] - test[item_index, mask_index[b]] + test[item_index, mask_index[0]])
                folds.append({"selection_half": selection_half, "selected": selected, "selection_vbar": vbar,
                    "selection_denominator": select_den, "evaluation_denominator": denominator,
                    "w": 1 + interaction / denominator if selected and denominator > 0 else None})
            # Retain both fold decisions; do not silently average only an admitted fold.
            record["pairs"][str(a) + "," + str(b)] = {"folds": folds,
                "two_fold_mean": float(np.mean([f["w"] for f in folds])) if all(f["w"] is not None for f in folds) else None}
        output.append(record)
    return {"scope": "item_level_descriptive_only_not_filtering_primary_aggregate", "items": output}


def bootstrap_ratio(numerator, denominator, topics, draws=BOOTSTRAPS, allow_negative=False):
    """Ratio of sums; keep every topic draw and flag sign-changing denominators."""
    numerator, denominator = np.asarray(numerator), np.asarray(denominator)
    den = float(denominator.sum())
    scale = float(np.abs(denominator).sum())
    valid = abs(den) > 1e-12 * scale and (allow_negative or den > 0)
    if not valid:
        return {"status": "unstable_denominator", "estimate": None, "interval95": None, "interval90": None}
    estimate = float(numerator.sum() / den)
    weights = cluster_weights(topics, draws)
    if weights is None:
        return {"status": "incomplete", "estimate": estimate, "interval95": None, "interval90": None,
            "reason": "fewer_than_two_topic_clusters"}
    dens = weights @ denominator
    unstable = (dens * den <= 0) | (np.abs(dens) <= 1e-12 * (weights @ np.abs(denominator)))
    if np.any(unstable):
        return {"status": "unstable_denominator", "estimate": estimate, "interval95": None, "interval90": None,
            "invalid_replicates": int(unstable.sum()), "no_replicates_dropped": True}
    boot = (weights @ numerator) / dens
    return {"status": "complete", "estimate": estimate, "interval95": np.percentile(boot, [2.5, 97.5]).tolist(),
        "interval90": np.percentile(boot, [5, 95]).tolist(), "draws": draws, "seed": SEED}


def full_normalized(panel, coefficients, means, draws=BOOTSTRAPS):
    singles = means[:, [1, 2, 4, 8, 16]] - means[:, [0]]
    output = {}
    for mask in range(1, 32):
        bits = [bit for bit in range(5) if mask & (1 << bit)]
        if len(bits) < 2:
            continue
        denominator = np.abs(singles[:, bits]).mean(axis=1)
        estimate = bootstrap_ratio(coefficients[:, mask], denominator, panel["topics"], draws)
        ci = estimate["interval90"]
        equivalent = ci[0] > -.1 and ci[1] < .1 if ci is not None else None
        output[str(mask)] = {"order": len(bits), "normalization": "ratio_of_mean_m_to_mean_item_mean_absolute_singletons",
            **estimate, "bootstrap_TOST_approximate_equivalent_margin_0_1": equivalent,
            "TOST_rule": "90_percentile_cluster_CI_strictly_inside_(-0.1,0.1); approximate bootstrap TOST, not a parametric t-test"}
    ally = {}
    for relation in ("same", "different"):
        ids = [i for i, source in enumerate(panel["item_sources"]) if source.get("ally_relation") == relation]
        if not ids:
            ally[relation] = {"status": "incomplete", "reason": "no_prespecified_items_with_this_ally_relation"}
            continue
        den = (singles[ids, 3] + singles[ids, 4]) / 2
        estimate = bootstrap_ratio(coefficients[ids, 24], den, [panel["topics"][i] for i in ids], draws, allow_negative=True)
        same_sign = float(singles[ids, 3].mean()) * float(singles[ids, 4].mean()) > 0
        ally[relation] = {**estimate, "n_items": len(ids), "aggregate_singletons_same_sign": same_sign,
            "w_ally": 1 + estimate["estimate"] if same_sign and estimate["estimate"] is not None else None,
            "w_ally_95_interval": [1 + x for x in estimate["interval95"]] if same_sign and estimate["interval95"] is not None else None}
    mixed = {str(a | b): output[str(a | b)] for a in (1, 2, 4) for b in (8, 16)}
    mixvalues = [value["estimate"] for value in mixed.values()]
    return {"by_mask": output, "ally_weights_by_prespecified_relation": ally,
        "mixed_six_pairs": mixed, "mixed_equal_pair_mean": float(np.mean(mixvalues)) if all(v is not None for v in mixvalues) else None,
        "all_tested_interactions_bootstrap_equivalent": all(v["bootstrap_TOST_approximate_equivalent_margin_0_1"] is True for v in output.values()),
        "no_additivity_claim_from_non_significance": True,
        "uncertainty": "topic-cluster percentile bootstrap, approximate finite-cluster inference; no render pseudoreplication"}


def l3_reversals(panel, draws=2000):
    slots = panel["oriented"].mean(axis=3)
    index = {mask: j for j, mask in enumerate(panel["masks"])}
    results = []
    for i, item in enumerate(panel["item_ids"]):
        d1 = slots[i, index[2], :] - slots[i, index[4], :]
        d2 = slots[i, index[3], :] - slots[i, index[5], :]
        seed = int(hashlib.sha256(("v2_L3_2000\n" + item).encode()).hexdigest()[:16], 16)
        selected = np.random.default_rng(seed).integers(0, 10, size=(draws, 10))
        boot = np.stack((d1[selected].mean(axis=1), d2[selected].mean(axis=1)), axis=1)
        ci = np.percentile(boot, [2.5, 97.5], axis=0).T
        reliable = (ci[0, 0] > 0 and ci[1, 1] < 0) or (ci[0, 1] < 0 and ci[1, 0] > 0)
        results.append({"item_id": item, "topic": panel["topics"][i], "d1": float(d1.mean()), "d2": float(d2.mean()),
            "d1_95_interval": ci[0].tolist(), "d2_95_interval": ci[1].tolist(),
            "reliable_reversal": bool(reliable), "direction": "+-" if reliable and d1.mean() > 0 else "-+" if reliable else None})
    return {"n_items": len(results), "slot_bootstrap_draws": draws, "joint_resampling_of_both_differences": True,
        "reliable_reversal_count": sum(row["reliable_reversal"] for row in results),
        "reliable_reversal_fraction": float(np.mean([row["reliable_reversal"] for row in results])), "items": results,
        "scope": "Within-item slot robustness of label-averaged aggregated observables; no latent copy-weight identification.",
        "invariance": "A common strictly increasing transform of already aggregated cell values preserves rank signs; applying it before averaging may not."}


def gate(value=None, passed=None, reason=None):
    return {"status": "incomplete" if passed is None else "pass" if passed else "failed", "value": value, "reason": reason}


def generated_checks(panel):
    rows = sorted(panel["physical_rows"], key=lambda r: (hashlib.sha256(row_id(r).encode()).hexdigest(), row_id(r)))
    results = {}
    for name, size, field in (("first_token_format", 50, "diagnostic_format_selected"), ("R2_concordance", 200, "diagnostic_r2_selected")):
        selected = rows[:size]
        if len(selected) != size:
            results[name] = gate(reason="fewer_than_fixed_diagnostic_count")
            continue
        missing = [row_id(r) for r in selected if r.get(field) is not True or "generated_first_token_id" not in r or not isinstance(r.get("generated_first_token_text"), str)]
        if missing:
            results[name] = gate(reason="missing_prespecified_real_generate_rows")
            results[name]["missing_ids"] = missing
            continue
        success = 0
        ties = 0
        alias_differences = []
        for row in selected:
            decoded = row["generated_first_token_text"].strip()
            letter = decoded if decoded in ("A", "B") else None
            if row.get("generated_first_token_letter") != letter:
                alias_differences.append(row_id(row))
            if name == "first_token_format":
                success += letter in ("A", "B")
            else:
                value = row.get("R0_y_AB", row.get("y_AB"))
                expected = "A" if value > 0 else "B" if value < 0 else None
                ties += expected is None
                success += expected is not None and letter == expected
        results[name] = gate(success / size, success >= (48 if size == 50 else 180))
        results[name].update(n=size, successes=success, ties_counted_nonconcordant=ties,
            selected_ids=[row_id(r) for r in selected], decoded_text_vs_engine_letter_differences=alias_differences)
    return results


def measure(panel, draws=BOOTSTRAPS):
    slots, means, singles, interactions, dens = components(panel)
    point = primary_point(singles, interactions, dens)
    bootstrap = bootstrap_primary(singles, interactions, dens, panel["topics"], draws)
    index = {mask: j for j, mask in enumerate(panel["masks"])}
    slot_sd = slots[:, [index[m] for m in panel["masks"] if m != 0], :].std(axis=2, ddof=0)
    vbar = point["vbar"]
    ratio = float(slot_sd.mean() / vbar) if vbar > 0 else None
    a_forward = slots[:, index[1], :5].mean(axis=1) - slots[:, index[0], :5].mean(axis=1)
    a_reverse = slots[:, index[1], 5:].mean(axis=1) - slots[:, index[0], 5:].mean(axis=1)
    correlation = float(np.corrcoef(a_forward, a_reverse)[0, 1]) if len(singles) >= 2 and np.std(a_forward) > 0 and np.std(a_reverse) > 0 else None
    baseline = means[:, index[0]]
    saturation = float(np.mean(np.abs(baseline) > math.log(999)))
    label_bias = float(panel["raw"].mean(axis=3).mean())
    gates = {"positive_vbar": gate(vbar, vbar > 0),
        "primary_ratio_denominators": gate(point["all_primary_denominators_positive_stable"], point["all_primary_denominators_positive_stable"]),
        "slot_effect": gate(ratio, ratio is not None and ratio <= .5),
        "split_half_vA_Pearson": gate(correlation, correlation is not None and correlation >= .7),
        "empty_saturation": gate(saturation, saturation <= .2),
        "ASI_precision": gate(bootstrap.get("ASI_95_halfwidth"),
            bootstrap["ASI_95_halfwidth"] <= .15 if bootstrap["status"] == "complete" else None if bootstrap["status"] == "incomplete" else False,
            bootstrap["status"] if bootstrap["status"] != "complete" else None)}
    result = {"point": point, "bootstrap": bootstrap, "gates": gates,
        "slot_effect": {"population_sd_ddof": 0, "empty_excluded": True, "mean_sd": float(slot_sd.mean()),
            "sd_p95": float(np.percentile(slot_sd, 95)), "mean_sd_over_vbar": ratio,
            "p95_sd_over_vbar": float(np.percentile(slot_sd, 95) / vbar) if vbar > 0 else None,
            "distribution": slot_sd.tolist()},
        "label_bias": {"mean_raw_AB_label_average": label_bias, "over_vbar": label_bias / vbar if vbar > 0 else None, "gate": "report_only"},
        "item_means": {item: {str(mask): float(means[i, j]) for j, mask in enumerate(panel["masks"])} for i, item in enumerate(panel["item_ids"])},
        "crossfit_item_weights": crossfit_item_weights(panel)}
    result["L3"] = l3_reversals(panel)
    if set(panel["masks"]) == set(range(32)):
        coefficients = np.stack([mobius(row) for row in means])
        result["full_mobius"] = {"coefficients_by_item": dict(zip(panel["item_ids"], coefficients.tolist())),
            "aggregate_mean_by_mask": coefficients.mean(axis=0).tolist(),
            "all_orders_available": True, "absence_of_significance_called_additivity": False}
        result["full_normalized_interactions"] = full_normalized(panel, coefficients, means, draws)
    else:
        result["full_mobius"] = {"all_orders_available": False, "status": "not_identified_from_seven_masks", "higher_order_claims_allowed": False}
    return result


def overall(gates):
    states = [value["status"] for value in gates.values()]
    return "failed" if "failed" in states else "incomplete" if "incomplete" in states else "pass"


def analyze(inputs, r0_rows, r1_rows=None, draws=BOOTSTRAPS):
    panel = prepare(inputs, r0_rows)
    if panel["status"] != "complete":
        return {"status": "incomplete", "R0": panel, "development_acceptance": "incomplete"}
    r0 = measure(panel, draws)
    gates = dict(r0["gates"])
    q = [row["q"] for row in panel["physical_rows"]]
    gates["option_mass_median"] = gate(float(np.median(q)), np.median(q) >= .9)
    gates.update(generated_checks(panel))
    r1_inputs = [row for row in inputs if row["subset_mask"] in SCREEN]
    if r1_rows is None:
        r1_rows = [row for row in r0_rows if row["subset_mask"] in SCREEN and row.get("R1_y_AB") is not None]
    else:
        r1_rows = [row for row in r1_rows if row["subset_mask"] in SCREEN]
    r1_panel = prepare(r1_inputs, r1_rows, "R1")
    r1 = {"status": "incomplete"}
    differences = None
    if r1_panel["status"] == "complete":
        _, _, singles, interactions, denominators = components(r1_panel)
        r1 = primary_point(singles, interactions, denominators)
        keys = ("w_same", "w_diff", "ASI")
        if all(r0["point"][key] is not None and r1[key] is not None for key in keys):
            differences = {key: abs(r0["point"][key] - r1[key]) for key in keys}
            gates["R0_R1_primary_agreement"] = gate(differences, max(differences.values()) <= .15)
        else:
            gates["R0_R1_primary_agreement"] = gate(reason="undefined_primary_ratio", passed=False)
    else:
        gates["R0_R1_primary_agreement"] = gate(reason="missing_full_seven_mask_R1_panel")
    screen_status = overall(gates)
    full_design = set(panel["masks"]) == set(range(32))
    intended_panel = len(panel["item_ids"]) >= 100 and full_design and draws == BOOTSTRAPS
    return {"schema": "nonadditivity_v2_analysis_v1", "status": screen_status,
        "development_acceptance": screen_status if intended_panel else "incomplete",
        "intended_development_panel_complete": intended_panel, "n_items": len(panel["item_ids"]),
        "n_topic_clusters": len(set(panel["topics"])), "n_unique_prompts": len(inputs),
        "subset_masks": panel["masks"], "full32_R0": full_design,
        "screen_pass_does_not_imply_full_model_acceptance": not intended_panel,
        "gates": gates, "R0": r0, "R1_primary": r1,
        "low_q_ids": [row_id(row) for row in panel["physical_rows"] if row["q"] < .5],
        "low_q_rows_not_removed_from_primary": True, "quantization_gate": "not_applicable_BF16_required_by_execution",
        "L3_scope": "Rank invariance applies to a common monotone transform of defined aggregated cells, not arbitrary transforms before averaging.",
        "mobius_sign_correction": "Interaction signs need not survive strictly monotone nonlinear transforms; positive affine transforms preserve them.",
        "no_probability_MAE_or_old_R4_R5_gate": True, "publication_material_readiness_checked": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--r0", required=True, type=Path)
    parser.add_argument("--r1", type=Path)
    parser.add_argument("--r1-inputs", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Do not overwrite an analysis")
    r0, audit0 = read_scored(args.r0, args.inputs)
    r1, audit1 = None, None
    if args.r1:
        r1, audit1 = read_scored(args.r1, args.r1_inputs or args.inputs)
        if audit0["status"] == audit1["status"] == "complete":
            for key in ("engine", "calibration_file_sha256", "option_ids"):
                if audit0["run"][key] != audit1["run"][key]:
                    raise ValueError("R0/R1 model/calibration identity differs")
    result = analyze(read_rows(args.inputs), r0, r1)
    result["execution_integrity"] = {"R0": audit0, "R1": audit1}
    if audit0["status"] != "complete" or (audit1 is not None and audit1["status"] != "complete"):
        result["development_acceptance"] = "incomplete"
        result["execution_integrity_incomplete"] = True
    paths = [Path(__file__), ROOT / "protocol_v2.txt", args.inputs, args.r0]
    if args.r1:
        paths.append(args.r1)
    if args.r1_inputs:
        paths.append(args.r1_inputs)
    result["file_sha256"] = {str(path.resolve()): sha(path) for path in paths}
    result["execution_identity_requires_separate_frozen_runner_audit"] = True
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "development_acceptance")}))


if __name__ == "__main__":
    main()
