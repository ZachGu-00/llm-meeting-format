"""Export the nine manuscript tables and check their published values."""
from pathlib import Path
import re
import numpy as np
from .data import MODELS


FILES = {"format_effects": "table1_rows.tex", "models": "tableA_models_rows.tex",
         "scope_checks": "tableA_scope_rows.tex", "predictor_scores": "tableA_r2_rows.tex",
         "weight_comparisons": "table2_rows.tex", "size_weights": "tableA_weights_rows.tex",
         "majority_shifts": "tableA_majpred_rows.tex", "auxiliary_averaging": "tableA_simplemean_rows.tex",
         "micro_summary": "tableA_micro_rows.tex"}


def number(value, digits=2):
    if value is None or not np.isfinite(value):
        return "--"
    text = f"{abs(value):.{digits}f}"
    return ("$-$" if value < 0 and float(text) > 0 else "") + text


def estimate(value, low, high):
    text = number(value)
    return text + (" [" + number(low) + ", " + number(high) + "]" if np.isfinite(low) and np.isfinite(high) else "")


def row(values):
    return " & ".join(values) + r" \\" + "\n"


def one(frame, **where):
    selected = frame
    for name, value in where.items():
        selected = selected[selected[name] == value]
    assert len(selected) == 1, where
    return selected.iloc[0]


def render(tables):
    result = {name: "" for name in FILES}
    effects, majority = tables["format_effects"], tables["majority_shifts"]
    for model in [*MODELS, "All six"]:
        m = one(majority, model=model)
        if model == "All six":
            e = one(tables["scope_checks"], scope="main", weighting="person")
            rows = effects[effects.weighting == "person"]
            ratio = number(rows.ratio.min()) + "--" + number(rows.ratio.max())
            count = f"{e.resize_larger_cells}/{e.total_cells}"
            result["format_effects"] += "\\midrule\n"
            result["majority_shifts"] += "\\midrule\n"
        else:
            e = one(effects, model=model, weighting="person")
            ratio, count = number(e.ratio), f"{e.resize_larger_cells}/{e.n_topics}"
        result["format_effects"] += row([model, number(e.resize_rms_pp), number(e.regroup_rms_pp),
            estimate(e.difference_pp, e.ci_low, e.ci_high), ratio, count,
            estimate(m.observed, m.ci_low, m.ci_high), number(m.regroup_observed), number(m["size"]), number(m.network)])
        result["majority_shifts"] += row([model, *[number(m[k]) for k in ("observed", "sum", "mean", "size", "stance", "network")]])
    models = tables["model_revisions"].copy()
    models["order"] = ~models.model.isin(MODELS)
    for m in models.sort_values(["order", "model"]).itertuples():
        result["models"] += row([m.model, r"\texttt{" + m.repository + "}",
                                  r"\texttt{" + m.revision[:10] + "}", m.role])
    descriptions = [("main", "Main: 6 models, 14 topics"), ("additional_topics", "6 models, 6 further topics"),
                    ("additional_models", "5 other models, 6 further topics"), ("answer_likelihood", "6 models, 2 topics, R1")]
    for scope, label in descriptions:
        for weighting in ("person", "incidence"):
            e = one(tables["scope_checks"], scope=scope, weighting=weighting)
            result["scope_checks"] += row([label if weighting == "person" else "", weighting,
                estimate(e.difference_pp, e.ci_low, e.ci_high), f"{e.resize_larger_cells}/{e.total_cells}",
                f"{e.positive_models}/{e.n_models}"])
    for intervention in ("resize", "regroup"):
        if intervention == "regroup":
            result["predictor_scores"] += "\\midrule\n"
        result["predictor_scores"] += r"\multicolumn{6}{l}{\textit{" + intervention.capitalize() + r" $-$ base}} \\" + "\n"
        for model in MODELS:
            values = [model]
            for method in ("sum", "mean", "size", "stance", "network"):
                r = one(tables["predictor_scores"], model=model, intervention=intervention, method=method)
                values.append("0 (constr.)" if intervention == "regroup" and method == "network" else estimate(r.r2, r.ci_low, r.ci_high))
            result["predictor_scores"] += row(values)
    for model in MODELS:
        values = [model]
        for method, reference in (("size", "mean"), ("stance", "size")):
            r = one(tables["predictor_gains"], model=model, intervention="resize", method=method, reference=reference)
            values.append(estimate(r.r2_gain, r.ci_low, r.ci_high))
        result["weight_comparisons"] += row(values)
        w = one(tables["size_weights"], model=model)
        result["size_weights"] += row([model, number(w.two_peers, 3), number(w.four_peers, 3), number(w.total_influence_ratio)])
        a = one(tables["auxiliary_averaging"], model=model)
        result["auxiliary_averaging"] += row([model, number(a.observed_rms_pp), number(a.mean_meeting_rms_pp), number(a.mean_network_max_abs_pp)])
    for m in tables["micro_summary"].itertuples():
        result["micro_summary"] += row([m.model, number(m.six_opponents_logit),
            estimate(m.power_exponent, m.power_ci_low, m.power_ci_high),
            estimate(m.neutral_effect, m.neutral_ci_low, m.neutral_ci_high),
            estimate(m.fixed_share_effect, m.fixed_share_ci_low, m.fixed_share_ci_high)])
    return result


def write(tables, output, manuscript=None):
    directory = output / "tables"
    directory.mkdir(exist_ok=True)
    exported = render(tables)
    checks = []
    for name, content in exported.items():
        (directory / (name + ".tex")).write_text(content, encoding="utf-8")
        if manuscript is not None:
            source = Path(manuscript) / "tables" / FILES[name]
            normalize = lambda s: re.sub(r"\s+", "", s)
            if normalize(content) != normalize(source.read_text(encoding="utf-8")):
                raise AssertionError("Manuscript table mismatch: " + name)
            checks.append(name)
    return checks
