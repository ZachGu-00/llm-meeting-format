"""Topic-cluster inference with a fixed set of models."""
from functools import lru_cache
import numpy as np


SEED = 20261105
DRAWS = 20000


@lru_cache(maxsize=None)
def bootstrap_weights(n_topics):
    return np.random.default_rng(SEED).multinomial(
        n_topics, np.full(n_topics, 1 / n_topics), size=DRAWS) / n_topics


@lru_cache(maxsize=None)
def bootstrap_indices(n_topics):
    # Preserve the indexed draws used for the manuscript's direction estimates.
    return np.random.default_rng(SEED).integers(0, n_topics, size=(DRAWS, n_topics))


def interval(values):
    return np.quantile(values, [0.025, 0.975]).tolist()


def rms_comparison(resize_second_moments, regroup_second_moments):
    resize = np.asarray(resize_second_moments)
    regroup = np.asarray(regroup_second_moments)
    weights = bootstrap_weights(len(resize))
    difference = 100 * (np.sqrt(resize.mean()) - np.sqrt(regroup.mean()))
    low, high = interval(100 * (np.sqrt(weights @ resize) - np.sqrt(weights @ regroup)))
    return {"resize_rms_pp": 100 * np.sqrt(resize.mean()),
            "regroup_rms_pp": 100 * np.sqrt(regroup.mean()),
            "difference_pp": difference, "ci_low": low, "ci_high": high,
            "ratio": np.sqrt(resize.mean() / regroup.mean()),
            "resize_larger_cells": int((resize > regroup).sum()), "n_topics": len(resize)}


def contrast_score(observed_second_moments, prediction_errors):
    observed = np.asarray(observed_second_moments)
    error = np.asarray(prediction_errors)
    draws = bootstrap_weights(len(observed))
    low, high = interval(1 - (draws @ error) / (draws @ observed))
    return {"r2": 1 - error.mean() / observed.mean(), "ci_low": low, "ci_high": high}


def paired_gain(observed_second_moments, error, reference_error):
    observed = np.asarray(observed_second_moments)
    improvement = np.asarray(reference_error) - np.asarray(error)
    draws = bootstrap_weights(len(observed))
    low, high = interval((draws @ improvement) / (draws @ observed))
    return {"r2_gain": improvement.mean() / observed.mean(), "ci_low": low, "ci_high": high}
