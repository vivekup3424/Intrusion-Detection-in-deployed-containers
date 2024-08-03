"""
Module containing utility functions for scoring.
"""
from __future__ import annotations

from numbers import Number
from typing import Callable

import numpy as np
import pandas as pd
from river import metrics
from sklearn.metrics import accuracy_score


def safe_division(dividend: Number, divisor: Number) -> float:
    """Perform safe division between two numbers, returning
    0 in case of infinite division.

    Args:
        dividend (Number): the nominator
        divisor (Number): the denominator

    Returns:
        Number: the result of the division
    """
    if divisor == 0:
        return 0
    return dividend / divisor


def compute_metric_percategory(
        ytrue: list, ypred: list, labels: pd.Series,
        scorer: Callable = accuracy_score) -> dict[str, float]:
    """Function to compute the metric foreach category available.

    Args:
        ytrue (list): array containing the true labels
        ypred (list): array containing the predicted labels
        labels (pd.Series): categories assigned to each entry
        scorer (Callable, optional): evaluation metric. Defaults to accuracy_score.

    Returns:
        dict[str, float]: dictionary with the score for each category.
    """
    ret = {}
    ytrue = _ensure_type(ytrue)
    ypred = _ensure_type(ypred)
    labels = _ensure_type(labels)

    ret['Global'] = scorer(ytrue, ypred)
    for k in np.unique(labels):
        indexes = np.where(labels == k)[0]
        ret[k] = scorer(ytrue[indexes], ypred[indexes])
    return ret


def compute_metric_incremental(
        ytrue: list, ypred: list, metric: metrics.base.Metric | str = 'Accuracy') -> list[float]:
    """Function to compute the metric incrementally point after point.

    Args:
        ytrue (list): the list of true labels
        ypred (list): the list of predicted labels
        metric (metrics.base | str, optional): the incremental metric to be used. Defaults to metrics.Accuracy().

    Returns:
        list[float]: list of the metric computed after each point.
    """
    if isinstance(metric, str):
        metric = getattr(metrics, metric)()
    ytrue = _ensure_type(ytrue)
    ypred = _ensure_type(ypred)

    metric = metric.clone()
    m = []
    for i, v in enumerate(ytrue):
        metric.update(v, ypred[i])
        m.append(metric.get())
    return m

def _ensure_type(x):
    if hasattr(x, 'numpy'):
        return x.numpy()
    if hasattr(x, 'to_numpy'):
        return x.to_numpy()
    return x
