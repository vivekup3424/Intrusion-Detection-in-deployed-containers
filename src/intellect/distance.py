"""
Module with utility function concerning the distance between models and vectors of data.
"""
from itertools import combinations

import numpy as np
from river.base import DriftDetector
from river.drift import ADWIN
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

from .dataset import Dataset


def vector_to_dict_probs(x: np.ndarray) -> dict[float, float] | list[dict[float, float]]:
    """Function to transform vector of values into distribution probability.

    Args:
        x (np.ndarray): the input array to convert.

    Raises:
        NotImplementedError: if the array has more than 2 dimensions.

    Returns:
        dict[float, float] | list[dict[float, float]]: dictionary of value-percentage
            of occurrence for each value in the array.
    """
    if x.ndim == 1:
        a, b = np.unique(x, return_counts=True)
        b = b / np.sum(b)
        return dict(zip(a, b))

    if x.ndim == 2:
        ret = []
        for j in range(x.shape[1]):
            a, b = np.unique(x[:, j], return_counts=True)
            b = b / np.sum(b)
            ret.append(dict(zip(a, b)))
        return ret

    raise NotImplementedError()

def distributions_to_probabilities(x: np.ndarray, y: np.ndarray, only_common=False) -> tuple[list[float],
                                                                                             np.ndarray, np.ndarray]:
    """Function to convert two vectors in probability distributions of their values.

    Args:
        x (np.ndarray): one vector.
        y (np.ndarray): the other vector.
        only_common (bool, optional): True if only the overlapping values should
            be accounted during the generation of probability arrays. Defaults to False.

    Returns:
        tuple[list[float], np.ndarray, np.ndarray]: list of value points and the two array of probabilities.
    """
    o1, o2 = vector_to_dict_probs(x), vector_to_dict_probs(y)
    if only_common:
        points = set(o1.keys()).intersection(set(o2.keys()))
    else:
        points = set(o1.keys()).union(set(o2.keys()))
    r1, r2 = [], []
    for k in points:
        r1.append(o1.get(k, 0))
        r2.append(o2.get(k, 0))
    r1, r2 = np.array(r1), np.array(r2)
    if only_common:
        r1, r2 = r1 / np.sum(r1), r2/np.sum(r2)
    return points, r1, r2


def distance_between_categories(ds: Dataset, only_common=False) -> dict[tuple[str, str], dict[str, float]]:
    """Function to measure the distance between all the categories in a dataset. It leverages
    the Jensen Shannon distance computed on each feature column for each pair of categories in the dataset.

    Args:
        ds (Dataset): the dataset of interest.
        only_common (bool, optional): whether to consider only common values in the
            feature distributions when converting values to probabilities. Defaults to False.

    Returns:
        dict[tuple[str, str], dict[str, float]]: for each pair of categories, a dictionary of the
            Jensen Shannon measure of each feature column.
    """
    ret = {}
    for x, y in tqdm(combinations(ds.categories, 2)):
        c, d = ds.filter_categories([x]), ds.filter_categories([y])
        out = {}
        for col in c.X.columns.values:
            _, v1, v2 = distributions_to_probabilities(
                c.X[col].to_numpy(),
                d.X[col].to_numpy(),
                only_common=only_common)
            out[col] = jensenshannon(v1, v2)
        ret[(x, y)] = out
    return ret

def get_data_drifts(ds_origin: Dataset, ds_target: Dataset, detector: DriftDetector = ADWIN()) -> dict[str, list[int]]:
    """Function to compute data drifts between each column (feature) of two different datasets.

    Args:
        ds_origin (Dataset): one dataset
        ds_target (Dataset): the other dataset
        detector (DriftDetector, optional): detection method to use. Defaults to ADWIN().

    Returns:
        dict[str, list[int]]: dictionary with list of drift points for each feature name
    """
    feature_drifts = {}
    for col in ds_origin.X.columns:
        detector = detector.clone()
        for row in ds_origin.X[col]:
            detector.update(row)
        for i, row in enumerate(ds_target.X[col]):
            detector.update(row)
            if detector.drift_detected:
                if col not in feature_drifts:
                    feature_drifts[col] = [i]
                else:
                    feature_drifts[col].append(i)
    return feature_drifts
