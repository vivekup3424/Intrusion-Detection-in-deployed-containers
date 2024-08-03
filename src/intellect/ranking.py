"""
Module with utilities function and generic approaches to perform feature ranking
with a provided classifier.
"""
from typing import Callable, Iterator

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

from .dataset import Dataset
from .model.base import BaseModel
from .scoring import compute_metric_percategory


def rank_metric_zero(
        model: BaseModel, data: Dataset, current_features: list[str] = None,
        metric: Callable = accuracy_score) -> dict[str, float]:
    """Function to compute feature importance by measuring the model accuracy deltas
    when using the feature or not.

    Args:
        model (BaseModel): model to be used for inference
        data (Dataset): dataset of interest
        current_features (list[str], optional): list of features to preserve, while zeroing the others.
            Defaults to None.
        metric (str, optional): evaluation metric. Defaults to "accuracy".

    Returns:
        dict[str, float]: dictionary with the score associated to each feature
    """
    if current_features is not None:
        data = data.filter_features(current_features)
    else:
        current_features = data.features
    scores = {}
    baseline = metric(data.y, model.predict(data.X))
    for x in current_features:
        tmp = data.X[x]
        data.X[x] = 0.
        scores[x] = baseline - metric(data.y, model.predict(data.X))
        data.X[x] = tmp
    return scores


def rank_metric_permutation_sklearn(
        model: BaseModel, data: Dataset, current_features: list[str] = None,
        metric: str = 'accuracy', **kwargs) -> dict[str, float]:
    """Feature to compute feature importance using sklearn permutation method with the provided metric

    Args:
        model (BaseModel): model to be used for inference
        data (Dataset): dataset of interest
        current_features (list[str], optional): list of features to preserve, while zeroing the others.
            Defaults to None.
        metric (str, optional): evaluation metric. Defaults to "accuracy".

    Returns:
        dict[str, float]: dictionary with the score associated to each feature
    """
    if current_features is not None:
        data = data.filter_features(current_features)
    else:
        current_features = data.features
    positions = [i for i, x in enumerate(data.X.columns.values) if x in current_features]
    importances = permutation_importance(
        model, data.X, data.y, scoring=metric, **kwargs).importances_mean
    return dict(zip(current_features, [importances[i] for i in positions]))


def rank_random_forest(
        model: BaseModel, data: Dataset, current_features: list[str] = None, estimators: int = 100, depth: int = 7,
        features: str = 'sqrt', criterion='gini') -> dict[str, float]:
    """Function to compute feature importance by using an external random forest classifier. In case the model
    provided is a RandomForestClassifier, use the model itself.

    Args:
        model (BaseModel): model to be used
        data (Dataset): dataset of interest
        current_features (list[str], optional): list of features to preserve, while zeroing the others.
            Defaults to None.
        estimators (int, optional): number of trees. Defaults to 100.
        depth (int, optional): depth of the forest. Defaults to 7.
        features (str, optional): features used in the forest. Defaults to "sqrt".
        criterion (str, optional): criterion for measuring the quality of a split. Defaults to "gini".

    Returns:
        dict[str, float]: dictionary with the score associated to each feature
    """
    if current_features is not None:
        data = data.filter_features(current_features)
    else:
        current_features = data.features
    positions = [i for i, x in enumerate(data.X.columns.values) if x in current_features]
    if isinstance(model, RandomForestClassifier):
        rf = model
    else:
        rf = RandomForestClassifier(n_estimators=estimators, max_depth=depth,
                                    max_features=features, criterion=criterion)
        rf.fit(data.X.values, data.y.values)

    return dict(zip(current_features, [rf.feature_importances_[i] for i in positions]))


def rank_principal_component_analysis(model: BaseModel, data: Dataset, current_features=None) -> dict[str, float]:
    """Function to compute feature importance by using Principal Component Analysis.

    Args:
        model (BaseModel): model to be used
        data (Dataset): dataset of interest
        current_features (list[str], optional): list of features to preserve, while zeroing the others.
            Defaults to None.

    Returns:
        dict[str, float]: dictionary with the score associated to each feature
    """
    if current_features is not None:
        data = data.filter_features(current_features)
    else:
        current_features = data.features
    positions = [i for i, x in enumerate(data.X.columns.values) if x in current_features]

    if isinstance(model, PCA):
        pca = model
    else:
        pca = PCA()
        pca.fit(data.X.values, data.y.values)

    out = np.array([pca.components_[i] * np.sqrt(pca.explained_variance_ratio_[i])
                    for i in range(len(pca.explained_variance_ratio_))])

    values = np.power(out, 2).sum(axis=1)

    return dict(zip(values[positions], current_features))


def sequential_backward_elimination(
        model: BaseModel, data: Dataset, rank_algorithm: Callable = None,
        fixed_rank: dict[str, float] = None, metric: Callable = accuracy_score,
        remove_zero_first=True, **kwargs) -> Iterator[tuple[dict[str, float], list[float], list[str]]]:
    """Function to perform Sequential Backward Elimination with the provided info.

    Args:
        model (BaseModel): the model to be used
        data (Dataset): the dataset of interest
        rank_algorithm (Callable): rank algorithm to be used. It can be either one of the above methods
            or one custom defined in the model directory. Provided in case at each step the rank should
            be computed. Default to None.
        fixed_rank (dict[str, float], optional): fixed rank to be used instead of computing the recursive
            one (rank_algorithm). Defaults to None.
        metric (Callable, optional): evaluation metric. Defaults to accuracy_score.

    Raises:
        ValueError: when at least one among fixed_rank and rank_algorithm is not specified

    Returns:
        Iterator[tuple[dict[str, float], list[float], list[str]]]: tuple with ranks, list of scores,
            and list of removed features at each step.
    """
    if fixed_rank is None and rank_algorithm is None:
        raise ValueError('One among fixed_rank and rank_algorithm must be different than None')

    data = data.clone()
    current_features = list(fixed_rank.keys()) if fixed_rank else data.X.columns.values.tolist()
    while current_features:
        score = compute_metric_percategory(data.y.values, model.predict(data.X), data._y, scorer=metric)

        if fixed_rank is None:
            scores = rank_algorithm(
                model, data, current_features=current_features, **kwargs)
        else:
            scores = {k: v for k, v in fixed_rank.items() if k in current_features}

        zero_keys = None
        if remove_zero_first:
            zero_keys = [k for k, v in scores.items() if v == 0.]

        if zero_keys:
            worst_feature = zero_keys[-1]
        else:
            worst_feature, _ = min(scores.items(), key=lambda x: x[1])

        current_features.remove(worst_feature)
        data.X[worst_feature] = 0.
        yield score, worst_feature, scores

def subset_search(
        model: BaseModel, ds: Dataset, ratio: float, attempts: int,
        rank: dict[str, float] = None, metric: Callable = accuracy_score,
        stuck_guard: int = 1000, with_score=True) -> Iterator[tuple[list[str], float]]:
    """Function to perform only subset search given a classifier.

    Args:
        model (BaseModel): the model to be used for the subset evaluation.
        ds (Dataset): the dataset containing the data to be used.
        ratio (float): percentage of the features to preserve.
        attempts (int): number of possible combinations to explore
        rank (dict[str, float], optional): previously computed rank of
            all the available features in the dataset. If not None, then
            a weighted random search of the subset is performed. Defaults to None.
        metric (Callable, optional): evaluation metric to be used. Defaults to accuracy_score.
        baseline (float, optional): baseline value to which refer during the computation
            of the performance drop. Defaults to None.
        stuck_guard (int, optional): maximum attempts to randomly pick a new
            combination of features to avoid infinite loops. Defaults to 1000.

    Raises:
        ValueError: when unable to pick a new unexplored combination of features.

    Yields:
       Iterator[tuple[list[str], float, bool]]: for each explored attempt, the list of active features
                    in the subset, the metric score achieved and whether is accepted or
                    not with respect to the performance drop ratio value provided.
    """
    weights = None
    if rank is not None:
        weights = np.array([rank[k] for k in ds.features])
        weights += np.abs(np.min(weights))
        weights += np.min(weights[np.nonzero(weights)])/2
        weights = weights/weights.sum()

    explored_set_names = {}
    for _ in range(attempts):
        for i in range(stuck_guard):
            choices = tuple(np.random.choice(ds.features, round(ratio*ds.n_features), replace=False, p=weights))
            if choices not in explored_set_names:
                break
        if i == stuck_guard:
            raise ValueError(f'Unable to find a subset within the number {stuck_guard=}')
        explored_set_names[choices] = True
        tmp = ds.filter_features(choices, default=0)
        score = None
        if with_score:
            score = compute_metric_percategory(tmp.y, model.predict(tmp.X), tmp._y, scorer=metric)
        yield choices, score


def prune_search(
        model: BaseModel, ds: Dataset, prune_algorithm: Callable, prune_ratios: list[float], *args,
        metric: Callable = accuracy_score, with_score=True, **kwargs) -> Iterator[tuple[float, float]]:
    """Function to perform only the pruning of a given classifier.

    Args:
        model (BaseModel): the model to be used for the subset evaluation.
        ds (Dataset): the dataset containing the data to be used.
        prune_algorithm (Callable): pruning algorithm to be used.
        prune_ratios (list[float]): list of pruning ratios to try.
        metric (Callable, optional): evaluation metric to be used. Defaults to accuracy_score.
        baseline (float, optional): baseline value to which refer during the computation
            of the performance drop. Defaults to None.
        baseline (float, optional): _description_. Defaults to None.

    Yields:
        Iterator[tuple[float, float, bool]]: for each explored attempt, the pruning ratio,
                    the score associated to the pruned model and whether it is accepted
                    or not considering the provided performance drop ratio.
    """
    if isinstance(prune_ratios, (float)):
        prune_ratios = [prune_ratios]

    for ratio in prune_ratios:
        pruned = prune_algorithm(model, ratio, *args, **kwargs)
        score = None
        if with_score:
            score = compute_metric_percategory(ds.y, pruned.predict(ds.X), ds._y, scorer=metric)
        yield ratio, score

def prune_and_subset_search(
        model: BaseModel, prune_algorithm: Callable, ds: Dataset, prune_ratios: list[float],
        subset_ratio: float, subset_attempts: int, *args, rank: dict[str, float] = None,
        metric: Callable = accuracy_score, ** kwargs) -> Iterator[tuple[float, dict[tuple[str], float]]]:
    """Function to perform jointly (a) model pruning and then (b) the feature subset search.

    Args:
        model (BaseModel): model to be pruned using during the process.
        prune_algorithm (Callable): the pruning algorithm.
        ds (Dataset): the data to be used.
        prune_ratios (list[float]): list of pruning ratios to explore.
        subset_ratio (float): percentage of active feature in the subset to look for.
        subset_attempts (int): number of explored subsets.
        rank (dict[str, float], optional): previously computed rank of
            all the available features in the dataset. If not None, then
            a weighted random search of the subset is performed. Defaults to None.
        metric (Callable, optional): the evaluation metric. Defaults to accuracy_score.

    Yields:
        Iterator[tuple[float, dict[tuple[str], float]]]: for each value, returns the prune ratio,
            and a dictionary with the list of active features as a key and the obtained
            final accuracy as value.
    """

    if isinstance(prune_ratios, (float)):
        prune_ratios = [prune_ratios]

    for ratio in prune_ratios:
        pruned = prune_algorithm(model, ratio, *args, **kwargs)
        for x, v in subset_search(
                pruned, ds, subset_ratio, subset_attempts, metric=metric, rank=rank):
            yield ratio, x, v

def subset_search_and_prune(
        model: BaseModel, prune_algorithm: Callable, ds: Dataset, prune_ratios: list[float], subset_ratio: float,
        subset_attempts: int, *args, rank: dict[str, float] = None, metric: Callable = accuracy_score,
        **kwargs) -> Iterator[tuple[list[str], float, float]]:
    """Function to perform jointly (a) the feature subset search and then (b) the pruning of the classifier.

    Args:
        model (BaseModel): model to be pruned using during the process.
        prune_algorithm (Callable): the pruning algorithm.
        ds (Dataset): the data to be used.
        prune_ratios (list[float]): list of pruning ratios to explore.
        subset_ratio (float): percentage of active feature in the subset to look for.
        subset_attempts (int): number of explored subsets.
        rank (dict[str, float], optional): previously computed rank of
            all the available features in the dataset. If not None, then
            a weighted random search of the subset is performed. Defaults to None.
        metric (Callable, optional): the evaluation metric. Defaults to accuracy_score.

    Yields:
        Iterator[tuple[list[str], float, float]]: list with active features for the subset, the metric score,
            and the pruning ratio accepted. Multiple ratios are provided in different return values.
    """

    if isinstance(prune_ratios, float):
        prune_ratios = [prune_ratios]

    for k, _ in subset_search(
            model, ds, subset_ratio, subset_attempts, rank=rank):
        tmp = ds.filter_features(k, default=0)
        for ratio in prune_ratios:
            pruned = prune_algorithm(model, ratio, *args, **kwargs)
            score = compute_metric_percategory(tmp.y, pruned.predict(tmp.X), tmp._y, scorer=metric)
            yield k, score, ratio
