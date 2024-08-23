# https://towardsdatascience.com/feature-subset-selection-6de1f05822b0
import functools
import sys
from typing import List

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import \
    TabularNeuralNetTorchModel
from multiprocess.pool import ThreadPool
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from .model import test_model_on_subset


def get_all_ranking_algorithms():
    return ('autogluon_sbe', 'autogluon_sfs',
            'custom_sbe', 'custom_sfs',
            'pca_sbe', 'pca_sfs',
            'rf_sbe', 'rf_sfs')


def get_ranking_algorithm(name):
    return getattr(sys.modules[__name__], name)


def autogluon_sbe(
        X: pd.DataFrame, y: pd.Series, current_features: List[str],
        predictor: TabularPredictor = None, model_name=None,
        time_limit=None, only_last=False, num_shuffle_sets=10, **_):

    if 'Label' not in X:
        X['Label'] = y
    scores = predictor.feature_importance(
        X, features=current_features[-1:] if only_last else current_features, model=model_name,
        time_limit=time_limit, include_confidence_band=True, num_shuffle_sets=num_shuffle_sets)
    scores.index.name = 'ID'
    return scores['importance'].to_dict()


def autogluon_sfs(
        X: pd.DataFrame, y: pd.Series, current_features: List[str],
        predictor: TabularPredictor = None, model_name=None,
        time_limit=None, cpus=None, **_):
    remaining = [x for x in X.columns.values if x not in current_features]

    scores = {}
    with ThreadPool(processes=cpus) as pool, tqdm(total=len(remaining)) as pbar:
        for x in pool.imap(
                functools.partial(
                    autogluon_sbe, X, y, predictor=predictor, model_name=model_name,
                    time_limit=time_limit, only_last=True),
                [current_features + [f] for f in remaining]):
            scores.update(x)
            pbar.update()
    return scores


def custom_sbe(
        X: pd.DataFrame, y: pd.Series, current_features: List[str],
        model: TabularNeuralNetTorchModel = None,
        predictor: TabularPredictor = None,
        cpus=None, eval_metric='accuracy', **_):
    baseline = test_model_on_subset(predictor, X, y, model, subset=current_features)[eval_metric]

    scores = {}
    with ThreadPool(processes=cpus) as pool, tqdm(total=len(current_features)) as pbar:
        for i, x in enumerate(pool.imap(
                functools.partial(test_model_on_subset, predictor, X, y, model),
                [[x for x in current_features if x != f] for f in current_features])):
            scores[current_features[i]] = baseline - x[eval_metric]
            pbar.update()
    return scores


def custom_sfs(
        X: pd.DataFrame, y: pd.Series, current_features: List[str],
        model: TabularNeuralNetTorchModel = None,
        predictor: TabularPredictor = None,
        cpus=None, eval_metric='accuracy', **_):
    remaining = [x for x in X.columns.values if x not in current_features]

    baseline = test_model_on_subset(predictor, X, y, model, subset=current_features)[eval_metric]
    scores = {}
    with ThreadPool(processes=cpus) as pool, tqdm(total=len(remaining)) as pbar:
        for i, x in enumerate(pool.imap(
                functools.partial(test_model_on_subset, predictor, X, y, model),
                [current_features + [f] for f in remaining])):
            scores[remaining[i]] = x[eval_metric] - baseline
            pbar.update()
    return scores


def pca_sbe(X: pd.DataFrame, y: pd.Series, current_features: List[str], only_last=False, **_):
    positions = [i for i, x in enumerate(X.columns.values) if x in current_features]

    pca = PCA()
    pca.fit(X, y)

    out = np.array([pca.components_[i] * np.sqrt(pca.explained_variance_ratio_[i])
                    for i in range(len(pca.explained_variance_ratio_))])

    values = np.power(out, 2).sum(axis=1)

    if only_last:
        return {current_features[-1]: values[positions][-1]}
    return {k: v for k, v in zip(values[positions], current_features)}


def pca_sfs(
        X: pd.DataFrame, y: pd.Series, current_features: List[str], cpus=None, **_):
    remaining = [x for x in X.columns.values if x not in current_features]

    scores = {}
    with ThreadPool(processes=cpus) as pool, tqdm(total=len(remaining)) as pbar:
        for x in pool.imap(
                functools.partial(
                    pca_sbe, X, y, only_last=True),
                [current_features + [f] for f in remaining]):
            scores.update(x)
            pbar.update()
    return scores


def rf_sbe(
        X: pd.DataFrame, y: pd.Series, current_features: List[str],
        only_last=False, cpus=None, estimators=100, depth=7, features='sqrt', criterion='gini', **_):
    positions = [i for i, x in enumerate(X.columns.values) if x in current_features]

    rf = RandomForestClassifier(n_estimators=estimators, max_depth=depth,
                                max_features=features, criterion=criterion, n_jobs=cpus)
    rf.fit(X, y)

    if only_last:
        return {current_features[-1]: rf.feature_importances_[-1]}
    return dict(zip(current_features, [rf.feature_importances_[i] for i in positions]))


def rf_sfs(
        X: pd.DataFrame, y: pd.Series, current_features: List[str],
        cpus=None, **_):
    remaining = [x for x in X.columns.values if x not in current_features]

    scores = {}
    with ThreadPool(processes=cpus) as pool, tqdm(total=len(remaining)) as pbar:
        for x in pool.imap(functools.partial(rf_sbe, X, y, only_last=True),
                           [current_features + [f] for f in remaining]):
            scores.update(x)
            pbar.update()
    return scores
