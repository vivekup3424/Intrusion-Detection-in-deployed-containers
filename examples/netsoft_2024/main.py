import argparse
import importlib
import os
import time
import warnings
from dataclasses import dataclass

import pandas as pd
import river
import threadpoolctl
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from intellect.dataset import Dataset
from intellect.distance import get_data_drifts
from intellect.inspect import set_seed
from intellect.io import create_dir, dump, load
from intellect.model.ensembles import WrapRiverEnsemble
from intellect.model.sklearn.model import EnhancedMlpRegressor
from intellect.model.sklearn.pruning import globally_neurons_l1
from intellect.ranking import rank_random_forest


@dataclass
class Config:
    train_size: float
    validation_size: float
    train_epochs: int
    epochs_wo_improve: int

    hidden_units: int
    hidden_layers: int
    dropout_hidden: float

    batch_size: int
    pruning_amounts: list[float]
    seeds: list[int]

    benchmark_iter: int

    label: str
    dataset: str
    dataset_name: str
    detectors_to_test: list[str]


def datadrift(config: Config, output_dir: str, seed=42, verbose=False):
    train_ds, test_ds = _load_dataset(config.dataset, config.label, config.train_size, seed=seed)
    set_seed(seed)

    for i in range(
            len(config.detectors_to_test)) if not verbose else (
            pbar := tqdm(range(len(config.detectors_to_test)))):
        d = config.detectors_to_test[i]
        tmp = d.split('.')
        mod, d_name = '.'.join(tmp[:-1]), tmp[-1]
        d_instance = getattr(importlib.import_module(mod), d_name)()
        if verbose:
            pbar.set_description(f'Testing {d_instance.__class__.__name__}')
        x = get_data_drifts(train_ds, test_ds, detector=d_instance)
        dump(x, os.path.join(output_dir, f'{d_instance.__class__.__name__}.json'))


def _load_dataset(dataset: str, label_str: str, train_ratio: float, seed: int = 42):
    set_seed(seed)

    df: pd.DataFrame = load(dataset, index_col=None)
    label = df.pop(label_str)
    X_normalized = df.apply(
        lambda x: (x - x.min()) / (x.max() - x.min()))
    X_normalized[label_str] = label

    X_train_normalized, X_test_normalized = train_test_split(X_normalized, train_size=train_ratio, shuffle=False)
    return Dataset(data=X_train_normalized, label=label_str, label_type=label_str, shuffle=False), Dataset(
        data=X_test_normalized, label=label_str, label_type=label_str, shuffle=False)


def train_online_learning(
        config: Config, output_dir: str, prune_ratio: float = None, seed: int = 42, is_ensemble=False, verbose=False):
    train_ds, _ = _load_dataset(config.dataset, config.label, config.train_size, seed=seed)

    m_path = os.path.join(output_dir, 'ol_train')

    set_seed(seed)
    if prune_ratio:
        base_model = EnhancedMlpRegressor.load(m_path)
        base_model = globally_neurons_l1(base_model, prune_ratio)
        base_model = base_model.clone(init=True)
        m_path += f'_retrain_prune_{prune_ratio}'
    else:
        base_model = EnhancedMlpRegressor(
            train_ds.classes, hidden_layer_sizes=(config.hidden_units,) * config.hidden_layers,
            dropout=config.dropout_hidden, drift_detector=river.drift.ADWIN(),
            max_iter=config.train_epochs, validation_fraction=0.)

    if is_ensemble:
        model = WrapRiverEnsemble(river.ensemble.LeveragingBaggingClassifier,
                                  model=base_model, n_models=config.n_ensemble_models)
        m_path += '_ensemble'
    else:
        model = base_model

    ypred, ytrue, ylabels, drifts = model.continuous_learning(
        train_ds, validation_dataset=0., max_epochs=1, batch_size=1, epochs_wo_improve=1, shuffle=False, verbose=verbose)

    model.save(m_path)
    df2 = pd.DataFrame({'Labels': ytrue, 'Predictions': ypred, 'Type': ylabels})
    dump(df2, m_path + '_incremental_raw.h5')

    ypred_after = model.predict(train_ds.X)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df = pd.DataFrame(
            {'Accuracy': [accuracy_score(ytrue, ypred), accuracy_score(ytrue, ypred_after)],
             'Recall': [recall_score(ytrue, ypred), recall_score(ytrue, ypred_after)],
             'Precision': [precision_score(ytrue, ypred), precision_score(ytrue, ypred_after)],
             'F1': [f1_score(ytrue, ypred), f1_score(ytrue, ypred_after)],
             'Drifts': [str(drifts), '']},
            index=['During OL Train', 'After OL Train'])
    dump(df, m_path + '_incremental.csv')


def train_traditional(
        config: Config, output_dir: str, prune_ratio: float = None, seed: int = 42, verbose=False):
    train_ds, _ = _load_dataset(config.dataset, config.label, config.train_size, seed=seed)

    m_path = os.path.join(output_dir, 'tl_train')

    set_seed(seed)
    if prune_ratio:
        model = EnhancedMlpRegressor.load(m_path)
        model = globally_neurons_l1(model, prune_ratio)
        model = model.clone(init=True)
        m_path += f'_retrain_prune_{prune_ratio}'
    else:
        model = EnhancedMlpRegressor(
            train_ds.classes, hidden_layer_sizes=(config.hidden_units,) * config.hidden_layers,
            dropout=config.dropout_hidden, drift_detector=river.drift.ADWIN(),
            max_iter=config.train_epochs, validation_fraction=0.)

    hs, _ = model.fit(
        train_ds, validation_dataset=config.validation_size, batch_size=config.batch_size, max_epochs=config.train_epochs,
        epochs_wo_improve=config.epochs_wo_improve, shuffle=True, verbose=verbose)
    model.save(m_path)
    dump(hs, m_path + '_history.csv')

    ypred_after = model.predict(train_ds.X)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df = pd.DataFrame(
            {'Accuracy': accuracy_score(train_ds.y, ypred_after),
             'Recall': recall_score(train_ds.y, ypred_after),
             'Precision': precision_score(train_ds.y, ypred_after),
             'F1': f1_score(train_ds.y, ypred_after),
             'Drifts': ''},
            index=['TrainSet after TL Train'])
    dump(df, m_path + '.csv')


# test a model with and without online learning
def test_with_without_and_after_ol(
        config: Config, output_dir: str, seed=42, prune_ratio=None, is_ol=False, is_ensemble=False, is_retrain=False,
        is_zeroed=False, verbose=False):
    _, test_ds = _load_dataset(config.dataset, config.label, config.train_size, seed=seed)
    set_seed(seed)

    m_path = os.path.join(output_dir, ('ol' if is_ol else 'tl') + '_train')

    if is_ensemble:
        if prune_ratio:
            m_path += f'_prune_{prune_ratio}_ensemble'
        model = WrapRiverEnsemble.load(m_path)
    else:
        if is_retrain:
            m_path += f'_retrain_prune_{prune_ratio}'
            model = EnhancedMlpRegressor.load(m_path)
        else:
            model = EnhancedMlpRegressor.load(m_path)
            if prune_ratio:
                model = globally_neurons_l1(model, prune_ratio)
                m_path += f'_prune_{prune_ratio}'

    if is_zeroed:
        rank = load(os.path.join(output_dir, 'feature_importance_random_forest.json'))
        keep_names = [x[0] for x in sorted(rank.items(), key=lambda x: x[1], reverse=True)[:int(len(rank)/2)]]
        test_ds = test_ds.filter_features(keep_names)
        m_path += f'_feature_zeroed'

    yp_normal = model.predict(test_ds.X)

    ypred, ytrue, ylabels, drifts = model.continuous_learning(
        test_ds, validation_dataset=0., max_epochs=1, batch_size=1, epochs_wo_improve=1, shuffle=False, verbose=verbose)
    model.save(f'{m_path}_test')
    df2 = pd.DataFrame({'Labels': ytrue, 'Predictions Without OL': yp_normal,
                       'Predictions With OL': ypred, 'Type': ylabels})
    dump(df2, f'{m_path}_test_incremental_raw.h5')

    yp_normal_after = model.predict(test_ds.X)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df = pd.DataFrame(
            {'Accuracy': [accuracy_score(ytrue, yp_normal), accuracy_score(ytrue, ypred), accuracy_score(ytrue, yp_normal_after)],
             'Recall': [recall_score(ytrue, yp_normal), recall_score(ytrue, ypred), recall_score(ytrue, yp_normal_after)],
             'Precision': [precision_score(ytrue, yp_normal), precision_score(ytrue, ypred), precision_score(ytrue, yp_normal_after)],
             'F1': [f1_score(ytrue, yp_normal), f1_score(ytrue, ypred), f1_score(ytrue, yp_normal_after)],
             'Drifts': ['', str(drifts), '']},
            index=['Without OL', 'During OL', 'After OL'])
    dump(df, f'{m_path}_test_incremental.csv')

def train_evaluation(config: Config, output_dir: str, seed=42, prune_ratio=None, verbose=False):
    train_ds, _ = _load_dataset(config.dataset, config.label, config.train_size, seed=seed)
    m_path = os.path.join(output_dir, 'evaluation_full' if prune_ratio is None else f'evaluation_pruned_{prune_ratio}')

    new_hidu = int((1-prune_ratio) * config.hidden_units) if prune_ratio else config.hidden_units

    set_seed(default=seed)
    m = EnhancedMlpRegressor(
        train_ds.classes, hidden_layer_sizes=(new_hidu,) * config.hidden_layers,
        dropout=config.dropout_hidden, drift_detector=river.drift.ADWIN(),
        max_iter=config.train_epochs, validation_fraction=0.)

    hs, _ = m.fit(
        train_ds, validation_dataset=config.validation_size, batch_size=config.batch_size,
        max_epochs=config.train_epochs, epochs_wo_improve=config.epochs_wo_improve, shuffle=True, verbose=verbose)

    m.save(m_path)
    dump(hs, f'{m_path}_history.json')

def test_evaluation(config: Config, output_dir, seed=42, prune_ratio=None, verbose=False):
    _, test_ds = _load_dataset(config.dataset, config.label, config.train_size, seed=seed)
    m_path = os.path.join(output_dir, 'evaluation_full' if prune_ratio is None else f'evaluation_pruned_{prune_ratio}')
    set_seed(default=seed)
    m = EnhancedMlpRegressor.load(m_path)
    res = []
    # warmup
    for _ in range(20):
        m.predict(test_ds.X)
    for _ in range(config.benchmark_iter) if not verbose else tqdm(range(config.benchmark_iter)):
        start = time.time_ns()
        m.predict(test_ds.X)
        end = time.time_ns()
        res.append(end-start)
    dump(res, m_path + '.json')

def feature_importance(config: Config, output_dir, seed=42, verbose=False):
    _, test_ds = _load_dataset(config.dataset, config.label, config.train_size, seed=seed)
    rank = rank_random_forest(None, test_ds)
    dump(rank, os.path.join(output_dir, 'feature_importance_random_forest.json'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Path to configuration file', required=True, type=str)
    parser.add_argument('-s', '--seed', help='Random seed', type=int, default=42)
    parser.add_argument('-p', '--parallel', type=int, default=1,
                        help='Parallel Threads to be used in the underlying libraries')
    parser.add_argument('-v', '--verbose', help='Set Verbosity', action='store_true')
    parser.add_argument('-r', '--ratio', help='Pruning ratio', type=float, default=None)
    parser.add_argument('-a', '--action', help='Action to perform', required=True,
                        type=str, choices=['datadrift',
                                           'train_tl', 'prune_tl', 'test_tl',
                                           'train_tl_pruned', 'test_tl_pruned',
                                           'train_ol', 'prune_ol', 'test_ol',
                                           'train_ol_pruned', 'test_ol_pruned',
                                           'train_ol_ensemble', 'test_ol_ensemble',
                                           'train_ol_ensemble_pruned', 'test_ol_ensemble_pruned',
                                           'test_ol_zeroed', 'prune_ol_zeroed',
                                           'evaluate_train', 'evaluate_test',
                                           'feature_importance'])

    # This is an example parallelized pipeline that applies also to the *ol* one
    # train_tl -> test_tl
    #         |-> prune_tl
    #         |-> train_tl_pruned -> test_tl_pruned

    args = parser.parse_args()
    config = Config(**load(args.config))
    seed = args.seed
    action = args.action
    verbose = args.verbose
    parallel = args.parallel
    prune_ratio = args.ratio

    threadpoolctl.threadpool_limits(limits=parallel)

    output_dir = os.path.join(os.path.dirname(args.config), f'{config.dataset_name}_output', f'seed_{seed}_output')

    if not os.path.isdir(output_dir):
        create_dir(output_dir)

    if action in ('train_ol', 'train_ol_pruned', 'train_ol_ensemble', 'train_ol_ensemble_pruned'):
        train_online_learning(
            config, output_dir, prune_ratio=prune_ratio,
            is_ensemble='ensemble' in action, seed=seed, verbose=verbose)
    elif action in ('train_tl', 'train_tl_pruned'):
        train_traditional(config, output_dir, prune_ratio=prune_ratio
                          if 'pruned' in action else None, seed=seed, verbose=verbose)
    elif action in ('test_tl', 'test_ol', 'test_ol_ensemble',
                    'test_ol_zeroed', 'test_ol_pruned_zeroed',
                    'test_tl_pruned', 'test_ol_pruned', 'test_ol_ensemble_pruned'):
        test_with_without_and_after_ol(
            config, output_dir, seed=seed, prune_ratio=prune_ratio if 'pruned' in action else None,
            is_ensemble='ensemble' in action, is_ol='ol' in action, is_retrain='pruned' in action, verbose=verbose,
            is_zeroed='zeroed' in action)
    elif action in ('prune_ol', 'prune_tl', 'prune_ol_zeroed'):
        test_with_without_and_after_ol(config, output_dir, seed=seed,
                                       prune_ratio=prune_ratio, is_ol='ol' in action, verbose=verbose,
                                       is_zeroed='zeroed' in action)
    elif action == 'datadrift':
        datadrift(config, output_dir, seed=seed, verbose=verbose)
    elif action == 'evaluate_train':
        train_evaluation(config, output_dir, seed=seed, prune_ratio=prune_ratio, verbose=verbose)
    elif action == 'evaluate_test':
        test_evaluation(config, output_dir, seed=seed, prune_ratio=prune_ratio, verbose=verbose)
    elif action == 'feature_importance':
        feature_importance(config, output_dir, seed=seed, verbose=verbose)
    elif 'prova' in action:
        print(f'{action} - {seed} {args.config}')


if __name__ == '__main__':
    main()
