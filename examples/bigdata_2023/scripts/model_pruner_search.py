"""
Script to create a new network given a sample one, but pruning
few layers/neurons.
https://towardsdatascience.com/pruning-neural-networks-1bb3ab5791f9
"""
import ast
import functools
import os
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from typing import List

import click
import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer_names
from tqdm import tqdm
from utils.common import (create_dir, get_default_cpu_number, get_logger,
                          load_model_from_predictor, set_seed)
from utils.model import (get_prunable, network_layers_sparsity,
                         test_model_on_subset)
from utils.pruning import get_all_pruning_algorithms, get_pruning_algorithm

logger = get_logger(f"{os.path.basename(__file__).replace('.py', '')}")


def prune_model(original_model, algorithm, X, args):
    if isinstance(args, (list, tuple)):
        amount, subset_features = args[0], ast.literal_eval(args[1])
    else:
        amount, subset_features = args, None
    copied = deepcopy(original_model)
    layers = get_prunable(copied.model)
    copied, layers = algorithm(copied, layers, amount, X=X, subset_features=subset_features)
    return copied, layers


def test_prune_amount(predictor, original_model, algorithm, X, y, args):
    global logger
    if isinstance(args, (list, tuple)):
        amount, subset_features = args[0], ast.literal_eval(args[1])
    else:
        amount, subset_features = args, None
    copied, layers = prune_model(original_model, algorithm, amount, X, subset_features)
    glob_sparsity, local_sparsity = network_layers_sparsity(layers)
    test_results = test_model_on_subset(predictor, X, y, copied)
    return amount, glob_sparsity, local_sparsity, test_results


def test(directory: str, algorithm: str, top_subsets: int, subsets_degradation: float, models_degradation: float,
         cpus: int, metric: str):
    global logger
    set_seed()

    logger.info('Loading Predictor')
    logger.info('Loading Model from predictor')
    original_model, predictor, _ = load_model_from_predictor(os.path.join(directory, os.pardir, os.pardir, os.pardir))

    logger.info('Loading Data')
    X = pd.read_csv(
        os.path.realpath(os.path.join(directory, os.pardir, os.pardir, os.pardir,
                                      os.pardir, os.pardir, os.pardir, os.pardir, 'finetune.csv')),
        index_col='ID')
    y = X.pop('Label')

    logger.info('Loading Subsets')
    subsets_file = pd.read_csv(
        os.path.join(directory, 'subsets.csv'),
        index_col='ID')
    subset_leaderboard = pd.read_csv(
        os.path.join(directory, 'leaderboard.csv'),
        index_col='ID')

    if '_for_subset' in algorithm:
        prune_dir = os.path.join(directory, algorithm)
    else:
        prune_dir = os.path.join(directory, os.pardir, os.pardir, os.pardir, 'prune_search', algorithm)

    ss_size = float(os.path.basename(directory).replace('feature_subsets_', '').replace('s', ''))
    ss_size = round(ss_size * len(X.columns.values)) or 1
    baseline_rfe = pd.read_csv(
        os.path.join(directory, os.pardir, os.pardir, 'leaderboard.csv'),
        index_col='ID').loc[ss_size][metric]
    valid_subsets = subset_leaderboard[baseline_rfe - subset_leaderboard[metric] <= subsets_degradation]
    valid_subsets = valid_subsets.iloc[:top_subsets]
    valid_subsets = subsets_file.loc[valid_subsets.index.values]

    logger.info("Loading Models' stats")
    models_file = pd.read_csv(
        os.path.join(prune_dir, 'models_stats.csv'),
        index_col='ID')
    models_leaderboard = pd.read_csv(
        os.path.join(prune_dir, 'leaderboard.csv'),
        index_col='ID')
    valid_models = models_file

    baseline_general = pd.read_csv(
        os.path.join(directory, os.pardir, os.pardir, os.pardir, 'leaderboard.csv'),
        index_col='ID').loc['finetune'][metric]

    valid_models = models_leaderboard[baseline_general - models_leaderboard[metric] <= models_degradation].index.values
    valid_models = models_file.loc[valid_models]
    algorithm_func = get_pruning_algorithm(algorithm)

    if '_for_subset' in algorithm:
        combs = {k['amount']: [valid_subsets.loc[int(k['subset_ID'])]['features']] for k in valid_models}
        ids = [(i, int(valid_models.loc[i]['subset_ID'])) for i in valid_models.index.values]
        amounts = [k['amount'] for k in valid_models]
    else:
        combs = {amount: valid_subsets['features'] for amount in valid_models['amount']}
        ids = [(i, ii) for i in valid_models.index.values for ii in valid_subsets.index.values]
        amounts = [amount for amount in valid_models['amount']]

    models = {}
    with ThreadPool(processes=cpus) as pool, tqdm(total=len(ids)) as pbar:
        for i, (model, _) in enumerate(
                pool.imap(functools.partial(prune_model, original_model, algorithm_func, X), amounts)):
            models[amounts[i]] = model
            pbar.update()

    to_test = [(models[k], ast.literal_eval(j)) for k, v in combs.items() for j in v]

    results = pd.DataFrame()
    logger.info(f'Testing {len(ids)=} combinations of subset-model asyncronously')
    with ThreadPool(processes=cpus) as pool, tqdm(total=len(to_test)) as pbar:
        for i, res in enumerate(
                pool.imap(functools.partial(test_model_on_subset, predictor, X, y), to_test)):
            results = pd.concat((results, pd.DataFrame(
                {'model_ID': ids[i][0],
                 'subset_ID': ids[i][1],
                 **res},
                index=pd.Index([i], name='ID'))))
            pbar.update()
    results.sort_values(by=metric, ascending=False, inplace=True)
    results.to_csv(
        os.path.join(directory, f'leaderboard_{algorithm}.csv'))


def search(directory: str, algorithm: str, subsets: str, top_subsets: int, subsets_degradation: float,
           amount: List[float], attempts: int, cpus: int, metric: str):
    global logger
    set_seed()

    logger.info('Loading Model from predictor')
    original_model, predictor, _ = load_model_from_predictor(directory)

    logger.info('Loading Data')
    X = pd.read_csv(
        os.path.realpath(os.path.join(directory, os.pardir, os.pardir, os.pardir,
                                      os.pardir, 'finetune.csv')), index_col='ID')
    y = X.pop('Label')

    store_path = os.path.join(original_model.path, 'prune_search', algorithm)
    subsets_data: pd.DataFrame = None

    if subsets:
        if not algorithm.endswith('_for_subset'):
            raise ValueError('Used subset search when providing a wrong algorithm')
        store_path = os.path.join(subsets, algorithm)
        logger.info('Loading provided subsets')
        subsets_data = pd.read_csv(
            os.path.join(subsets, 'leaderboard.csv'),
            index_col='ID')
        ss_size = float(os.path.basename(subsets).replace('feature_subsets_', '').replace('s', ''))
        ss_size = round(ss_size * len(X.columns.values)) or 1
        baseline_rfe = pd.read_csv(
            os.path.join(subsets, os.pardir, os.pardir, 'leaderboard.csv'),
            index_col='ID').loc[ss_size][metric]
        valid_subsets = subsets[baseline_rfe - subsets_data[metric] <= subsets_degradation]
        valid_subsets = valid_subsets.index.values[:top_subsets]
        subsets_data = pd.read_csv(
            os.path.join(subsets, 'subsets.csv'),
            index_col='ID').loc[valid_subsets]

    create_dir(store_path, overwrite=False)
    algorithm_func = get_pruning_algorithm(algorithm)

    amount = sorted(amount, reverse=True)

    if algorithm.startswith('locally'):
        if 0. not in amount:
            amount = amount + [0.]
        if 1. not in amount:
            amount = [1.] + amount
        n_layers = len(get_prunable(original_model.model))
        all_combs = {}
        wanted = min(attempts, len(amount)**n_layers)
        cnt = 0
        logger.info(f'Computing combination {wanted=} {n_layers=}')
        probs = np.linspace(1, len(amount), len(amount))
        probs = probs / probs.sum()
        while len(all_combs) < wanted:
            news = tuple(np.random.choice(
                amount, n_layers, p=probs, replace=True).tolist())
            v = len(all_combs)
            all_combs.setdefault(news, 0)
            cnt = cnt + 1 if v == len(all_combs) else 0
            if cnt == 1000:
                break
        amount = list(all_combs.keys())
    elif algorithm.startswith('globally'):
        if 0. in amount:
            amount.remove(0.)
        if 1. in amount:
            amount.remove(1.)
        amount = sorted(np.random.choice(amount, min(
            len(amount), attempts), replace=False), reverse=True)

    if subsets:
        logger.info('Associating Prune amount to each subset')
        amount = amount[:min(attempts, round(attempts / len(subsets_data)))]
        prev_len = len(amount)
        amount = [[k, v] for k in amount for v in subsets_data['features'].to_list()]
        subsets_data = [i for _ in range(prev_len) for i in subsets_data.index.values]

    stats = pd.DataFrame(index=pd.Index([], name='ID'))
    results = pd.DataFrame(index=pd.Index([], name='ID'))
    with ThreadPool(processes=cpus) as pool, tqdm(total=len(amount)) as pbar:
        logger.info(f'Running {len(amount)=} pruning tasks')
        for i, (am, glob_sparsity, local_sparsity, test_res) in enumerate(pool.imap(
                functools.partial(
                    test_prune_amount, predictor, original_model, algorithm_func, X, y),
                amount)):

            stats = pd.concat((stats, pd.DataFrame({
                **({'subset_ID': subsets_data[i]} if subsets else {}),
                'amount': str(am),
                'global_sparsity': str(glob_sparsity),
                **{f'sparsityOf_{i}_{str(k)}': str(v) for i, (k, v) in enumerate(local_sparsity.items())}
            }, index=pd.Index([i], name='ID'))))
            results = pd.concat((results, pd.DataFrame(test_res, index=pd.Index([i], name='ID'))))
            pbar.update()

    logger.info('Saving models stats')
    stats.to_csv(os.path.join(store_path, 'models_stats.csv'))

    logger.info('Saving models leaderboard')
    results.sort_values(
        by=metric, ascending=False).to_csv(
        os.path.join(store_path, 'leaderboard.csv'))


@click.group()
@click.pass_context
def main(ctx):
    pass


@main.command(help='Test the pruned models obtained',
              context_settings={'show_default': True}, name='test')
@click.option('--directory', type=str, required=True,
              help='path to a result of a stochastic search')
@click.option('--algorithm', type=click.Choice(get_all_pruning_algorithms(), case_sensitive=False),
              help='pruning algorithm to be used', required=True)
@click.option('--top-subsets', type=int, default=10,
              help='top subsets within the degradation to consider')
@click.option('--subsets-degradation', type=float, default=0.10,
              help='maximum performance drop with respect to rfe greedy')
@click.option('--models-degradation', type=float, default=0.10,
              help='maximum performance drop with respect to rfe greedy')
@click.option('--cpus', type=int, default=get_default_cpu_number(),
              help='number of CPU cores to assign')
@click.option('--metric', type=click.Choice(get_scorer_names(), case_sensitive=False),
              help='evaluation metric', default='accuracy')
def test_command(*args, **kwargs):
    return test(*args, **kwargs)


@main.command(help='Search the pruned models', context_settings={'show_default': True}, name='search')
@click.option('--directory', type=str, required=True,
              help='path to a result of a stochastic search')
@click.option('--algorithm', type=click.Choice(get_all_pruning_algorithms(), case_sensitive=False),
              help='pruning algorithm to be used', required=True)
@click.option('--subsets', type=str, default=None,
              help='search activation using the identiifed subsets in the path')
@click.option('--top-subsets', type=int, default=10,
              help='top subsets within the degradation to consider')
@click.option('--subsets-degradation', type=float, default=0.10,
              help='maximum performance drop with respect to rfe greedy')
@click.option('--amount', type=float, multiple=True,
              default=np.round(np.linspace(0., 1., 21, endpoint=True), 2).tolist(),
              help='pruning amount to try')
@click.option('--attempts', type=int, default=1000,
              help='pruned models to look for')
@click.option('--cpus', type=int, default=get_default_cpu_number(),
              help='number of CPU cores to assign')
@click.option('--metric', type=click.Choice(get_scorer_names(), case_sensitive=False),
              help='evaluation metric', default='accuracy')
def search_command(*args, **kwargs):
    search(*args, **kwargs)


if __name__ == '__main__':
    main()
