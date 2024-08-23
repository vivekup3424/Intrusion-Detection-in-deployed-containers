import functools
import math
import os
import sys
from multiprocessing.pool import ThreadPool
from typing import Dict, List, Tuple

import click
import numpy as np
import pandas as pd
from feature_ranking import test_model_on_subset
from sklearn.metrics import get_scorer_names
from tqdm import tqdm
from utils.common import (create_dir, get_default_cpu_number, get_logger,
                          load_model_from_predictor,
                          set_seed)

logger = get_logger(f"{os.path.basename(__file__).replace('.py', '')}")
SAFE_STOP_IT_ATTEMPTS = 1000


def get_combinations(trials, all_features, rank_file, size, best_list, is_weighted):
    global logger
    n_curr_features = round(len(all_features) * size) or 1

    if n_curr_features in rank_file.index:
        current_rank = rank_file.loc[n_curr_features].sort_values(ascending=False)
    else:
        current_rank = rank_file.loc[len(all_features)].sort_values(ascending=False)
    all_max_features_rank = rank_file.loc[len(all_features)].sort_values(ascending=False)
    remaining_weights = None

    all_combs: Dict[Tuple, float] = {}
    n_all_combs = math.comb(len(all_features), n_curr_features)
    logger.info(
        f'All possible combinations are {n_all_combs=}'
        f' as {n_curr_features=} and {len(all_features)=}')
    best_list = sorted(best_list, reverse=True)

    for i, b in enumerate(best_list):
        if len(all_combs) == n_all_combs or len(all_combs) == trials:
            break
        n_best = round(b * n_curr_features)
        constant_features = []
        remaining__features = all_features
        n_remaining_features = n_curr_features
        if n_best:
            constant_features = current_rank.index[:n_best].values.tolist()
            remaining__features = [x for x in all_features if x not in constant_features]
            n_remaining_features -= len(constant_features)
        remaining_rank = all_max_features_rank.drop(index=constant_features)

        if is_weighted:
            tmp = np.array([x for x in range(len(remaining_rank), 0, -1)]) + sys.float_info.epsilon
            remaining_weights = (tmp / tmp.sum())

        len_before_it = len(all_combs)
        current_max_combination_with_best = math.comb(len(remaining__features), n_remaining_features)
        n_left_attempt = min(
            max(
                round((trials - len_before_it) / (len(best_list) - i)),
                1),
            current_max_combination_with_best)

        taken_this_it = 0
        logger.info(f'{b=} {n_best=} {current_max_combination_with_best=} {len_before_it=}'
                    f' {n_left_attempt=} {n_remaining_features=}')
        while len(all_combs) < len_before_it + n_left_attempt:
            news = tuple(sorted(np.random.choice(remaining__features, n_remaining_features,
                         p=remaining_weights, replace=False).tolist() + constant_features))
            v = len(all_combs)
            all_combs.setdefault(news, b)
            taken_this_it = taken_this_it + 1 if v == len(all_combs) else 0
            if taken_this_it == SAFE_STOP_IT_ATTEMPTS:
                break

    return all_combs


def search(
        directory: str, subset_size: float, weighted: bool, best: List[float],
        cpus: int, attempts: int, metric: str):
    global logger
    set_seed()

    logger.info('Loading the model')
    model, predictor, _ = load_model_from_predictor(os.path.join(directory, os.pardir))

    logger.info('Loading the data')
    X = pd.read_csv(
        os.path.join(directory, os.pardir, os.pardir,
                     os.pardir, os.pardir, os.pardir, 'finetune.csv'),
        index_col='ID')
    y = X.pop('Label')

    store_path = os.path.join(
        directory, f"feature_subset_stochastic_search_{'weighted' if weighted else 'random'}",
        f'feature_subsets_{subset_size:.2f}s')
    create_dir(store_path, overwrite=False)

    rank_file = pd.read_csv(os.path.join(directory, 'feature_importance.csv'), index_col='ID')

    logger.info('Computing Combinations')
    all_combs = get_combinations(attempts, X.columns.values, rank_file, subset_size,
                                 best, weighted)
    pd.DataFrame(
        {'best': [v for v in all_combs.values()],
         'features': [str(tuple(x)) for x in all_combs]},
        index=pd.Index(range(len(all_combs)), name='ID')).to_csv(os.path.join(store_path, 'subsets.csv'))

    res = pd.DataFrame()
    logger.info('Launching test jobs')
    with ThreadPool(processes=cpus) as pool, tqdm(total=len(all_combs)) as pbar:
        for i, x in enumerate(
            pool.imap(
                functools.partial(test_model_on_subset, predictor, X, y, model),
                all_combs.keys())):
            res = pd.concat((res, pd.DataFrame(x, index=pd.Index([i], name='ID'))))
            pbar.update()
    res.sort_values(by=metric, inplace=True, ascending=False)
    res.to_csv(os.path.join(store_path, 'leaderboard.csv'))


@click.command(help='Run the search of the subsets for the given size',
               context_settings={'show_default': True})
@click.option('--directory', type=str, required=True,
              help='working directory with the greedy ranking of the features')
@click.option('--subset-size', type=float, required=True,
              help='percentage size of the subset to look for')
@click.option('--left-policy', type=click.Choice(['random', 'weighted'], case_sensitive=False),
              help='whether to select remaining features randomly or weighted', default='random')
@click.option('--best', type=float, multiple=True,
              default=np.round(np.linspace(0., 1., 21, endpoint=True), 2).tolist(),
              help='percentage size of the best features to keep from greedy search')
@click.option('--cpus', type=int, default=get_default_cpu_number(),
              help='number of CPU cores to assign')
@click.option('--attempts', type=int, default=1000,
              help='number of subsets of the given size to look for')
@click.option('--metric', type=click.Choice(get_scorer_names(), case_sensitive=False),
              help='evaluation metric', default='accuracy')
def main(*args, **kwargs):
    search(*args, **kwargs)


if __name__ == '__main__':
    main()
