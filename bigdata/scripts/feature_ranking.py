import click
import os
import random
from sklearn.metrics import get_scorer_names
import pandas as pd
from utils.ranking import get_all_ranking_algorithms, get_ranking_algorithm
from utils.common import create_dir, get_logger, set_seed, load_model_from_predictor, get_default_cpu_number
from utils.model import test_model_on_subset

logger = get_logger(f"{os.path.basename(__file__).replace('.py', '')}")


def draw_best(scores, top=True):
    scores_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    comparison_value = scores_sorted[0][1] if top else scores_sorted[-1][1]
    return random.choice([x[0] for x in scores_sorted if x[1] == comparison_value])


def rank(directory: str, algorithm: str, metric: str, cpus: int, recursive: bool, time_limit: int):
    global logger
    set_seed()

    logger.info('Loading Model')
    model, predictor, model_name = load_model_from_predictor(directory)
    logger.info('Loading data')

    X = pd.read_csv(os.path.join(directory, os.pardir, os.pardir,
                                 os.pardir, os.pardir, 'finetune.csv'), index_col='ID')
    y = X.pop('Label')

    store_path = os.path.join(model.path, f'feature_ranking_{algorithm}')
    create_dir(store_path, overwrite=False)

    is_sbe = algorithm.endswith('sbe')
    algorithm_func = get_ranking_algorithm(algorithm)

    current_features = X.columns.values.tolist() if is_sbe else []

    feature_importances = pd.DataFrame(columns=X.columns.values, index=pd.Index([], name='ID'))
    model_scores: pd.DataFrame = None

    if is_sbe:
        test_scores = test_model_on_subset(predictor, X, y, model, subset=current_features)
        logger.info(f'Testing {len(current_features)=} -> {test_scores[metric]=}')
        model_scores = pd.DataFrame(test_scores, index=pd.Index([len(current_features)], name='ID'))

    for i in range(len(X.columns.values), 0, -1):
        logger.info(f'Ranking {i=} models when using {len(current_features)=} features')

        scores = algorithm_func(X, y, current_features,
                                predictor=predictor, model_name=model_name, model=model,
                                n_cpus=cpus, target_metric=metric, time_limit=time_limit)

        feature_importances.loc[len(scores)] = scores

        if not is_sbe:
            worst_or_best = draw_best(scores, top=True)
            current_features.append(worst_or_best)
            logger.info(f'Adding {worst_or_best=} to the current best features')

        if is_sbe:
            worst_or_best = draw_best(scores, top=False)
            current_features.remove(worst_or_best)
            logger.info(f'Removing {worst_or_best=} from current features')

        test_scores = test_model_on_subset(predictor, X, y, model, subset=current_features)
        logger.info(f'Testing {len(current_features)=} -> {test_scores[metric]=}')
        if model_scores is None:
            model_scores = pd.DataFrame(test_scores, index=pd.Index([len(current_features)], name='ID'))
        else:
            model_scores.loc[len(current_features)] = test_scores

        if not recursive:
            break

    feature_importances.to_csv(os.path.join(
        store_path, f'feature_importance.csv'))
    model_scores.to_csv(os.path.join(
        store_path, f'leaderboard.csv'))


@click.command(help='Rank the features using one of the greedy algorithms and strategies',
               context_settings={'show_default': True})
@click.option('--directory', type=str, required=True,
              help='working directory with the automl chosen model')
@click.option('--algorithm', type=click.Choice(get_all_ranking_algorithms(), case_sensitive=False), required=True,
              help='the ranking algorithm to be used')
@click.option('--metric', type=click.Choice(get_scorer_names(), case_sensitive=False), default='accuracy',
              help='evaluation metric')
@click.option('--cpus', type=int, default=get_default_cpu_number(),
              help='number of CPU cores to assign')
@click.option('--mode', type=click.Choice(['recursive', 'first'], case_sensitive=False),
              help='whether to perform the algorithm at each step', default='recursive')
@click.option('--time-limit', type=int, default=60,
              help='time limit for the computation if using autogluon')
def main(*args, **kwargs):
    rank(*args, **kwargs)


if __name__ == '__main__':
    main()
