"""
Main file to run autoML with HOP and NAS techniques
"""
import os

import click
import pandas as pd
from autogluon.common import space
from autogluon.features.generators import IdentityFeatureGenerator
from autogluon.tabular import TabularPredictor
from utils.common import (create_dir, get_default_cpu_number, get_logger,
                          load_predictor, set_seed)
from utils.model import get_prunable, network_layers_sparsity

logger = get_logger(f"{os.path.basename(__file__).replace('.py', '')}")


def get_hpo_scheduler(trials: int):
    return {
        'num_trials': trials,
        'scheduler': 'local',
        'searcher': 'bayes'
    }


def get_hyperparameters(epochs: int, epochs_wo_improve: int):
    return {
        'NN_TORCH': {
            # https://github.com/autogluon/autogluon/blob/80c0ac6ec52248bcecd51841307a123c99736a59/tabular/src/autogluon/tabular/models/tabular_nn/hyperparameters/parameters.py#L25
            # using embed_min_categories there should not be embedding
            'proc.embed_min_categories': 1000,
            'embedding_size_factor': 1.0,
            'embed_exponent': 0.56,
            'max_embedding_dim': 100,
            'use_ngram_features': False,
            'proc.skew_threshold': 100,
            'proc.max_category_levels': 10000,
            'proc.impute_strategy': 'median',
            'weight_decay': 1e-6,
            'y_range_extend': 0,
            'y_range': 1,
            'use_batchnorm': space.Categorical(True, False),
            'optimizer': space.Categorical('adam', 'sgd'),
            'activation': space.Categorical('relu', 'tanh'),
            'learning_rate': space.Real(1e-5, 1e-1, default=5e-4, log=True),
            'dropout_prob': space.Real(0.0, 0.8, default=0.2),
            'num_layers': space.Int(1, 100, default=3),
            'hidden_size': space.Int(1, 100, default=16),
            'batch_size': space.Int(1, 512, default=256),
            'max_batch_size': 512,
            'loss_function': 'auto',
            'num_epochs': epochs,
            'epochs_wo_improve': epochs_wo_improve
        }
    }


def dump(directory: str, model_name: str):
    set_seed()
    predictor = load_predictor(directory)

    logger.info('Storing predictor fit summary and leaderboard')
    predictor.fit_summary(verbosity=2)
    predictor.leaderboard(extra_info=True, silent=True).to_csv(
        os.path.join(predictor.path, 'leaderboard.csv'))

    logger.info(f'Loading model {model_name}')
    model = predictor._trainer.load_model(model_name if model_name != 'best_model' else predictor.get_model_best())

    logger.info(f'Saving {model_name=} info')
    model.save_info()

    logger.info(f'Storing {model_name=} network sparsity')
    layers = get_prunable(model.model)
    glob_sparsity, local_sparsities = network_layers_sparsity(layers)
    pd.DataFrame({
        'global_sparsity': glob_sparsity,
        **{k: v for k, v in zip([f'sparsityOf_{i}_{str(x)}' for i, x in enumerate(layers)], local_sparsities)}
    }, index=pd.Index([0], name='ID')).to_csv(os.path.join(model.path, 'model_stats.csv'))

    logger.info(f'Logging {model_name=} model architecture')
    with open(os.path.join(model.path, 'model.log'), 'w') as f:
        f.write(str(model.model))

    logger.info(f'Storing {model_name=} architecture')
    df = pd.DataFrame(index=pd.Index([], name='ID'))
    for f in ('train', 'validation', 'finetune', 'test'):
        logger.info(f'Testing on {f} set')
        X = pd.read_csv(os.path.join(directory, os.pardir, f'{f}.csv'), index_col='ID')
        Y = X.pop('Label')
        df = pd.concat(
            (df, pd.DataFrame(predictor.evaluate_predictions(
                Y, pd.Series(model.predict(X)),
                auxiliary_metrics=True, silent=True), index=pd.Index([f], name='ID'))))
    df.to_csv(
        os.path.join(model.path, 'leaderboard.csv'))

    if model_name == 'best_model':
        os.chdir(predictor._trainer.path)
        if not os.path.exists(model_name):
            logger.info(f"Creating symling 'best_model' -> {model_name=}")
            os.symlink(model.name, 'best_model', target_is_directory=True)


def search(directory: str, epochs: int, patience: int,
           cpus: int, gpus: int, attempts: int, time_limit: int, metric: str):
    set_seed()

    os.environ['RAY_ADDRESS'] = 'local'
    os.environ.pop('AG_DISTRIBUTED_MODE', None)

    train_data = pd.read_csv(os.path.join(directory, 'train.csv'), index_col='ID', low_memory=False)
    validation_data = pd.read_csv(os.path.join(directory, 'validation.csv'), index_col='ID', low_memory=False)

    create_dir(os.path.join(directory, 'automl_search'), overwrite=False)

    predictor = TabularPredictor(
        label='Label', eval_metric=metric,
        verbosity=2, log_to_file=False,
        path=os.path.join(directory, 'automl_search'))

    predictor.fit(
        train_data,
        tuning_data=validation_data,
        fit_weighted_ensemble=False,
        time_limit=time_limit,
        num_cpus=cpus,
        num_gpus=gpus,
        hyperparameter_tune_kwargs=get_hpo_scheduler(attempts),
        hyperparameters=get_hyperparameters(epochs, patience),
        feature_generator=IdentityFeatureGenerator(),
        feature_prune_kwargs=None)


@click.group(context_settings={'show_default': True})
@click.pass_context
def main(ctx):
    pass


@main.command(help='Test and dump parameters of a given model previously generated via automl.', name='dump')
@click.option('--directory', type=str, required=True,
              help='working directory with the automl generated models')
@click.option('--model-name', type=str, default='best_model',
              help='name of the automl generated model to use')
def dump_command(*args, **kwargs):
    dump(*args, **kwargs)


@main.command(help='Perform the automl model HPO search.', name='search')
@click.option('--directory', type=str, required=True,
              help='working directory with the dataset to be used')
@click.option('--epochs', type=int, default=1000,
              help='number of maximum training epochs')
@click.option('--patience', type=int, default=20,
              help='number of maximum epochs without improvements')
@click.option('--cpus', type=int, default=get_default_cpu_number(),
              help='number of CPU cores to assign')
@click.option('--gpus', type=int, default=0,
              help='number of GPUs, if any available')
@click.option('--attempts', type=int, default=1000,
              help='number of HPO trails to search')
@click.option('--time-limit', type=int, default=60 * 60 * 1,
              help='time limit for the running')
@click.option('--metric', type=str, default='accuracy',
              help='evaluation metric for the training')
def search_command(*args, **kwargs):
    search(*args, **kwargs)


if __name__ == '__main__':
    main()
