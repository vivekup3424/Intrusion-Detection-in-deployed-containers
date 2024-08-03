"""
File containing used parameters throughout the notebook execution
"""
import argparse
import logging
import os
import time
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
import threadpoolctl
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

from intellect import ranking
from intellect.dataset import (ContinuouLearningAlgorithm, Dataset,
                               FeatureAvailability, InputForLearn,
                               balance_classes, format_dataframe_columns_names,
                               format_dataframe_columns_values,
                               indexes_for_oracle_learning, load_dataframes,
                               portions_from_data, remove_constant_columns)
from intellect.distance import distributions_to_probabilities
from intellect.inspect import set_seed
from intellect.io import (TimeoutIterator, create_dir, dump, get_logger, load,
                          recursive_find_file)
from intellect.model.sklearn import model, pruning
from intellect.ranking import (prune_and_subset_search, prune_search,
                               sequential_backward_elimination, subset_search,
                               subset_search_and_prune)
from intellect.scoring import compute_metric_percategory


@dataclass
class Config:

    #################################
    # Parameters for dataset creation
    #################################

    # path to datasets 2017 and 2019
    datasets: list[str]
    samples_per_file: int
    samples_per_category: int

    # column marked as label in these datasets
    label: str

    # labels to be considered as benign (class=0 in binary classification)
    benign_labels: list[str]

    # columns to remove from datasets (session identifiers and non-commond features)
    excluded_columns: list[str]

    ###############################
    # Parameters for model training
    ###############################

    # model parameters
    n_hidden_layers: int
    n_hidden_units: int
    dropout_hidden: float
    learning_rate: str
    activation: str

    # training parameters
    batch_size: float
    max_epochs: int
    epochs_wo_improve: int

    # dataset portions: Train, Validation, Finetune, Refit, Test
    dataset_portions: list[float]

    ##################################################
    # Parameters for feature ranking and model pruning
    ##################################################

    # traffic categories that only the specific client (organization) has
    client_categories: list[str]

    # timeout for algorithms (seconds)
    time_limit: int

    # sizes of the feature subsets to search
    target_subset_ratios: list[str]

    # ratios of connections to be pruned from the network
    prune_ratios: list[str]

    # explored solution for each pruning/subset ratio
    explored_per_ratio: int

    # maximum performance drop ratio accepted
    performance_drops: list[float]

    prune_method: str

    rank_method: str

    benchmark_iter: int

    ###################################
    # Parameters for client model refit
    ###################################

    # common hyperparameters
    common_parameters: dict[str, object]

    # all possible tested scenarios:
    # o_to_o:  oracle to oracle scenario, the student/client model is a copy of the oracle model.
    # o_to_po: oracle to pruned oracle scenario, where the student/client model is a pruned version of the oracle.
    # o_to_eo: oracle to edge oracle scenario, where the student/client is a copy of the oracle model, but it is provided
    #               with only a limited set of features
    # o_to_ec: oracle to edge client scenario, where the student/client is a pruned version of the oracle model AND it is provided
    #               with only a limited set of features
    scenarios: dict[str, dict[str, object]]

    # tested seeds
    seeds: list[int]

    def __post_init__(self):
        for v in self.scenarios.values():
            for c in v:
                c.update(self.common_parameters)
                c['algorithm'] = ContinuouLearningAlgorithm[c['algorithm']]
                c['availability'] = FeatureAvailability[c['availability']]
                c['learn_input'] = InputForLearn[c['learn_input']]
        if not all(k in ('o_to_o', 'o_to_po', 'o_to_eo', 'o_to_ec') for k in self.scenarios):
            raise Exception('Some key is not recognized')
        self.rank_method = getattr(ranking, self.rank_method)
        self.prune_method = getattr(pruning, self.prune_method)

    @property
    def output_path(self):
        return '-'.join([os.path.basename(os.path.normpath(x)) for x in self.datasets]) + '_output'

    def get_dataset(self, seed=42):
        set_seed(seed)
        train, validation, test = portions_from_data(os.path.join(self.output_path, 'dataset.h5'), normalize=True,
                                                     benign_labels=self.benign_labels, ratios=self.dataset_portions)
        return train, validation, test


def datasetter(config: Config, seed, verbose=True):
    # Dataset Preparation
    # This notebook is used as a reference for the preparation of a dataset.
    # From the given *CSV* files, we perform some processing operation to create the final dataset.
    set_seed(seed)
    logger = get_logger('datasetter', log_level=logging.INFO if verbose else logging.ERROR)

    # At first, load only the label column for both the two datasets
    df = []
    tot_files = []
    names = []
    for ds in config.datasets:
        set_seed(seed)
        files = recursive_find_file(ds, endswith_condition='.csv')
        tot_files += files
        tmp = load_dataframes(files, maxlines=config.samples_per_file, only_labels_str=config.label)
        name = os.path.basename(os.path.normpath(ds))
        names += [name]
        # Keep track of the original file and index (row in the file) to which each label comes from.
        # Insert to each label the string containing the dataset name.
        for k, v in tmp.items():
            v['File'] = k
            v['Indexes'] = v.index.values
            v[config.label] += f'-{name}'
            df.append(v)
    df = pd.concat(df)

    # Print distribution of labels.
    logger.info(df[config.label].value_counts())

    # Balance between benign and malicious classes.
    df_balanced = balance_classes(df, [x + f'-{name}' for x in config.benign_labels for name in names])
    logger.info(df_balanced[config.label].value_counts())

    # Take a maximum of samples per category
    to_pick = dict(df_balanced[config.label].value_counts())
    for k in to_pick:
        if k in config.benign_labels:
            continue
        if to_pick[k] > config.samples_per_category:
            to_pick[k] = config.samples_per_category

    shrinked = balance_classes(df_balanced.groupby(by=config.label).apply(lambda x: x.sample(
        n=to_pick[x.name])).droplevel(level=0), [f'{x}-{y}' for y in names for x in config.benign_labels])

    logger.info(shrinked[config.label].value_counts())

    # Remove columns that do not appear in both the two datasets, and remove the additionally specified ones.
    cols = set([y
                for x in tot_files
                for y in pd.read_csv(x, index_col=0, skipinitialspace=True, nrows=0).columns.tolist()])
    for x in config.excluded_columns:
        cols.discard(x)
    cols = list(cols)
    logger.info(f'{len(cols)=}')

    # Load original samples from the files.
    final = []
    for x in tot_files:
        tmp = dict.fromkeys((shrinked[shrinked['File'] == x]['Indexes'] + 1).tolist() + [0])
        tmp_df = pd.read_csv(x, index_col=0, skipinitialspace=True, usecols=cols,
                             skiprows=lambda x: x not in tmp, nrows=config.samples_per_file)
        final.append(tmp_df)
    final = pd.concat(final, ignore_index=True)
    logger.info(final.columns.values)
    logger.info(final[config.label].value_counts())

    # Transform all the possible features in numeric. This is useful in case the *read_csv* function was not able to correctly detect the data type when loading the *CSV*.
    df_numeric = final.apply(pd.to_numeric, errors='ignore')

    # Converting remaining categorical features (e.g., HTTP method, GET, POST) into numerical, if any.
    cat_columns = [col_name for col_name,
                   dtype in df_numeric.dtypes.items() if dtype == object and col_name not in (config.label, 'Source')]
    if cat_columns:
        logger.info(f'Converting following categorical to numerical {cat_columns}')
        df_numeric[cat_columns] = df_numeric[cat_columns].astype('category')
        df_numeric[cat_columns] = df_numeric[cat_columns].apply(lambda x: x.cat.codes)
        df_numeric[cat_columns] = df_numeric[cat_columns].astype('int')

    # Removing rows with missing values (e.g., NaN or Inf).
    logger.info(f'Shape before dropping NaN {df_numeric.shape}')
    df_wo_nan = df_numeric.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
    logger.info(f'Resulting shape {df_wo_nan.shape}')

    logger.info(df_wo_nan[config.label].value_counts())

    # Removing all the constant features among the selected dataset. These feature might result constant after the balancing of the samples, meaning that few samples with potentially different values for these features have been removed.
    prevs = set(df_wo_nan.columns.values)
    df_wo_const: pd.DataFrame = remove_constant_columns(df_wo_nan)
    logger.info(f'Remove constant removed the features {prevs - set(df_wo_const.columns.values)}')

    # Remove identical features (cloned columns or same distribution). *Fwd Header Length.1* is an exact copy of *Fwd Header Length*, while the others present the same values of other columns in the dataset, hence only one is kept.
    prevs = set(df_wo_const.columns.values)
    df_wo_dup = df_wo_const[df_wo_const.describe(include='all').T.drop_duplicates().T.columns]
    logger.info(f'Identical columns removed {prevs - set(df_wo_dup.columns.values)}')

    # Format column and label names (remove extra white spaces, UNICODE characters if any, etc.).
    df_formatted = format_dataframe_columns_values(format_dataframe_columns_names(df_wo_dup), config.label)
    df_formatted.index.name = 'ID'

    # Dump the dataset.
    dump(df_formatted, os.path.join(config.output_path, 'dataset.h5'))

def correlations(config: Config, seed, output_dir: str, verbose=True):
    set_seed(seed)
    logger = get_logger('correlations', log_level=logging.INFO if verbose else logging.ERROR)
    train, validation, test = config.get_dataset(seed=seed)
    unified = train.join(validation).join(test)
    ret = {}
    for x, y in combinations(unified.categories, 2):
        c, d = unified.filter_categories([x]), unified.filter_categories([y])
        out = {}
        for col in c.X.columns.values:
            _, v1, v2 = distributions_to_probabilities(c.X[col].to_numpy(), d.X[col].to_numpy(), only_common=False)
            out[col] = jensenshannon(v1, v2)
        ret[(x, y)] = out
        logger.info(f'Finished combination {x} {y}')
    pd.DataFrame(ret.values(), index=ret.keys()).to_csv(os.path.join(output_dir, 'jensenshannon_perfeature.csv'))

def train(config: Config, seed, output_dir: str, verbose=True):
    set_seed(seed)
    train, validation, _ = config.get_dataset(seed=seed)
    set_seed(seed)
    logger = get_logger('train', log_level=logging.INFO if verbose else logging.ERROR)
    oracle = model.EnhancedMlpRegressor(train.classes, dropout=config.dropout_hidden,
                                        hidden_layer_sizes=(config.n_hidden_units,)*config.n_hidden_layers,
                                        learning_rate=config.learning_rate, activation=config.activation)
    logger.info('Training the model')
    history, _ = oracle.fit(train, validation_dataset=validation, batch_size=config.batch_size,
                            max_epochs=config.max_epochs, epochs_wo_improve=config.epochs_wo_improve, verbose=verbose)

    oracle.save(os.path.join(output_dir, 'oracle'))

    history_df = pd.DataFrame(history)
    dump(history_df, os.path.join(output_dir, 'oracle_history.csv'))

def test_baselines(config: Config, seed, output_dir: str, verbose=True):
    set_seed(seed)
    train, validation, test = config.get_dataset(seed=seed)
    logger = get_logger('test_baselines', log_level=logging.INFO if verbose else logging.ERROR)
    train_few = train.filter_categories(config.client_categories).balance_categories()
    validation_few = validation.filter_categories(config.client_categories).balance_categories()
    test_few = test.filter_categories(config.client_categories).balance_categories()

    oracle = model.EnhancedMlpRegressor.load(os.path.join(output_dir, 'oracle'))
    logger.info('Computing metrics against entire datasets with all categories and with only client categories')
    x = pd.DataFrame([compute_metric_percategory(train.y, oracle.predict(train.X), train._y),
                      compute_metric_percategory(validation.y, oracle.predict(validation.X), validation._y),
                      compute_metric_percategory(test.y, oracle.predict(test.X), test._y),
                      compute_metric_percategory(train_few.y, oracle.predict(train_few.X), train_few._y),
                      compute_metric_percategory(validation_few.y, oracle.predict(validation_few.X), validation_few._y),
                      compute_metric_percategory(test_few.y, oracle.predict(test_few.X), test_few._y)], index=[('AllC', 'Train'),
                                                                                                               ('AllC',
                                                                                                                'Validation'),
                                                                                                               ('AllC',
                                                                                                                'Test'),
                                                                                                               ('FewC',
                                                                                                                'Train'),
                                                                                                               ('FewC',
                                                                                                                'Validation'),
                                                                                                               ('FewC', 'Test')])
    dump(x, os.path.join(output_dir, 'oracle_baselines.csv'))

def train_evaluation(
        config: Config, seed, output_dir: str, prune_ratio=None, ds_only_client_traffic=True, verbose=False):
    set_seed(seed)
    train, validation, _ = config.get_dataset(seed=seed)

    if ds_only_client_traffic:
        train = train.filter_categories(config.client_categories).balance_categories()
        validation = validation.filter_categories(config.client_categories).balance_categories()

    m_path = os.path.join(output_dir, 'evaluation_full' if prune_ratio is None else f'evaluation_pruned_{prune_ratio}')

    new_hidu = int((1-prune_ratio) * config.n_hidden_units) if prune_ratio else config.n_hidden_units

    set_seed(default=seed)
    m = model.EnhancedMlpRegressor(
        train.classes, hidden_layer_sizes=(new_hidu,) * config.n_hidden_layers,
        dropout=config.dropout_hidden, max_iter=config.max_epochs)

    hs, _ = m.fit(
        train, validation_dataset=validation, batch_size=config.batch_size,
        max_epochs=config.max_epochs, epochs_wo_improve=config.epochs_wo_improve, shuffle=True, verbose=verbose)

    m.save(m_path)
    dump(hs, f'{m_path}_history.json')

def test_evaluation(config: Config, seed, output_dir, prune_ratio=None, ds_only_client_traffic=True, verbose=False):
    set_seed(seed)
    _, _, test = config.get_dataset(seed=seed)

    if ds_only_client_traffic:
        test = test.filter_categories(config.client_categories).balance_categories()

    m_path = os.path.join(output_dir, 'evaluation_full' if prune_ratio is None else f'evaluation_pruned_{prune_ratio}')
    set_seed(default=seed)
    m = model.EnhancedMlpRegressor.load(m_path)
    res = []
    # warmup
    for _ in range(20):
        m.predict(test.X)
    for _ in range(config.benchmark_iter) if not verbose else tqdm(range(config.benchmark_iter)):
        start = time.time_ns()
        m.predict(test.X)
        end = time.time_ns()
        res.append(end-start)
    dump(res, m_path + '.json')


def compare_pruning(config: Config, seed, output_dir: str, verbose=True, ds_only_client_traffic=True):
    config.prune_method = pruning.globally_neurons_l1
    only_pruning(config, seed, output_dir, verbose=verbose, ds_only_client_traffic=ds_only_client_traffic)
    config.prune_method = pruning.globally_unstructured_connections_l1
    only_pruning(config, seed, output_dir, verbose=verbose, ds_only_client_traffic=ds_only_client_traffic)


def recursive_ss(config: Config, seed, output_dir: str, verbose=True,
                 ds_only_client_traffic=True, fixed_rank=True, remove_zero_first=True):
    set_seed(seed)
    _, validation, _ = config.get_dataset(seed=seed)

    if ds_only_client_traffic:
        validation = validation.filter_categories(config.client_categories).balance_categories()

    oracle = model.EnhancedMlpRegressor.load(os.path.join(output_dir, 'oracle'))

    logger = get_logger('recursive_ss', log_level=logging.INFO if verbose else logging.ERROR)

    t = 'few_c' if ds_only_client_traffic else 'all_c'
    t2 = 'fixed_rank' if fixed_rank else 'iterative_rank'
    t3 = 'zero_first' if remove_zero_first else 'zero_not_first'

    if fixed_rank:
        kwargs = {'fixed_rank': config.rank_method(oracle, validation)}
    else:
        kwargs = {'rank_algorithm': config.rank_method}

    set_seed(seed)
    asd = pd.DataFrame(columns=validation.features, index=pd.Index([], name='#Features'))
    asd2 = pd.DataFrame(columns=['Global'] + validation.categories, index=pd.Index([], name='#Features'))
    logger.info('Running sequential backward elimination')
    for metric_score, _, features_scores in sequential_backward_elimination(
            oracle, validation, remove_zero_first=remove_zero_first, **kwargs):
        set_seed(seed)
        asd.loc[len(features_scores)] = features_scores
        asd2.loc[len(features_scores)] = metric_score
    if not os.path.isdir(os.path.join(output_dir, 'ranking')):
        create_dir(os.path.join(output_dir, 'ranking'))

    dump(asd, os.path.join(output_dir, 'ranking', f'recursive_ss_{t}_{t2}_{t3}_features.csv'))
    dump(asd2, os.path.join(output_dir, 'ranking', f'recursive_ss_{t}_{t2}_{t3}_scores.csv'))

def only_pruning(config: Config, seed, output_dir: str, verbose=True,
                 ds_only_client_traffic: bool = True):
    set_seed(seed)
    _, validation, _ = config.get_dataset(seed=seed)

    if ds_only_client_traffic:
        validation = validation.filter_categories(config.client_categories).balance_categories()

    oracle = model.EnhancedMlpRegressor.load(os.path.join(output_dir, 'oracle'))

    traffic_prefix = 'few_c' if ds_only_client_traffic else 'all_c'
    new_df_comb = pd.DataFrame(columns=['Prune Ratio', 'Global'] + validation.categories)
    set_seed(seed)
    logger = get_logger('only_pruning', log_level=logging.INFO if verbose else logging.ERROR)
    logger.info('Running prune search')
    for k, v in TimeoutIterator(
            prune_search(
                oracle.clone(init=False),
                validation, config.prune_method, config.prune_ratios),
            time_limit=config.time_limit):
        new_df_comb.loc[len(new_df_comb)] = [k] + list(v.values())
    new_df_comb = new_df_comb.sort_values(by='Global', ascending=False)

    if not os.path.isdir(os.path.join(output_dir, 'ranking')):
        create_dir(os.path.join(output_dir, 'ranking'))

    dump(
        new_df_comb, os.path.join(
            output_dir, 'ranking',
            f'traffic_{traffic_prefix}_pruning_ratios_only_{config.prune_method.__name__}.csv'))

def only_stochastic_search(config: Config, seed, output_dir: str, subset_size_ratio, verbose=True,
                           rank_only_client_traffic: bool = True, ds_only_client_traffic: bool = True):
    set_seed(seed)
    _, validation, _ = config.get_dataset(seed=seed)

    if ds_only_client_traffic:
        validation = validation.filter_categories(config.client_categories).balance_categories()

    oracle = model.EnhancedMlpRegressor.load(os.path.join(output_dir, 'oracle'))

    rank_prefix = 'few_c' if rank_only_client_traffic else 'all_c'
    traffic_prefix = 'few_c' if ds_only_client_traffic else 'all_c'

    set_seed(seed)
    rank = config.rank_method(oracle, validation)

    logger = get_logger('only_stochastic_search', log_level=logging.INFO if verbose else logging.ERROR)
    logger.info('Running prune search')

    set_seed(seed)
    new_df_comb = pd.DataFrame(columns=validation.features, index=pd.Index([], name='ID'))
    new_df_comb2 = pd.DataFrame(columns=['Global'] + validation.categories, index=pd.Index([], name='ID'))
    for k, v in TimeoutIterator(
            subset_search(
                oracle.clone(init=False),
                validation, subset_size_ratio, config.explored_per_ratio, rank=rank),
            time_limit=config.time_limit):
        new_df_comb.loc[len(new_df_comb)] = {fname: (1 if fname in k else np.nan) for fname in validation.features}
        new_df_comb2.loc[len(new_df_comb2)] = v
    new_df_comb2 = new_df_comb2.sort_values(by='Global', ascending=False)

    if not os.path.isdir(os.path.join(output_dir, 'ranking')):
        create_dir(os.path.join(output_dir, 'ranking'))

    dump(new_df_comb, os.path.join(output_dir, 'ranking',
         f'rank_{rank_prefix}_traffic_{traffic_prefix}_subsetsize_{subset_size_ratio}_features.csv'))
    dump(new_df_comb2, os.path.join(output_dir, 'ranking',
         f'rank_{rank_prefix}_traffic_{traffic_prefix}_subsetsize_{subset_size_ratio}_scores.csv'))

def stochastic_search_then_pruning(config: Config, seed, output_dir: str, subset_size_ratio, verbose=True,
                                   rank_only_client_traffic: bool = True, ds_only_client_traffic: bool = True):
    set_seed(seed)
    _, validation, _ = config.get_dataset(seed=seed)

    if ds_only_client_traffic:
        validation = validation.filter_categories(config.client_categories).balance_categories()

    oracle = model.EnhancedMlpRegressor.load(os.path.join(output_dir, 'oracle'))

    rank_prefix = 'few_c' if rank_only_client_traffic else 'all_c'
    traffic_prefix = 'few_c' if ds_only_client_traffic else 'all_c'

    set_seed(seed)
    rank = config.rank_method(oracle, validation)

    logger = get_logger('stochastic_search_then_pruning', log_level=logging.INFO if verbose else logging.ERROR)
    logger.info('Running stochastic search then pruning')

    set_seed(seed)
    new_df_comb = pd.DataFrame(
        columns=validation.features, index=pd.Index([], name='ID'))
    new_df_comb2 = pd.DataFrame(
        columns=['Prune Ratio', 'Global'] + validation.categories, index=pd.Index([], name='ID'))
    for k, v, prune_ratio in TimeoutIterator(
            subset_search_and_prune(
                oracle.clone(init=False),
                config.prune_method, validation, config.prune_ratios, subset_size_ratio,
                config.explored_per_ratio, rank=rank),
            time_limit=config.time_limit):
        new_df_comb.loc[len(new_df_comb)] = {fname: (1 if fname in k else np.nan) for fname in validation.features}
        new_df_comb2.loc[len(new_df_comb2)] = {'Prune Ratio': prune_ratio, **v}
    new_df_comb2 = new_df_comb2.sort_values(by='Global', ascending=False)

    if not os.path.isdir(os.path.join(output_dir, 'ranking')):
        create_dir(os.path.join(output_dir, 'ranking'))

    dump(
        new_df_comb, os.path.join(
            output_dir, 'ranking',
            f'rank_{rank_prefix}_traffic_{traffic_prefix}_combo_subsetsize_{subset_size_ratio}_pruned_models_{config.prune_method.__name__}_features.csv'))
    dump(
        new_df_comb2, os.path.join(
            output_dir, 'ranking',
            f'rank_{rank_prefix}_traffic_{traffic_prefix}_combo_subsetsize_{subset_size_ratio}_pruned_models_{config.prune_method.__name__}_scores.csv'))

def pruning_then_stochastic_search(config: Config, seed, output_dir: str, subset_size_ratio, verbose=True,
                                   rank_only_client_traffic: bool = True, ds_only_client_traffic: bool = True):
    set_seed(seed)
    _, validation, _ = config.get_dataset(seed=seed)

    if ds_only_client_traffic:
        validation = validation.filter_categories(config.client_categories).balance_categories()

    oracle = model.EnhancedMlpRegressor.load(os.path.join(output_dir, 'oracle'))

    rank_prefix = 'few_c' if rank_only_client_traffic else 'all_c'
    traffic_prefix = 'few_c' if ds_only_client_traffic else 'all_c'

    set_seed(seed)
    rank = config.rank_method(oracle, validation)

    logger = get_logger('pruning_then_stochastic_search', log_level=logging.INFO if verbose else logging.ERROR)
    logger.info('Running pruning then stochastic search')

    set_seed(seed)
    new_df_comb = pd.DataFrame(columns=validation.features, index=pd.Index([], name='ID'))
    new_df_comb2 = pd.DataFrame(
        columns=['Prune Ratio', 'Global'] + validation.categories, index=pd.Index([], name='ID'))
    for prune_ratio, k, v in TimeoutIterator(
            prune_and_subset_search(
                oracle.clone(init=False),
                config.prune_method, validation, config.prune_ratios, subset_size_ratio,
                config.explored_per_ratio, rank=rank),
            time_limit=config.time_limit):
        new_df_comb.loc[len(new_df_comb)] = {fname: (1 if fname in k else np.nan) for fname in validation.features}
        new_df_comb2.loc[len(new_df_comb2)] = {'Prune Ratio': prune_ratio, **v}
    new_df_comb2 = new_df_comb2.sort_values(by='Global', ascending=False)

    if not os.path.isdir(os.path.join(output_dir, 'ranking')):
        create_dir(os.path.join(output_dir, 'ranking'))

    dump(
        new_df_comb, os.path.join(
            output_dir, 'ranking',
            f'rank_{rank_prefix}_traffic_{traffic_prefix}_combo_pruned_models_subsetsize_{subset_size_ratio}_{config.prune_method.__name__}_features.csv'))
    dump(
        new_df_comb2, os.path.join(
            output_dir, 'ranking',
            f'rank_{rank_prefix}_traffic_{traffic_prefix}_combo_pruned_models_subsetsize_{subset_size_ratio}_{config.prune_method.__name__}_scores.csv'))


oracle_cache = {}
student_cache = {}

def _run_test(config: Config, save_prefix: str,
              retrain_ds: Dataset, retrain_val_ds: Dataset, test_ds,
              features_available=None, prune_ratio=None,
              availability=None, seed=42, verbose=False, **kwargs):
    if features_available is None:
        features_available = []
    set_seed(seed)
    oracle_net: model.EnhancedMlpRegressor = model.EnhancedMlpRegressor.load(
        os.path.normpath(os.path.join(save_prefix, os.pardir, os.pardir, 'oracle')))
    student_net: model.EnhancedMlpRegressor = model.EnhancedMlpRegressor.load(
        os.path.normpath(os.path.join(save_prefix, os.pardir, os.pardir, 'oracle')))

    if prune_ratio is not None:
        student_net = config.prune_method(student_net, prune_ratio)

    idx, idx_oracle = indexes_for_oracle_learning(retrain_ds, features_available, availability)

    key_oracle = (hash(retrain_ds), hash(str(idx_oracle)))
    key_student = (hash(retrain_ds), hash(str(idx)), hash(prune_ratio))

    if key_oracle not in oracle_cache:
        oracle_tmp_test = test_ds.clone()
        oracle_tmp_test.X.iloc[:, idx_oracle] = 0.
        oracle_cache[key_oracle] = compute_metric_percategory(
            oracle_tmp_test.y, oracle_net.predict(oracle_tmp_test.X), oracle_tmp_test._y)

    tmp_test = test_ds.clone()
    tmp_test.X.iloc[:, idx] = 0.

    if key_student not in student_cache:
        student_cache[key_student] = compute_metric_percategory(
            tmp_test.y, student_net.predict(tmp_test.X), tmp_test._y)

    hs, m = student_net.fit(
        retrain_ds, validation_dataset=retrain_val_ds, oracle=oracle_net, idx_active_features=idx,
        idx_active_features_oracle=idx_oracle, monitor_ds=test_ds, verbose=verbose, **kwargs)

    dump(m, f'{save_prefix}_monitored.csv')
    dump(hs, f'{save_prefix}_history.csv')

    cols = ['Global'] + config.client_categories + [v
                                                    for v in test_ds._y.value_counts().sort_values(ascending=False).index.values
                                                    if v not in config.client_categories]
    df = pd.DataFrame(columns=cols)
    df.loc['Test Before'] = student_cache[key_student]
    df.loc['Test After'] = compute_metric_percategory(tmp_test.y, student_net.predict(tmp_test.X), tmp_test._y)
    df.loc['Oracle Test'] = oracle_cache[key_oracle]
    dump(df, f'{save_prefix}.csv')

def handle_scenario(config: Config, seed, output_dir: str, scenario_name, verbose=True,
                    subset_size=None, only_client_categories=True,
                    rank_only_client_traffic=True, performance_drop=None, prune_first=True):
    cols = config.client_categories if only_client_categories else ['Global']
    n = 'few_c' if only_client_categories else 'all_c'
    nn = 'few_c' if rank_only_client_traffic else 'all_c'
    n_camelized = ''.join(x.capitalize() for x in n.lower().split('_'))
    prune_ratio = None
    subset = []

    logger = get_logger('handle_scenario', log_level=logging.INFO if verbose else logging.ERROR)

    if scenario_name != 'o_to_o':
        bs = load(os.path.join(
            output_dir,
            'oracle_baselines.csv'),
            index_col=0)
        mins = (bs.loc[f"('{n_camelized}', 'Validation')"] * (1 - performance_drop)).to_dict()

        if scenario_name == 'o_to_po':
            x = load(
                os.path.join(
                    output_dir,
                    'ranking',
                    f'traffic_{n}_pruning_ratios_only_{config.prune_method.__name__}.csv'),
                index_col=0)

        if scenario_name == 'o_to_eo':
            x = load(
                os.path.join(
                    output_dir,
                    'ranking',
                    f'rank_{nn}_traffic_{n}_subsetsize_{subset_size}_scores.csv'),
                index_col=0)

        if scenario_name == 'o_to_ec':
            x = load(
                os.path.join(
                    output_dir,
                    'ranking',
                    f'rank_{nn}_traffic_{n}_combo_pruned_models_subsetsize_{subset_size}_{config.prune_method.__name__}_scores.csv')
                if prune_first else
                f'rank_{nn}_traffic_{n}_combo_subsetsize_{subset_size}_pruned_models_{config.prune_method.__name__}_scores.csv',
                index_col=0)
        accepted: pd.DataFrame = x[x[cols].apply(lambda x: x.between(mins[x.name], 1.)).all(axis=1)]
        col = x.loc[accepted.index]

        if col.empty:
            logger.info(f'No results within the performance drop {performance_drop}')
            return

        if scenario_name in ('o_to_po', 'o_to_ec'):
            col = col['Prune Ratio']
            idx = col[col == col.max()].index[-1]
            prune_ratio = x.loc[idx]['Prune Ratio']
        else:
            col = col['Global']
            idx = col[col == col.max()].index[-1]

        if scenario_name in ('o_to_eo', 'o_to_ec'):
            subset = load(
                os.path.join(
                    output_dir,
                    'ranking',
                    f'rank_{n}_traffic_{n}_combo_pruned_models_subsetsize_{subset_size}_{config.prune_method.__name__}_features.csv'
                    if scenario_name == 'o_to_ec' and prune_first else
                    f'rank_{n}_traffic_{n}_combo_subsetsize_{subset_size}_pruned_models_{config.prune_method.__name__}_features.csv'
                    if scenario_name == 'o_to_ec' and not prune_first else
                    f'rank_{n}_traffic_{n}_subsetsize_{subset_size}_features.csv'),
                index_col=0).loc[idx]
            subset = subset[subset.notnull()].index.values.tolist()

    dirname = os.path.join(
        output_dir, scenario_name + ('_few_c' if only_client_categories else '_all_c') +
        (f'_performance_drop_{performance_drop}' if performance_drop else '') +
        ((f'_prune_{prune_ratio}' if prune_ratio else '') if prune_first else (f'_subsetsize_{subset_size}' if subset_size is not None else '')) +
        ((f'_subsetsize_{subset_size}' if subset_size is not None else '') if prune_first else (f'_prune_{prune_ratio}' if prune_ratio else '')))

    if not os.path.isdir(dirname):
        create_dir(dirname)

    set_seed(seed)
    train, validation, test = config.get_dataset(seed=seed)

    if only_client_categories:
        set_seed(seed)
        train = train.filter_categories(config.client_categories).balance_categories()
        validation = validation.filter_categories(config.client_categories).balance_categories()

    for c in config.scenarios[scenario_name]:
        name = os.path.join(dirname, '-'.join(f"{k}_{v.name if hasattr(v, 'name') else v}" for k, v in c.items()))
        logger.info(f'Running scenario {name}')
        _run_test(config, name, train, validation, test, features_available=subset,
                  prune_ratio=prune_ratio, seed=seed, verbose=verbose, **c)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Path to configuration file', required=True, type=str)
    parser.add_argument('-s', '--seed', help='Random seed', type=int, default=42)
    parser.add_argument('-p', '--parallel', type=int, default=1,
                        help='Parallel Threads to be used in the underlying libraries')
    parser.add_argument('-v', '--verbose', help='Set Verbosity', action='store_true')
    parser.add_argument('-a', '--action', help='Action to perform', required=True,
                        type=str, choices=['dataset', 'correlations', 'train',
                                               'baselines', 'recursive_subset_search',
                                               'only_pruning', 'only_stochastic_search',
                                               'stochastic_search_then_pruning',
                                               'pruning_then_stochastic_search',
                                               'compare_pruning',
                                               'train_evaluation',
                                               'test_evaluation',
                                               'run_scenario'])
    parser.add_argument('-f', '--fixed-rank', help='Fixed Rank during ranking', action='store_true')
    parser.add_argument('-z', '--zero-first', help='Remove first feature with zero importance', action='store_true')
    parser.add_argument('-l', '--limited-traffic', help='Limit traffic to client data', action='store_true')
    parser.add_argument('-r', '--subset-ratio', help='Ratio of the subset to test', type=float, default=None)
    parser.add_argument('-n', '--scenario-name', help='Name of the scenario to run', type=str,
                        choices=['o_to_o', 'o_to_po', 'o_to_eo', 'o_to_ec'], default=None)
    parser.add_argument('-d', '--performance-drop', help='Performance Drop', type=float, default=None)
    parser.add_argument('-pr', '--prune-ratio', help='Prune ratio', type=float, default=None)
    parser.add_argument('-pf', '--prune-first', help='Prune first then subset', action='store_true')

    args = parser.parse_args()
    config = Config(**load(args.config))
    seed = args.seed
    action = args.action
    verbose = args.verbose
    parallel = args.parallel
    fixed = args.fixed_rank
    zero = args.zero_first
    limited = args.limited_traffic
    subset = args.subset_ratio
    scenario_name = args.scenario_name
    performance_drop = args.performance_drop
    prune_ratio = args.prune_ratio
    prune_first = args.prune_first

    threadpoolctl.threadpool_limits(limits=parallel)

    output_dir = os.path.join(os.path.dirname(args.config), config.output_path, f'seed_{seed}_output')

    if not os.path.isdir(output_dir):
        create_dir(output_dir)

    if action == 'dataset':
        datasetter(config, seed, verbose=verbose)
    elif action == 'correlations':
        correlations(config, seed, output_dir, verbose=verbose)
    elif action == 'train':
        train(config, seed, output_dir, verbose=verbose)
    elif action == 'compare_pruning':
        compare_pruning(config, seed, output_dir, verbose=verbose, ds_only_client_traffic=limited)
    elif action == 'recursive_subset_search':
        recursive_ss(config, seed, output_dir, verbose=verbose,
                     ds_only_client_traffic=limited, fixed_rank=fixed, remove_zero_first=zero)
    elif action == 'only_pruning':
        only_pruning(config, seed, output_dir, verbose=verbose, ds_only_client_traffic=limited)
    elif action == 'only_stochastic_search':
        only_stochastic_search(config, seed, output_dir, subset,
                               verbose=verbose,
                               ds_only_client_traffic=limited,
                               rank_only_client_traffic=limited)
    elif action == 'stochastic_search_then_pruning':
        stochastic_search_then_pruning(config, seed, output_dir, subset,
                                       verbose=verbose,
                                       ds_only_client_traffic=limited,
                                       rank_only_client_traffic=limited)
    elif action == 'pruning_then_stochastic_search':
        pruning_then_stochastic_search(config, seed, output_dir,
                                       subset, verbose=verbose,
                                       ds_only_client_traffic=limited,
                                       rank_only_client_traffic=limited)
    elif action == 'baselines':
        test_baselines(config, seed, output_dir, verbose=verbose)
    elif action == 'train_evaluation':
        train_evaluation(config, seed, output_dir, ds_only_client_traffic=limited,
                         prune_ratio=prune_ratio, verbose=verbose)
    elif action == 'test_evaluation':
        test_evaluation(config, seed, output_dir, ds_only_client_traffic=limited,
                        prune_ratio=prune_ratio, verbose=verbose)
    elif action == 'run_scenario':
        handle_scenario(config, seed, output_dir, scenario_name, verbose=verbose,
                        subset_size=subset, performance_drop=performance_drop,
                        only_client_categories=limited, rank_only_client_traffic=limited,
                        prune_first=prune_first)
    else:
        raise Exception(f'Unknown action {action}')


if __name__ == '__main__':
    main()
