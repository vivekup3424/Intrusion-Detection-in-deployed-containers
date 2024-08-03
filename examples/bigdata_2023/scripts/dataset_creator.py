import os
from multiprocessing.pool import ThreadPool

import click
import numpy as np
import pandas as pd
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils.common import (create_dir, get_default_cpu_number, get_logger,
                          set_seed)

logger = get_logger(f"{os.path.basename(__file__).replace('.py', '')}")


def get_dataset_parameters(filename):
    if 'CICIDS2017' in filename:
        return 'Label', 'BENIGN', ['Destination Port']
    if 'CICIDS2019' in filename:
        return 'Label', 'BENIGN', ['Unnamed: 0', 'Flow ID', 'Source IP', 'Source Port',
                                   'Destination IP', 'Destination Port', 'Timestamp', 'SimillarHTTP']
    if 'CICIOT2023' in filename:
        return 'label', 'BenignTraffic', ['Protocol Type', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH',
                                          'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC']
    if 'EDGE2022' in filename:
        return 'Attack_type', 'Normal', [
            'frame.time', 'ip.src_host', 'ip.dst_host', 'arp.src.proto_ipv4', 'arp.dst.proto_ipv4', 'tcp.srcport',
            'tcp.dstport', 'udp.port', 'http.tls_port', 'tcp.options', 'http.file_data', 'tcp.payload', 'mqtt.msg',
            'http.request.uri.query', 'http.request.full_uri', 'icmp.transmit_timestamp', 'Attack_label']
    if 'ICS-D1' in filename:
        return 'marker', 'Natural', []
    if 'ICS-D2' in filename:
        return 'Label', 'Good', []
    if 'ICS-D3' in filename:
        return 'result', 0, ['time']
    raise ValueError(filename)


def create(
        directory: str, test_size: float, finetune_size: float, validation_size: float,
        composition: str, problem_type: str, cpus: int):
    global logger
    set_seed()
    is_balanced = composition == 'balanced'
    is_binary = problem_type == 'binary'

    store_path = os.path.join(directory, os.pardir, f'{composition}_{problem_type}')
    create_dir(store_path, overwrite=False)

    logger.info(f'Setting variable for dataset {directory}')
    type_column, benign_label, excluded_columns = get_dataset_parameters(directory)

    files = [os.path.join(directory, x)
             for x in os.listdir(directory) if x.endswith('.csv')]

    logger.info(f'Loading {len(files)=} dataframes')
    df = pd.DataFrame()
    with ThreadPool(processes=cpus) as pool, tqdm(total=len(files)) as pbar:
        for x in pool.imap(pd.read_csv, files):
            df = pd.concat((df, x), ignore_index=True)
            pbar.update()

    logger.info(f'Dropping columns {excluded_columns=}')
    df.drop(excluded_columns, axis=1, errors='ignore', inplace=True)

    logger.info(f'Dropping columns with only 1 value')
    df.dropna(thresh=2, axis=1, inplace=True)

    logger.info(f"Renaming {type_column=} to 'Label'")
    df.rename({type_column: 'Label'}, axis=1, inplace=True)
    type_column = 'Label'

    if is_binary:
        logger.info(f'Binary problem, changing to 1 all columns different from {benign_label=}')
        df[type_column] = df[type_column].ne(benign_label).mul(1)

    logger.info(f'Transforming columns to numeric')
    df = df.apply(pd.to_numeric, errors='ignore')

    cat_columns = [col_name for col_name,
                   dtype in df.dtypes.items() if dtype == object and col_name != type_column]
    if cat_columns:
        logger.info(f'Transforming {cat_columns=} categorical columns also to numeric')
        df[cat_columns] = df[cat_columns].astype('category')
        df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
        df[cat_columns] = df[cat_columns].astype('int')

    logger.info(f'Removing rows with at NaN values')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=0, how='any', inplace=True)

    logger.info(f'Min-Max normalisation')
    df = df.apply(
        lambda x: (x - x.min()) / (x.max() - x.min()))

    if is_balanced:
        logger.info(f'Balanced problem, picking a maximum samples of the minor class')
        g = df.groupby(type_column)
        df = g.apply(lambda x: x.sample(g.size().min())).reset_index(drop=True)

    a = AutoMLPipelineFeatureGenerator()
    df = a.fit_transform(df)

    logger.info('First, splitting dataset into train and test')
    big_train, big_test, _, _ = train_test_split(df, df[type_column],
                                                 test_size=test_size,
                                                 shuffle=True,
                                                 stratify=df[type_column])
    logger.info('Now sub-splitting train into train and validation')
    train_data, validation_data, _, _ = train_test_split(big_train, big_train[type_column],
                                                         test_size=validation_size,
                                                         shuffle=True,
                                                         stratify=big_train[type_column])

    logger.info('Now sub-splitting test into test and finetune')
    test_data, finetune_data, _, _ = train_test_split(big_test, big_test[type_column],
                                                      test_size=finetune_size,
                                                      shuffle=True,
                                                      stratify=big_test[type_column])

    with tqdm(total=4) as pbar:
        for x, y in zip((train_data, validation_data, test_data, finetune_data),
                        ('train', 'validation', 'test', 'finetune')):
            x: pd.DataFrame
            logger.info(f'Storing {y=} with label value counts {x[type_column].value_counts()=}')
            x.index.name = 'ID'
            x.to_csv(os.path.join(store_path, f'{y}.csv'))
            pbar.update()


@click.command(help='Create and adjust the final dataset', context_settings={'show_default': True}, name='create')
@click.option('--directory', type=str, required=True,
              help='working directory with the original dataset')
@click.option('--test-size', type=float, default=0.3,
              help='size of the test portion (the remaining is left for training)')
@click.option('--finetune-size', type=float, default=0.5,
              help='size of the finetune portion inside the test one')
@click.option('--validation-size', type=float, default=0.2,
              help='size of the validation portion inside the training one')
@click.option('--composition', type=click.Choice(['balanced', 'imbalanced'], case_sensitive=False),
              help='whether to be a balanced or imbalanced problem', default='balanced')
@click.option('--problem-type', type=click.Choice(['binary', 'multiclass'], case_sensitive=False),
              help='set the problem to binary classification', default='binary')
@click.option('--cpus', type=int, default=get_default_cpu_number(),
              help='number of CPU cores to assign')
def create_command(*args, **kwargs):
    create(*args, **kwargs)


if __name__ == '__main__':
    create_command()
