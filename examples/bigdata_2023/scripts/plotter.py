import os

import click
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt
from utils.common import DEFAULT_PLOT_STYLE
from utils.pruning import get_pruning_algorithms


def plot_rfe(directory: str, metric: str):
    rfe = pd.read_csv(os.path.join(directory, 'leaderboard.csv'), index_col='ID')
    rfe['ID'] = rfe.index.values

    baseline_general = pd.read_csv(
        os.path.join(directory, os.pardir, 'leaderboard.csv'),
        index_col='ID').loc['finetune'][metric]

    with plt.style.context(DEFAULT_PLOT_STYLE):
        fig, ax = plt.subplots()

        rfe.plot.scatter(x='ID', y=metric, ax=ax, alpha=0.7, s=5)
        ax.invert_xaxis()
        ax.set_ylim(0, 1.05)
        ax.hlines(y=baseline_general, xmin=rfe.ID.min(), xmax=rfe.ID.max(), color='r', linestyle='--', label='Baseline')
        ax.set_ylabel(metric.capitalize())
        ax.set_xlabel('# Active features')
        ax.grid(axis='y')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(directory, f'leaderboard_{metric}.pdf'))


def plot_ss(directory: str, metric: str):
    rfe = pd.read_csv(os.path.join(directory, os.pardir, 'leaderboard.csv'), index_col='ID')
    max_features = rfe.index.values[0]

    sizes_used = []
    ss = pd.DataFrame()

    for x in os.listdir(directory):
        if not os.path.isdir(x) and not x.startswith('feature_subsets_'):
            continue
        tmp = pd.read_csv(os.path.join(directory, x, 'leaderboard.csv'), index_col='ID')
        kind = x.replace('feature_subsets_', '')
        size = round(float(kind[:-1]) * max_features) or 1
        tmp['kind'] = kind
        sizes_used.append(size)
        ss = pd.concat((ss, tmp))

    baseline_general = pd.read_csv(
        os.path.join(directory, os.pardir, os.pardir, 'leaderboard.csv'),
        index_col='ID').loc['finetune'][metric]
    sizes_used = sorted(sizes_used, reverse=False)
    ss.sort_values('kind', inplace=True, ascending=False)

    cmap = cm.get_cmap('rainbow', len(ss.kind.unique()))
    with plt.style.context(DEFAULT_PLOT_STYLE):
        fig, ax = plt.subplots()

        ax.plot(range(1, 1 + len(ss.kind.unique())), rfe.loc[sizes_used, metric].tolist(),
                color='g', linestyle='-.', label='Baseline RFE')

        bp = ss.plot.box(column=metric, by='kind', patch_artist=True, notch=True, ax=ax, return_type='both',
                         whiskerprops={'color': 'black'}, capprops={'color': 'black'}, medianprops={'color': 'blue'})

        [box.set(color='black', facecolor=cmap(i), alpha=0.8)
         for _, (__, row) in bp.items() for i, box in enumerate(row['boxes'])]

        ax.invert_xaxis()
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric.capitalize())
        ax.set_xlabel('Subset size ratio')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.grid(axis='y')
        ax.hlines(y=baseline_general, xmin=1, xmax=len(ss.kind.unique()), color='r', linestyle='--', label='Baseline')
        ax.legend()
        ax.set_title('')
        fig.tight_layout()
        fig.savefig(os.path.join(directory, f'leaderboard_{metric}.pdf'))


def plot_mp(directory: str, metric: str, subsets_dir: str):
    files = [os.path.join(directory, x, 'leaderboard.csv') for x in os.listdir(directory)
             if os.path.isdir(os.path.join(directory, x))]
    file_names = [os.path.basename(os.path.dirname(fname)) for fname in files]
    save_name = directory
    files2, files_names2 = [], []
    if subsets_dir:
        for x in os.listdir(subsets_dir):
            tmp = os.path.join(subsets_dir, x)
            if not os.path.isdir(tmp):
                continue
            for y in os.listdir(tmp):
                tmp2 = os.path.join(tmp, y)
                if not os.path.isdir(tmp2) or not tmp2.endswith('_for_subset'):
                    continue
                files_names2.append(y + '_' + x.split('_')[-1])
                files2.append(os.path.join(tmp2, 'leaderboard.csv'))
    if files2:
        save_name = subsets_dir
        files += files2
        file_names += files_names2

    df = pd.DataFrame()
    for z in files:
        tmp = pd.read_csv(z, index_col='ID')
        tmp['kind'] = ''.join([x[0] if len(x) > 2 else x for x in os.path.basename(
            os.path.dirname(z)).split('_')]).upper()
        df = pd.concat((df, tmp))

    baseline_general = pd.read_csv(
        os.path.join(directory, os.pardir, 'leaderboard.csv'),
        index_col='ID').loc['finetune'][metric]

    save_name = os.path.join(save_name, f'leaderboard_search_{metric}.pdf')
    df.sort_values('kind', inplace=True, ascending=False)
    df = df.reset_index(drop=True)
    cmap = cm.get_cmap('rainbow', len(df.kind.unique()))

    with plt.style.context(DEFAULT_PLOT_STYLE):
        fig, ax = plt.subplots()

        bp = df.plot.box(column=metric, by='kind', patch_artist=True, ax=ax, return_type='both',
                         whiskerprops={'color': 'black'}, capprops={'color': 'black'}, medianprops={'color': 'blue'})
        [box.set(color='black', facecolor=cmap(i), alpha=0.8)
         for _, (__, row) in bp.items() for i, box in enumerate(row['boxes'])]

        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric.capitalize())
        ax.set_xlabel('Pruning algorithm')
        ax.grid(axis='y')
        ax.hlines(y=baseline_general, xmin=1, xmax=len(df.kind.unique()), color='r', linestyle='--', label='Baseline')
        ax.legend()
        ax.set_title('')
        fig.tight_layout()
        fig.savefig(save_name)

    files = [x.replace('leaderboard.csv', 'models_stats.csv') for x in files]
    df_stats = pd.DataFrame(columns=['global_sparsity', 'kind'])
    for z in files:
        tmp = pd.read_csv(z, index_col='ID')
        metric = tmp.columns.values[0]
        df_stats = pd.concat((df_stats, tmp), join='inner')
    df_stats = df_stats.reset_index(drop=True)
    df_stats['kind'] = df['kind']
    df_stats['global_sparsity'] = 1 - df_stats['global_sparsity']
    save_name = os.path.join(os.path.dirname(save_name), 'models_stats.pdf')
    cmap = cm.get_cmap('rainbow', len(df_stats.kind.unique()))

    with plt.style.context(DEFAULT_PLOT_STYLE):
        fig, ax = plt.subplots()

        bp = df_stats.plot.box(column='global_sparsity', by='kind', patch_artist=True, ax=ax, return_type='both',
                               whiskerprops={'color': 'black'}, capprops={'color': 'black'},
                               medianprops={'color': 'blue'})
        [box.set(color='black', facecolor=cmap(i), alpha=0.8)
         for _, (__, row) in bp.items() for i, box in enumerate(row['boxes'])]

        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Global density')
        ax.set_xlabel('Pruning algorithm')
        ax.grid(axis='y')
        ax.hlines(y=1, xmin=1, xmax=len(df_stats.kind.unique()), color='r', linestyle='--', label='Baseline')
        ax.legend()
        ax.set_title('')
        fig.tight_layout()
        fig.savefig(save_name)


def plot_mt(directory: str, algorithm: str, metric: str, degradation: float):
    files = [os.path.join(directory, x, f'leaderboard_{algorithm}.csv') for x in os.listdir(directory)
             if os.path.isdir(os.path.join(directory, x)) and x.startswith('feature_subsets_')]

    df = pd.DataFrame()
    for z in files:
        tmp = pd.read_csv(z, index_col='ID')
        tmp['kind'] = os.path.basename(os.path.dirname(z)).replace('feature_subsets_', '').replace('s', '')
        df = pd.concat((df, tmp))

    baseline_general = pd.read_csv(
        os.path.join(directory, os.pardir, os.pardir, 'leaderboard.csv'),
        index_col='ID').loc['finetune'][metric]
    df.drop(['degradation_at_subset', 'degradation_at_rank', 'degradation_at_model'],
            inplace=True, axis=1, errors='ignore')
    cmap = cm.get_cmap('rainbow', len(df.kind.unique()))

    with plt.style.context(DEFAULT_PLOT_STYLE):
        fig, ax = plt.subplots()

        bp = df.plot.box(column=metric, by='kind', patch_artist=True, notch=True, ax=ax, return_type='both',
                         whiskerprops={'color': 'black'}, capprops={'color': 'black'}, medianprops={'color': 'blue'})

        [box.set(color='black', facecolor=cmap(i), alpha=0.8)
         for _, (__, row) in bp.items() for i, box in enumerate(row['boxes'])]

        ax.invert_xaxis()
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric.capitalize())
        ax.set_xlabel('Subset size ratio')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.grid(axis='y')
        ax.hlines(y=baseline_general, xmin=1, xmax=len(df.kind.unique()), color='r', linestyle='--', label='Baseline')
        ax.legend()
        ax.set_title('')
        fig.tight_layout()
        fig.savefig(os.path.join(directory, f'leaderboard_{metric}_{algorithm}.pdf'))

    if '_for_subsets' in algorithm:
        models_stats = pd.DataFrame()
        for f in files:
            tmp = pd.read_csv(os.path.join(os.path.dirname(f), 'models_stats.csv'), index='ID')
            models_stats = pd.concat((models_stats, tmp))
        models_stats.reset_index(drop=True, inplace=True)
        score_with_pruning_amounts = df.join(models_stats, on=['model_ID', 'subset_ID'], how='left')
    else:
        models_stats = pd.read_csv(os.path.join(directory, os.pardir, os.pardir,
                                                'prune_search', algorithm, 'models_stats.csv'), index_col='ID')
        score_with_pruning_amounts = df.join(models_stats, on='model_ID', how='left')

    if 'locally' in algorithm:
        import ast
        score_with_pruning_amounts['amount'] = score_with_pruning_amounts['amount'].apply(
            lambda x: round(sum(x for x in ast.literal_eval(x)) / len(ast.literal_eval(x)), 2))

    with plt.style.context(DEFAULT_PLOT_STYLE):
        sorted_by_index = score_with_pruning_amounts.sort_index(ascending=False)
        sorted_by_index['kind'] = pd.to_numeric(sorted_by_index['kind'])

        fig = plt.figure()
        ax: plt.Axes = fig.add_subplot(111, projection='3d')

        sctt = ax.scatter3D(sorted_by_index['kind'], sorted_by_index['amount'], sorted_by_index[metric],
                            alpha=0.8, s=15,
                            c=sorted_by_index[metric],
                            cmap=cm.jet,
                            marker='.', rasterized=True)
        fig.colorbar(sctt, label=metric.capitalize(), ax=ax, shrink=0.5, aspect=12, pad=0.09, location='right')
        ax.tick_params(axis='both', which='major', labelsize=6, pad=-4)
        ax.set_facecolor('white')
        ax.view_init(30, -60, 0)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_zlim(0, 1)
        ax.invert_yaxis()
        ax.tick_params(axis='y', which='minor', left=False, right=False)
        ax.invert_xaxis()
        ax.minorticks_off()
        ax.set_ylabel('Pruning ratio', labelpad=-9.2)
        ax.set_xlabel('Subset size ratio', labelpad=-9.2)
        ax.set_zlabel(f'Combination Accuracy', labelpad=-9.2)
        fig.tight_layout()
        fig.savefig(os.path.join(directory, f'leaderboard3D_{metric}_{algorithm}.pdf'))
    score_with_pruning_amounts.loc[baseline_general - score_with_pruning_amounts[metric] > degradation, 'amount'] = None
    with plt.style.context(DEFAULT_PLOT_STYLE):

        fig, ax = plt.subplots()
        bp = score_with_pruning_amounts.plot.box(
            column='amount', by='kind', patch_artist=True, notch=True, ax=ax, return_type='both',
            whiskerprops={'color': 'black'}, capprops={'color': 'black'}, medianprops={'color': 'blue'})

        cmap = cm.get_cmap('rainbow', len(score_with_pruning_amounts.kind.unique()))

        [box.set(color='black', facecolor=cmap(i), alpha=0.8)
         for _, (_, row) in bp.items() for i, box in enumerate(row['boxes'])]
        ax.invert_xaxis()
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Pruning ratio')
        ax.set_xlabel('Subset size ratio')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.grid(axis='y')
        ax.set_title('')
        fig.tight_layout()
        fig.savefig(os.path.join(directory, f'subsets_pruning_below_10_degradation_{metric}_{algorithm}.pdf'))


@click.group()
@click.pass_context
def main(ctx):
    pass


@main.command(help='Plot the result of the ranking', context_settings={'show_default': True}, name='plot_rfe')
@click.option('--directory', type=str, help='working directory with the automl chosen model', required=True)
@click.option('--metric', type=str, help='metric to be plot from the evaluation', default='accuracy')
def plot_rfe_command(*args, **kwargs):
    plot_rfe(*args, **kwargs)


@main.command(help='Plot the result of the subset search tested on the base model',
              context_settings={'show_default': True}, name='plot_ss')
@click.option('--metric', type=str, help='metric to be plot from the evaluation', default='accuracy')
@click.option('--directory', type=str, help='working directory with the greedy ranking of the features', required=True)
def plot_ss_command(*args, **kwargs):
    plot_ss(*args, **kwargs)


@main.command(help='Test the pruned models obtained',
              context_settings={'show_default': True}, name='plot_mp')
@click.option('--directory', type=str, help='path to a result of a model pruning', required=True)
@click.option('--metric', type=str, help='metric to be plot from the evaluation', default='accuracy')
@click.option('--subsets-dir', type=str, help='path to the pruned models specific for subsets', default='')
def plot_mp_command(*args, **kwargs):
    return plot_mp(*args, **kwargs)


@main.command(help='Test the combination of pruned models and subsets',
              context_settings={'show_default': True}, name='plot_mt')
@click.option('--directory', type=str, help='path to a result of a model pruning', required=True)
@click.option('--algorithm', type=click.Choice(get_pruning_algorithms(), case_sensitive=False),
              help='pruning algorithm to be plot', required=True)
@click.option('--metric', type=str, help='metric to be plot from the evaluation', default='accuracy')
@click.option('--degradation', type=float, help='degradation ratio with respect to baseline', default=0.10)
def plot_mt_command(*args, **kwargs):
    return plot_mt(*args, **kwargs)


if __name__ == '__main__':
    main()
