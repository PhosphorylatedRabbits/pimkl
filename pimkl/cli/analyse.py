import glob
import logging
import os
import sys
import traceback
from functools import partial, reduce
from multiprocessing import Pool, cpu_count

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import KernelPCA
from tqdm import tqdm

from pimkl.analysis import significant_pathways
from pimkl.inducers import read_inducer
from pimkl.models import PIMKL
from pimkl.run import fold_generator, run_model
from pimkl.utils.preprocessing import Standardizer

# plotting setup
sns.set_palette(sns.color_palette('colorblind'))
sns.set_style('white')
sns.set_context('talk')
significant_color = sns.color_palette('colorblind')[2]
other_color = sns.color_palette('colorblind')[0]


def read_preprocessed(
    data_names, network_name, gene_sets_name, preprocess_dir
):
    inducers_filenames = {}
    inducers_names = {}
    inducers = {}
    data = {}
    for data_name in data_names:
        # data
        data[data_name] = pd.read_csv(
            os.path.join(
                preprocess_dir, '{}_{}_{}_{}.csv'.format(
                    'data', data_name, network_name, gene_sets_name
                )
            ),
            index_col=0
        )
        # inducers
        inducers_filenames[data_name] = glob.glob(
            os.path.join(
                preprocess_dir,
                '{}_{}_{}_*.csv.gz'.format(
                    'inducer', data_name, network_name
                    # * matches gene set (inducer) name which can contain "_"
                )
            )
        )
        inducers_names[data_name] = [
            '_'.join(os.path.basename(filename).split('.')[0].split('_')[4:])
            for filename in inducers_filenames[data_name]
        ]
        assert len(inducers_names[data_name]
                   ) == len(set(inducers_names[data_name]))
        inducers[data_name] = [
            read_inducer(filename, size=data[data_name].shape[1])
            for filename in inducers_filenames[data_name]
        ]
        inducers_extended_names = [
            '{}-{}'.format(data_name, name)
            for data_name, inducer_names in inducers_names.items()
            for name in inducer_names
        ]
    return data, inducers, inducers_extended_names


def analyse(
    data_names, network_name, gene_sets_name,
    preprocess_dir, output_dir,
    class_label_file,
    model_name='EasyMKL', lam=.2, k=5,
    number_of_folds=2, max_per_class=20,
    seed=0, max_processes=cpu_count()
):
    # reproducible results
    np.random.seed(seed)

    # model parameters
    regularization_factor = False
    kernel_normalization = True

    estimator_parameters = {
        'trace_normalization': kernel_normalization,
        'regularization_factor': regularization_factor,
        'lam': lam
    }
    if model_name == 'UMKLKNN':
        mkl_parameters = {'trace_normalization': kernel_normalization, 'k': k}
    elif model_name == 'AverageMKL':
        mkl_parameters = {'trace_normalization': kernel_normalization}
    else:  # EasyMKL
        mkl_parameters = estimator_parameters

    # prepare: read data and inducers
    data, inducers, inducers_extended_names = read_preprocessed(
        data_names, network_name, gene_sets_name, preprocess_dir
    )

    # prepare: classification labels
    class_labels = pd.read_csv(
        class_label_file, index_col=0, header=None, squeeze=True
    )
    class_labels = class_labels[~pd.isna(class_labels)]

    # match samples in data and labels
    measurement_data_samples = sorted(
        list(
            reduce(
                lambda a, b: a & b,
                (set(data[data_name].index) for data_name in data_names)
            )
        )
    )
    samples = sorted(
        list(set(measurement_data_samples) & set(class_labels.index))
    )

    labels = (
        class_labels[samples].values if model_name == 'EasyMKL' else
        None  # TODO keep series, check model later
    )
    for data_name in data_names:
        data[data_name] = data[data_name].loc[samples].values

    # no more pandas labels
    # learn support vector and kernel weights for different data splits

    all_trace_factors = {}
    all_aucs = {}
    all_weights = {}

    # parallel
    processes = max_processes if max_processes < number_of_folds else number_of_folds  # noqa

    if (processes == 1) or (
        logging.root.level <= logging.DEBUG
    ):  # serial, allows easier debugging
        for fold_parameters in tqdm(
            fold_generator(number_of_folds, data, labels, max_per_class)
        ):
            try:
                aucs, weights, trace_factors = run_model(
                    inducers=inducers,
                    induction_name="induce_linear_kernel",
                    mkl_name=model_name,
                    estimator_name="EasyMKL",
                    mkl_parameters=estimator_parameters,
                    estimator_parameters=estimator_parameters,
                    induction_parameters={},
                    inducers_extended_names=inducers_extended_names,
                    fold_parameters=fold_parameters
                )
                all_trace_factors[fold_parameters['fold']] = trace_factors
                all_aucs[fold_parameters['fold']] = aucs
                if isinstance(weights, list):
                    for label, weights_per_label in weights:
                        all_weights[(fold_parameters['fold'], label)
                                    ] = weights_per_label
                else:
                    all_weights[fold_parameters['fold']] = weights
            except TypeError:  # run returned None
                logging.debug(
                    'fold {} not appended'.format(fold_parameters['fold'])
                )
                traceback.print_exc()
    else:  # parallel
        run_fold = partial(
            run_model, inducers, 'induce_linear_kernel', model_name, 'EasyMKL',
            mkl_parameters, estimator_parameters, {}, inducers_extended_names
        )
        logging.debug('fold runs start')
        with Pool(processes=processes) as pool:
            runner = pool.imap(
                run_fold,
                fold_generator(number_of_folds, data, labels, max_per_class)
            )
            logging.debug('lazy iterator created')
            results = list(runner)
            logging.debug('fold runs done')

        results = filter(lambda x: x is not None, results)  # a generator
        for i, (aucs, weights, trace_factors) in enumerate(results):
            all_trace_factors[i] = trace_factors
            all_aucs[i] = aucs
            if isinstance(weights, list):
                for label, weights_per_label in weights:
                    all_weights[(i, label)] = weights_per_label
            else:
                all_weights[i] = weights

    # preparing output
    aucs_df = pd.DataFrame(all_aucs).T
    weights_df = pd.DataFrame(all_weights).T
    trace_factors_df = pd.DataFrame(all_trace_factors).T
    output_filename_part = '{}_{}_{}_cv={}_mc={}_{}'.format(
        '-'.join(data_names), network_name, gene_sets_name, number_of_folds,
        max_per_class, model_name
    )
    print(aucs_df)

    # write to file
    aucs_df.to_csv('{}/auc_{}.csv'.format(output_dir, output_filename_part))
    weights_df.to_csv(
        '{}/weights_{}.csv'.format(output_dir, output_filename_part)
    )
    trace_factors_df.to_csv(
        '{}/tracefactors_{}.csv'.format(output_dir, output_filename_part)
    )

    # visualize weights
    inducers_ordering = weights_df.median().sort_values(ascending=False).index
    plt.close()

    colors = [
        significant_color if is_significant else other_color for is_significant
        in significant_pathways(weights_df[inducers_ordering])
    ]
    sns.boxplot(data=weights_df[inducers_ordering], palette=colors)
    plt.xlabel('Pathway')
    plt.ylabel('Weight')
    _ = plt.xticks(rotation=90, fontsize=8)
    plt.axhline(y=1. / weights_df.shape[1], lw=1., ls='--', c='black')

    plt.savefig(
        '{}/weights_{}.pdf'.format(output_dir, output_filename_part),
        bbox_inches='tight'
    )
    plt.close()

    # visualize auc
    aucs_df_box = aucs_df.melt(value_name='AUC')
    sns.boxplot(data=aucs_df_box, y='AUC', x='variable')

    plt.savefig(
        '{}/aucs_{}.pdf'.format(output_dir, output_filename_part),
        bbox_inches='tight'
    )
    plt.close()

    print("Files *_{}.* written to disk".format(output_filename_part))
    return output_filename_part


def kpca(
    data_names,
    network_name,
    gene_sets_name,
    preprocess_dir,
    output_dir,
    class_label_file,
    weights_csv_file,
    fold,
):
    # model parameters
    kernel_normalization = True

    # prepare: read data and inducers
    data, inducers, inducers_extended_names = read_preprocessed(
        data_names, network_name, gene_sets_name, preprocess_dir
    )

    # prepare: classification labels
    # if class_label_file:
    class_labels = pd.read_csv(
        class_label_file, index_col=0, header=None, squeeze=True
    )
    class_labels = class_labels[~pd.isna(class_labels)]

    # match samples in data and labels
    measurement_data_samples = sorted(
        list(
            reduce(
                lambda a, b: a & b,
                (set(data[data_name].index) for data_name in data_names)
            )
        )
    )
    samples = sorted(
        list(set(measurement_data_samples) & set(class_labels.index))
    )

    labels = (class_labels[samples].values)
    for data_name in data_names:
        data[data_name] = data[data_name].loc[samples].values

    # read learned weight to compute final kernel on all data
    weights = pd.read_csv(weights_csv_file, index_col=0)
    kpca_basename = os.path.basename(weights_csv_file).split('.')[0]

    if fold == -1:
        weights = weights.median()
        fold_name = 'median'
    else:
        weights = weights.loc[fold, :]
        fold_name = 'fold{}'.format(fold)

    # kernel pca
    model_trained_weights = PIMKL(
        inducers=inducers,
        induction='induce_linear_kernel',
        mkl='WeightedAverageMKL',
        mkl_parameters={
            'trace_normalization': kernel_normalization,
            'kernels_weights': weights
        }
    )
    model_trained_weights.fit(data)
    optimal_kernel = model_trained_weights.predict(data)

    kernel_pca = KernelPCA(kernel='precomputed'
                           ).fit(Standardizer().apply(optimal_kernel))
    transformed_data = kernel_pca.transform(optimal_kernel)

    # # not really meaningful for KernelPCA
    # explained_variance = np.var(transformed_data, axis=0)
    # explained_variance_ratio = explained_variance / explained_variance.sum()
    # plt.plot(np.cumsum(explained_variance_ratio))
    # plt.xlabel('KernelPCA components')
    # plt.ylabel('Explained Variance Ratio')
    # plt.ylim((explained_variance_ratio[0], 1.0))

    # plt.savefig(
    #     '{}/kernel_pca_explained_variance_{}_{}.pdf'.format(
    #         output_dir, fold_name, kpca_basename
    #     ),
    #     bbox_inches='tight'
    # )

    components = 3
    kpca_columns = list(
        map(lambda index: 'KernelPC{}'.format(index + 1), range(components))
    )
    kernel_pca_signature = pd.DataFrame(
        transformed_data[:, :components], index=samples, columns=kpca_columns
    )
    kernel_pca_signature['class'] = labels

    plt.clf()
    sns.pairplot(
        kernel_pca_signature,
        kind='scatter',
        hue='class',
        vars=kpca_columns,
        plot_kws=dict(s=10, edgecolor='darkgrey', linewidth=1)
    )
    plt.legend()
    plt.savefig(
        '{}/kernel_pca_signature_{}_{}_{}.pdf'.format(
            output_dir, components, fold_name, kpca_basename
        ),
        bbox_inches='tight'
    )


@click.group()
def main():
    pass


@click.command(short_help='train and test many folds')
@click.option('-nd', '--data_name', 'data_names', required=True, multiple=True)
@click.argument('network_name', required=True)
@click.argument('gene_sets_name', required=True)
@click.argument(
    'preprocess_dir',
    required=True,
    type=click.Path(exists=True, file_okay=False)
)
@click.argument(
    'output_dir',
    required=True,
    type=click.Path(exists=True, file_okay=False, writable=True)
)
@click.argument(
    'class_label_file',
    required=True,
    type=click.Path(exists=True, file_okay=True)
)
@click.option(
    '--model_name',
    default='EasyMKL',
    type=click.Choice(['EasyMKL', 'UMKLKNN', 'AverageMKL'])
)
@click.argument('lam', default=0.2)
@click.argument('k', default=5)
@click.argument('number_of_folds', default=10)
@click.argument('max_per_class', default=20)
@click.argument('seed', default=0)
@click.argument('max_processes', default=1)
def run_performance_analysis(
    data_names, network_name, gene_sets_name, preprocess_dir, output_dir,
    class_label_file, model_name, lam, k, number_of_folds, max_per_class, seed,
    max_processes
):
    """
    Run classifications using pathway induced multiple kernel learning on
    preprocessed data and inducers on a number of train/test splits and analyse
    the resulting classification performance and learned pathway weights.

    The `class_label_file` should be readable with `pd.read_csv(
        class_label_file, index_col=0, header=None, squeeze=True)`
    """
    output_filename_core = analyse(
        data_names, network_name, gene_sets_name, preprocess_dir, output_dir,
        class_label_file, model_name, lam, k, number_of_folds, max_per_class,
        seed, max_processes
    )
    return 0



@click.command(short_help='KernelPCA with given pathway weights')
@click.option('-nd', '--data_name', 'data_names', required=True, multiple=True)
@click.argument('network_name', required=True)
@click.argument('gene_sets_name', required=True)
@click.argument(
    'preprocess_dir',
    required=True,
    type=click.Path(exists=True, file_okay=False)
)
@click.argument(
    'output_dir',
    required=True,
    type=click.Path(exists=True, file_okay=False, writable=True)
)
@click.argument(
    'class_label_file',
    required=True,
    type=click.Path(exists=True, file_okay=True)
)
@click.argument(
    'weights_csv_file',
    required=True,
    type=click.Path(exists=True, file_okay=True)
)
@click.argument('fold', default=-1)
def run_kpca(
    data_names,
    network_name,
    gene_sets_name,
    preprocess_dir,
    output_dir,
    class_label_file,
    weights_csv_file,
    fold,
):
    """
    Following pathway weight computation during performance analysis, perform
    KernelPCA on a final kernel defined by weights of either a given fold or by
    default the median pathway weight.
    """
    kpca(
        data_names,
        network_name,
        gene_sets_name,
        preprocess_dir,
        output_dir,
        class_label_file,
        weights_csv_file,
        fold,
    )

    return 0


main.add_command(run_performance_analysis)
main.add_command(run_kpca)

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
