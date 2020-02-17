#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Console script for pimkl."""
import sys
import click
from pimkl.cli.preprocess import preprocess_data_and_inducers
from pimkl.cli.analyse import analyse, kpca

@click.command()
@click.option('-fd', '--data_csv_file', 'data_csv_files', required=True, multiple=True, type=click.Path())
@click.option('-nd', '--data_name', 'data_names', required=True, multiple=True)
@click.argument('network_csv_file', required=True, type=click.Path())
@click.argument('network_name', required=True)
@click.argument('gene_sets_gmt_file', required=True, type=click.Path())
@click.argument('gene_sets_name', required=True)
@click.argument('preprocess_dir', required=True, type=click.Path(exists=True, file_okay=False, writable=True))
@click.argument('output_dir', required=True, type=click.Path(exists=True, file_okay=False, writable=True))
@click.argument('class_label_file', required=True, type=click.Path(exists=True, file_okay=True))
@click.option('--model_name', default='EasyMKL', type=click.Choice(['EasyMKL', 'UMKLKNN', 'AverageMKL']))
@click.argument('lam', default=0.2)
@click.argument('k', default=5)
@click.argument('number_of_folds', default=10)
@click.argument('max_per_class', default=20)
@click.argument('seed', default=0)
@click.argument('max_processes', default=1)
@click.argument('fold', default=-1)
def main(
    data_csv_files,
    data_names,
    network_csv_file,
    network_name,
    gene_sets_gmt_file,
    gene_sets_name,
    preprocess_dir,
    output_dir,
    class_label_file,
    model_name,
    lam,
    k,
    number_of_folds,
    max_per_class,
    seed,
    max_processes,
    fold,
):
    """Console script for a complete pimkl pipeline, including preprocessing
    and analysis. For more details consult the following console scripts, which
    are here executed in this order.
    `pimkl-preprocess --help`
    `pimkl-analyse run-performance-analysis --help`
    """

    preprocess_data_and_inducers(
    data_csv_files, data_names,
    network_csv_file, network_name,
    gene_sets_gmt_file, gene_sets_name,
    preprocess_dir,
    match_samples=True)


    output_filename_core = analyse(
    data_names, network_name, gene_sets_name,
    preprocess_dir, output_dir,
    class_label_file,
    model_name, lam, k,
    number_of_folds, max_per_class,
    seed, max_processes
    )

    weights_csv_file = '{}/weights_{}.csv'.format(
            output_dir, output_filename_core
        )

    # NOTE: the kernel PCA at the moment can be run in a separate step
    # kpca(
    #     data_names,
    #     network_name,
    #     gene_sets_name,
    #     preprocess_dir,
    #     output_dir,
    #     class_label_file,
    #     weights_csv_file,
    #     fold,
    # )

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
