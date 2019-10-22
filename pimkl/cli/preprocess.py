# -*- coding: utf-8 -*-
"""Main module."""
import sys
import click
import pandas as pd
from pimkl.network import get_network_from_csv
from pimkl.inducers import (
    read_gmt, get_matching_data_and_network, write_preprocessed
)


def invalid_name(name):
    return '_' in name or '-' in name


def assert_valid_names(*names):
    for name in names:
        if invalid_name(name):
            raise IOError(
                'names cannot contain "_" or "-", correct "{}"'.format(name)
            )


def reader(filename):
    return pd.read_csv(filename, sep=',', index_col=0).T


def read_data(filename, reader, gene_name_transformation=None):
    data = reader(filename)
    data.index.name = 'samples'
    data.columns.name = 'gene_name'
    if gene_name_transformation is not None:
        data.columns = [
            gene_name_transformation(gene_name) for gene_name in data.columns
        ]
    return data


def preprocess_data_and_inducers(
    data_csv_files,
    data_names,
    network_csv_file,
    network_name,
    gene_sets_gmt_file,
    gene_sets_name,
    preprocess_dir,
    match_samples
):
    """
    Inducers, that is Laplacian matrices for geneset subnetworks, and data are
    preprocessed and written to file.
    Data and inducers are filtered for genes (per dataset) available in the
    data and the network and the union of genesets.
    Conditionally, also the data is filtered for matching samples over all
    datasets.
    """
    assert_valid_names(*data_names, network_name, gene_sets_name)

    data_dict = {
        data_name: read_data(filepath, reader)
        for data_name, filepath in zip(data_names, data_csv_files)
        # TODO test data no list
    }

    # filter for matching samples
    if match_samples:
        samples_set = None
        for data_name, data in data_dict.items():
            print(data.shape)
            if samples_set is None:
                samples_set = set(data.index)
            else:
                samples_set &= set(data.index)
            print(data_name, data.shape)
        matching_samples = sorted(list(samples_set))
        for _, data in data_dict.items():
            data = data.loc[matching_samples]
            print(data.shape)

    # filter for matching genes
    gene_sets = read_gmt(gmt_file=gene_sets_gmt_file)

    all_genes_from_gene_sets = set()
    for set_name, genes in gene_sets.items():
        all_genes_from_gene_sets |= genes
    considered_genes = list(all_genes_from_gene_sets)

    network = get_network_from_csv(network_csv_file,
                                   sep=',').get_sub_network(considered_genes)

    # write data and inducers
    for data_name, data in data_dict.items():
        data_filtered, network_filtered = get_matching_data_and_network(
            data, network
        )
        write_preprocessed(
            data=data_filtered,
            data_name=data_name,
            network=network_filtered,
            network_name=network_name,
            gene_sets=gene_sets,
            gene_sets_name=gene_sets_name,
            output_dir=preprocess_dir
        )

@click.command()
@click.option('-fd', '--data_csv_file', 'data_csv_files', required=True, multiple=True, type=click.Path())
@click.option('-nd', '--data_name', 'data_names', required=True, multiple=True)
@click.argument('network_csv_file', required=True, type=click.Path())
@click.argument('network_name', required=True)
@click.argument('gene_sets_gmt_file', required=True, type=click.Path())
@click.argument('gene_sets_name', required=True)
@click.argument('preprocess_dir', required=True, type=click.Path(exists=True, file_okay=False, writable=True))
def main(
    data_csv_files,
    data_names,
    network_csv_file,
    network_name,
    gene_sets_gmt_file,
    gene_sets_name,
    preprocess_dir,
    match_samples=True
):
    """
    Compute incuding Laplacian matrices and preprocess data matrices for
    matching features.

    Multiple data_csv_files may be passed.
    Each data_csv_file should readable as pandas.DataFrames
    `pd.read_csv(filename, sep=',', index_col=0)` where index are features
    (rows) and columns a are samples.

    The `network_csv_file` is an edge list readable with `pd.read_csv(filename)`
    where the 3rd columns is a numeric value.

    The `gene_sets_gmt_file` should follow the gmt specification. See
    http://software.broadinstitute.org/cancer/software/gsea/wiki/index.php/Data_formats

    For each file, a name has to be passed. Names cannot contain "_" or "-".

    Results are written to `preprocess_dir`.
    """
    preprocess_data_and_inducers(
        data_csv_files,
        data_names,
        network_csv_file,
        network_name,
        gene_sets_gmt_file,
        gene_sets_name,
        preprocess_dir,
        match_samples
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
