import re
import gzip
import os
import numpy as np
import scipy.sparse as ss
import scipy.sparse.csgraph as ssc
import pandas as pd
from .network import is_symmetric
import logging

logger = logging.getLogger(__name__)


def get_matching_data_and_network(data, network):
    """Interesct data labels with network node labels."""
    matching_genes = (network.labels.index & data.columns).tolist()
    data_network = network.get_sub_network(matching_genes)
    genes = data_network.labels.sort_values().index
    data_filtered = data[genes]
    return data_filtered, data_network


def get_pathway_inducer(network, gene_set, normed=True):
    """Get a laplacian based pathway inducer."""
    index = network.labels.loc[network.labels.index.intersection(gene_set)
                               ].values.astype(np.int64)
    m = len(index)
    laplacian = None
    if m > 1:
        row = np.concatenate([np.repeat(i, m) for i in index])
        col = np.concatenate([index for _ in range(m)])
        values = np.ones(m * m)
        mask = ss.coo_matrix((values, (row, col)),
                             shape=network.graph.shape).tocsr()
        pathway_graph = mask.multiply(network.graph).tocsr()
        if not is_symmetric(pathway_graph):
            raise RuntimeError('Error: adjacency matrix is not symmetric')
        laplacian = ssc.laplacian(pathway_graph, normed=normed).tocsr()
        if len(laplacian.nonzero()[0]) < 1:  # invalid inducer
            laplacian = None
    if laplacian is None:
        logger.debug('invalid laplacian')
    return laplacian


def get_pathway_selector(network, gene_set):
    """Get a pathway selector."""
    index = network.labels.loc[network.labels.index.intersection(gene_set)
                               ].values.astype(np.int64)
    m = len(index)
    if m > 1:
        values = np.ones(m)
        return ss.coo_matrix(
            (values, (index, index)), shape=network.graph.shape
        ).tocsr()
    else:
        return None


more_than_one_whitespace = re.compile(r'\s+')


def read_gmt_from_file_pointer(fp):
    sets = {}
    for line in fp:
        splitted = more_than_one_whitespace.split(line.strip().decode())
        sets[splitted[0]] = set(splitted[2:])
    return sets


def read_gmt(gmt_file):
    """Read a .gmt file."""
    sets = {}
    with open(gmt_file, 'rb') as fp:
        sets = read_gmt_from_file_pointer(fp)
    return sets


def write_inducer(inducer, filename, sep=','):
    """Write and inducer in COO format."""
    inducer = inducer.tocoo()
    with gzip.open(filename, 'wt') as fp:
        for triplet in zip(inducer.row, inducer.col, inducer.data):
            fp.write(sep.join(map(str, triplet)) + os.linesep)


def read_inducer(filename, size, header=None, sep=','):
    """Read inducer in CSC format."""
    inducer = pd.read_csv(filename, header=header, sep=',')
    inducer.columns = ['row', 'column', 'data']
    return ss.coo_matrix(
        (inducer['data'], (inducer['row'], inducer['column'])),
        shape=(size, size)
    ).tocsc()


def write_inducers(
    data,
    network,
    gene_sets,
    data_type,
    network_type,
    output_dir,
    selection_only=False,
    gene_set_type=''
):
    """Write inducers and data for a specific data-network combination."""
    data_filtered, data_network = get_matching_data_and_network(data, network)
    suffix = (
        '{}-selector'.format(network_type) if selection_only else network_type
    )

    # save inducers
    for name, gene_set in gene_sets.items():
        L = (
            get_pathway_selector(data_network, gene_set=gene_set)
            if selection_only else
            get_pathway_inducer(data_network, gene_set=gene_set)
        )
        if L is not None:
            write_inducer(
                L, '{}/{}_{}_{}.csv.gz'.format(
                    output_dir, name.lower(), data_type, suffix
                )
            )

    if len(gene_set_type) > 0:
        suffix = '{}-{}'.format(suffix, gene_set_type)
    # save the data
    data_filtered.to_csv('{}/{}_{}.csv'.format(output_dir, data_type, suffix))


def write_preprocessed(
    data, data_name, network, network_name, gene_sets, gene_sets_name,
    output_dir
):
    """Write inducers and data for a specific data-network combination."""
    # save inducers
    for name, gene_set in gene_sets.items():
        L = (get_pathway_inducer(network, gene_set=gene_set))
        if L is not None:
            write_inducer(
                L, '{}/{}_{}_{}_{}.csv.gz'.format(
                    output_dir,
                    'inducer',
                    data_name,
                    network_name,
                    name,
                )
            )
    # save the data
    data.to_csv(
        '{}/{}_{}_{}_{}.csv'.format(
            output_dir, 'data', data_name, network_name, gene_sets_name
        )
    )
