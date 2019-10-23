import random
import pandas as pd
import scipy.sparse as ss
import numpy as np
import networkx as nx
from scipy.sparse import lil_matrix

machine_epsilon = np.finfo(float).eps


def is_symmetric(m):
    if m.shape[0] != m.shape[1]:
        raise ValueError('m must be a square matrix')

    if not isinstance(m, ss.coo_matrix):
        m = ss.coo_matrix(m)

    r, c, v = m.row, m.col, m.data
    tril_no_diag = r > c
    triu_no_diag = c > r

    if triu_no_diag.sum() != tril_no_diag.sum():
        return False

    rl = r[tril_no_diag]
    cl = c[tril_no_diag]
    vl = v[tril_no_diag]
    ru = r[triu_no_diag]
    cu = c[triu_no_diag]
    vu = v[triu_no_diag]

    sortl = np.lexsort((cl, rl))
    sortu = np.lexsort((ru, cu))
    vl = vl[sortl]
    vu = vu[sortu]

    check = np.allclose(vl, vu)

    return check


class Network(object):

    def __init__(self, graph, labels):
        self.graph = graph
        self.labels = labels

    def get_sub_network(self, labels):
        matching_labels = set(self.labels.index) & set(labels)
        selected_labels = self.labels[matching_labels].values.astype(np.int64
                                                                     ).tolist()
        return Network(
            ss.csr_matrix(
                self.graph.tocsr()[selected_labels, :].tocsc()
                [:, selected_labels]
            ),
            pd.Series(
                {label: index
                 for index, label in enumerate(matching_labels)}
            )
        )

    def get_laplacian(self, normed=True, return_diag=False):
        # getting the symmetric normalized laplacian
        return ss.csgraph.laplacian(
            self.graph, normed=normed, return_diag=return_diag
        )


def scale(array):
    minimum, maximum = array.min(), array.max()
    if maximum != minimum:
        return (array - minimum) / float(maximum - minimum) + machine_epsilon
    else:
        return array


def force_undirected_coo_matrix_input(row, col, values):
    # WARNING: selection based on the order
    undirected_selection = get_unique_rows(
        np.sort(np.stack([row, col], axis=1), axis=1)
    )[1]

    selected_row = row[undirected_selection]
    selected_col = col[undirected_selection]
    selected_values = values[undirected_selection]

    stacked_undirected_row_and_col = np.stack(
        [
            np.concatenate([selected_row, selected_col]),
            np.concatenate([selected_col, selected_row])
        ],
        axis=1
    )
    undirected_values = np.concatenate([selected_values, selected_values])
    selected_indices = get_unique_rows(stacked_undirected_row_and_col)[1]
    return (
        stacked_undirected_row_and_col[selected_indices, 0],
        stacked_undirected_row_and_col[selected_indices, 1],
        undirected_values[selected_indices]
    )


def get_unique_rows(matrix, return_index=True):
    return np.unique(
        np.ascontiguousarray(matrix).view(
            np.dtype((np.void, matrix.dtype.itemsize * matrix.shape[1]))
        ),
        return_index=return_index
    )


def get_network_from_pandas_interactions_list(
    data, adjacency=False, threshold=None, force_undirected=True
):
    columns = data.columns

    if force_undirected:
        data.sort_values(by=columns[2], axis=0, ascending=False, inplace=True)

    labels = pd.Series(
        {
            label: index
            for index, label in
            enumerate(sorted(set(data.values[:, :2].flatten())))
        }
    )
    row = labels[data[columns[0]]].values
    col = labels[data[columns[1]]].values
    weights = data[columns[2]].values

    if adjacency:
        weights = np.array(weights != 0., dtype=np.float64)
    elif threshold is not None:
        weights = np.array(np.abs(weights) > threshold, dtype=np.float64)
    else:
        weights = scale(weights)

    if force_undirected:
        row, col, weights = force_undirected_coo_matrix_input(
            row, col, weights
        )

    number_of_nodes = len(labels)
    graph = ss.coo_matrix(
        (weights, (row, col)), shape=(number_of_nodes, number_of_nodes)
    ).tocsr()
    if force_undirected and not is_symmetric(graph):
        raise RuntimeError(
            'Error: force_undirected with non symmetric adjacency.'
        )
    return Network(graph, labels)


def get_network_from_csv(filename, sep=',', **kwargs):
    data = pd.read_csv(filename, sep=sep)
    return get_network_from_pandas_interactions_list(data, **kwargs)


def get_fantom5_network(fantom5_filename, **kwargs):
    return get_network_from_csv(fantom5_filename, sep='\t', **kwargs)


def get_string_network(string_filename, **kwargs):
    return get_network_from_csv(string_filename, sep=',', **kwargs)


def generate_random_sets(
    number_of_sets, max_nodes, nodes_labels, number_of_nodes=None
):
    sets = {}

    if number_of_nodes:

        def get_number_of_nodes(max_nodes):
            return number_of_nodes
    else:

        def get_number_of_nodes(max_nodes):
            return np.random.randint(50, max_nodes)

    for a_set in range(number_of_sets):
        number_of_nodes = get_number_of_nodes(max_nodes)
        sets['random_{}'.format(a_set)
             ] = set(random.sample(nodes_labels, number_of_nodes))
    return sets


def filter_interaction_table_by_labels(interaction_table, labels):
    pattern = r'|'.join([r'^{}$'.format(label) for label in labels])
    return interaction_table[interaction_table['e1'].str.match(pattern)
                             & interaction_table['e2'].str.match(pattern)]


def selected_set_to_weighted_adjacency(
    interaction_table, selected_set, all_nodes_labels
):
    n = len(all_nodes_labels)
    label_to_index = {
        label: index
        for index, label in enumerate(all_nodes_labels)
    }
    adjacency = lil_matrix((n, n))
    filtered_table = filter_interaction_table_by_labels(
        interaction_table, selected_set
    )
    for _, row in filtered_table.iterrows():
        i, j = label_to_index[row['e1']], label_to_index[row['e2']]
        adjacency[i, j] = row['intensity']
        adjacency[j, i] = row['intensity']
    return adjacency.tocsr()


def get_random_scale_free_interaction_df(nodes_labels, m=5):
    # http://barabasi.com/f/353.pdf barabasi-albert and
    # NETWORK BIOLOGY: UNDERSTANDING THE CELL'S FUNCTIONAL ORGANIZATION scale
    # free m=5
    n = len(nodes_labels)
    graph = nx.barabasi_albert_graph(n, m=m).to_undirected()
    index = []
    data = []
    for i, j in graph.edges():
        e1, e2 = sorted((nodes_labels[i], nodes_labels[j]))
        index.append('{}<->{}'.format(e1, e2))
        data.append([e1, e2, 1.0])
    return pd.DataFrame(data, index=index, columns=['e1', 'e2', 'intensity'])
