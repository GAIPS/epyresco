"""Consensus is the local estimation of a global state.

Given a network of processes where each node has an initial scalar value,
we consider the problem of computing their average asymptotically using a
distributed, linear iterative algorithm. At each iteration, each node
replaces its own value with a weighted average of its previous value and
the values of its neighbors. We introduce the Metropolis weights, a simple
choice for the averaging weights used in each step. We show that with these
weights, the values at every node converge to the average, provided the
infinitely occurring communication graphs are jointly connected. [3]

Reference
---------
..[1] Lin Xiao and Stephen Boyd, 2004,
  "Fast Linear Iterations For Distributed Averaging".
..[2] https://mathworld.wolfram.com/LaplacianMatrix.html
..[3] Lin Xiao, Stephen Boyd, Sanjay Lall, 2006,
  "Distributed Average Consensus with Time-Varying Metropolis Weights"

"""

import copy
import itertools as it
import operator as op

import networkx as nx
from networkx.readwrite.json_graph import adjacency
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import scipy.sparse as sp

from typing import List

Array = np.ndarray


def consensus_matrices(
    n_nodes: int,
    cm_n_edges: int = 0,
    cm_type: str = "metropolis",
) -> List[Array]:
    """Generates a list of consensus matrices

    Parameters
    ----------
    * n_nodes: int
        A two dimension array representing an adjacency matrix.

    * cm_n_edges: int
        Number of edges

    * cm_type: str = 'metropolis'
        A string with the algorithm for consensus.

    Returns
    -------
    * cwms: List[Array]
        A list containing consensus weights matrices
    """
    if n_nodes == 1:
        raise ValueError("%s invalid" % str(n_nodes))
    if cm_n_edges < 0 or cm_n_edges > (n_nodes * (n_nodes - 1) // 2):
        raise ValueError("max_edges: %s invalid" % str(cm_n_edges))
    if cm_type not in ("metropolis", "normalized_laplacian", "laplacian"):
        raise ValueError("%s invalid" % cm_type)
    else:
        fn = eval("%s_weights_matrix" % cm_type)

    cwms = []
    ams = _adjacency_matrices(n_nodes, cm_n_edges)
    cwms += map(fn, ams)
    return cwms


def _adjacency_matrices(n_nodes: int, n_edges: int) -> Array:
    """Produce all adjacency matrices with n_edges

                             A[i, i] = 0
                            /
     A is adjacency matrix < --->  A[i, j] = 1 iff i, j are neighbors
                            \
                            A[i, j] = 0 otherwise

    Parameters
    ----------
    n_nodes: int
    n_edges: int

    Returns
    -------
    * ams: Array
        A list of two dimension array representing an adjacency matrix.
    """
    if n_edges == 0:
        return [np.zeros((n_nodes, n_nodes))]

    full_edge_list = [(i, j) for i in range(n_nodes - 1) for j in range(i + 1, n_nodes)]
    ones = np.ones(n_edges, dtype=int)

    ams = []
    for edge_set in it.combinations(full_edge_list, n_edges):
        am = sp.csr_matrix(
            (ones, zip(*edge_set)), dtype=int, shape=(n_nodes, n_nodes)
        ).toarray()
        ams.append(am + am.T)

    return ams


def consensus_matrices2(
    nodes: int = 3, cm_type: str = "metropolis", debug: bool = False
) -> List[Array]:
    """Returns a consensus matrix for a connected undirected graph

    Parameters:
    -----------
    nodes: default: 3
        Number of agents.
    cm_type: default: `metropolis`
        Consensus matrix type either `metropolis` or `laplacian`.
    debug: dafault: False
        If True returns a fully connected matrix

    Returns:
    --------
    list
        List of matrices that guarantee consensus
    """

    if cm_type not in ("metropolis", "laplacian"):
        raise ValueError(f"cm_type not in ('metropolis', 'laplacian'). Got {cm_type}")

    def adj(x):
        return _graph_to_numpy_array(x)

    def weights(x):
        if cm_type == "metropolis":
            return metropolis_weights_matrix(x)
        else:
            return laplacian_weights_matrix(x, fast=True)

    if debug:
        connected_graphs_list = [nx.complete_graph(nodes)]

    else:
        connected_graphs_list = generate_connected_graphs(nodes=nodes)
    adjacency_list = [*map(adj, connected_graphs_list)]
    weights_list = [*map(weights, adjacency_list)]
    return weights_list


def consensus_from_neighbors(all_ts_ids, neighbors, to_torch=True):
    """Returns metropolis weights from neighbors
    Args:
        all_ts_ids (list): list of ts ids
        neighbors (dict): keys are tuples in which elements are from tls id,
        to tls id. Values are costs or hops.
        to_torch (bool): convert weights to torch.

    Returns:
        consensus matrix (numpy.ndarray | th.tensor): a matrix with consensus weights.
    """
    matrix = np.zeros((len(all_ts_ids), len(all_ts_ids)), dtype=np.float)

    for source, destination in neighbors:
        i = all_ts_ids.index(source)
        j = all_ts_ids.index(destination)
        matrix[i, j] = 1
        matrix[j, i] = 1

    cwm = metropolis_weights_matrix(matrix)
    if to_torch:
        cwm = th.from_numpy(cwm.astype(np.float32))
    return cwm


def generate_automorphisms(graph_list):
    """Return the automorphisms for a graph list.

    An automorphism of a graph is a form of symmetry in which the graph is
    mapped onto itself while preserving the edge–vertex connectivity.
    Formally, an automorphism of a graph G = (V, E) is a permutation σ of the
    vertex set V, such that the pair of vertices (u, v) form an edge if and
    only if the pair (σ(u), σ(v)) also form an edge. That is, it is a graph
    isomorphism from G to itself.

    Parameters:
    ----------
    graph_list: graph_list

    Returns:
    -------
    automorphisms: list
    graphs that are automorph in relation to graph (permutation of edges).
    permutations from original edge list
    """
    automorphisms = copy.deepcopy(graph_list)

    def permute(x, y):
        # Permutes nodes y in edges x
        return _sorted_edges(_permute_edges(x, y))

    for graph in graph_list:
        for perm in list(it.permutations([*graph.nodes])):
            edge_list = permute(graph.edges, perm)
            graph1 = nx.Graph(edge_list)

            g_list = filter(lambda x: _graphs_equal(graph1, x), automorphisms)
            if not any([*g_list]):
                automorphisms.append(graph1)
    return automorphisms


def generate_connected_graphs(nodes=3):
    """Returns all connected graphs with given nodes

    Parameters:
    ----------
    nodes: int
    number of nodes for the connected graph

    Returns:
    -------
    list: graphs
    """
    assert nodes < 8, f"Maximum of seven nodes supported got {nodes}"
    atlas = nx.graph_atlas_g()[3:]  # 0, 1, 2 => no edges. 208 is last 6 node graph
    graph_list = []
    for graph in atlas:
        if (
            len(graph.nodes) == nodes
            and len(graph.edges) > 0
            and nx.number_connected_components(graph) == 1
        ):
            graph_list.append(graph)
    if nodes < 6:
        return generate_automorphisms(graph_list)
    else:
        return graph_list


def metropolis_weights_matrix(am: Array) -> Array:
    """Consensus matrix[3] from an adjacency matrix

    Parameters
    ----------
    * adjacency: Array
        A two dimension array representing an adjacency matrix.

    Returns
    -------
    * mwm: Array
        A two dimension array the metropolis weights matrix
    """
    adj = np.array(am)
    degree = np.sum(adj, axis=1)
    mwm = np.zeros_like(am, dtype=float)
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[0]):
            if am[i, j] > 0:
                mwm[i, j] = 1 / (1 + max(degree[i], degree[j]))
                mwm[j, i] = mwm[i, j]  # symmetrical
        mwm[i, i] = 1 - (mwm[i, :].sum())
    return mwm


def laplacian_weights_matrix(am: Array, fast: bool = True) -> Array:
    """Consensus matrix[1] from an adjacency matrix

    Parameters
    ----------
    * adjacency: Array
        A two dimension array representing an adjacency matrix.

    Returns
    -------
    * lwm: Array
        A two dimension array the laplacian weights matrix
    """
    eye = np.eye(*am.shape)
    degree = np.sum(am, axis=1)
    laplacian = np.diag(degree) - am

    # fast computation -- two largest
    if fast:
        alpha = 1 / sum(sorted(degree, reverse=True)[:2])
    else:
        eig, _ = np.linalg.eig(laplacian)
        alpha = 2 / (eig[0] + eig[-2])

    lwm = eye - alpha * laplacian
    return np.array(lwm)


def random_adjacency_matrix(n_nodes: int, n_edges: int) -> Array:
    """Randomly produce an adjacency matrix

                             A[i, i] = 0
                            /
     A is adjacency matrix < --->  A[i, j] = 1 iff i, j are neighbors
                            \
                            A[i, j] = 0 otherwise

    Parameters
    ----------
    n_nodes: int
    n_edges: int

    Returns
    -------
    * ram: Array
        A two dimension array representing an adjacency matrix.
    """
    if n_edges == 0:
        return np.zeros((n_nodes, n_nodes))

    full_edge_list = [(i, j) for i in range(n_nodes - 1) for j in range(i + 1, n_nodes)]

    n_choices = min(len(full_edge_list), n_edges)
    edge_ids = np.random.choice(len(full_edge_list), replace=False, size=n_choices)

    edge_list = [full_edge_list[i] for i in sorted(edge_ids)]

    data = (np.ones(len(edge_list), dtype=int), zip(*edge_list))
    ram = csr_matrix(data, dtype=int, shape=(n_nodes, n_nodes)).toarray()
    ram = ram + ram.T
    return ram


def _graphs_equal(graph1, graph2):
    """Check if graphs are equal.

    Equality here means equal as Python objects (not isomorphism).
    Node, edge and graph data must match.

    Parameters
    ----------
    graph1, graph2 : graph

    Returns
    -------
    bool
        True if graphs are equal, False otherwise.
    """
    adj1 = nx.to_numpy_array(graph1, nodelist=sorted(graph1.nodes))
    adj2 = nx.to_numpy_array(graph2, nodelist=sorted(graph2.nodes))
    return np.array_equal(adj1, adj2)


def _sorted_edges(edges):
    """Sorts edges by source node to incidence node

    Parameters:
    ----------
    edges: list

    Returns:
    -------
    list
    """
    return sorted(sorted(edges, key=op.itemgetter(1)), key=op.itemgetter(0))


def _permute_edges(edges, perm):
    """Relable edges according to nodes_permutation list

    Parameters:
    ----------
    edges: list
    list of tuples each of which with a source and destination node
    perm: list
    permutation over nodes

    Returns:
    -------
    list
    """
    return [*map(lambda x: (perm[x[0]], perm[x[1]]), edges)]


def _graph_to_numpy_array(graph):
    """Converts a graph into an adjacency matrix

    Parameters:
    -----------
    graph: nx.Graph

    Returns:
    --------
    np.ndarray: adjacency matrix
    """
    return nx.to_numpy_array(graph, nodelist=sorted(graph.nodes))


def main(n_nodes: int = 5, target: int = 3):
    """Performs distributed averaging on a simple graph.

    Parameters
    ----------
    n_nodes: int = 5
        The side of the square matrix
    target: int = 3
        The integer with the average the nodes should agree on.
    """

    # n_edges = 2 * (n_nodes - 1)

    # adjacency = random_adjacency_matrix(n_nodes, n_edges)

    adjacency = np.zeros((4, 4), dtype=int)
    adjacency[0, 1:3] = 1
    adjacency[1, 0] = 1
    adjacency[2, [0, 3]] = 1
    adjacency[3, 2] = 1

    print("ADJACENCY:")
    print(adjacency)

    # generate an array with average == target
    x = np.random.randint(low=0, high=2 * target, size=n_nodes)
    res = target - np.mean(x)
    x = x.astype(np.float32) + res

    # print("DATA:")
    # print(dict(enumerate(x.tolist())))

    print("Metropolis:")
    C = metropolis_weights_matrix(adjacency)
    print(C)

    log = [x]
    n_steps = 99
    for _ in range(n_steps):
        x = C @ x
        log.append(x)

    X = np.linspace(1, n_steps + 1, n_steps + 1)
    Y = np.stack(log)

    # Beware that the graph must be fully connected
    plt.axhline(y=target, color=(0.2, 1.0, 0.2), linestyle="-")
    plt.suptitle("Consensus Iterations (%s, %s)" % (n_nodes, target))
    plt.ylabel("Data")
    plt.xlabel("Time")
    plt.plot(X, Y)
    plt.show()


if __name__ == "__main__":
    main()
