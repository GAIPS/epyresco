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
from typing import List, Dict, Tuple, Union

import networkx as nx
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import scipy.sparse as sp

Array = np.ndarray


def consensus_matrices(
    n_nodes: int,
    cm_n_edges: int = 0,
    cm_type: str = "metropolis",
) -> List[Array]:
    """Generates a list of consensus matrices

    Parameters
    ----------
    * n_nodes: Number of vertices.
    * cm_n_edges: Number of edges.
    * cm_type: Algorithm for consensus choice between
        ('metropolis', 'laplacian', 'normalized_laplacian').

    Returns
    -------
    * cwms: Consensus weights matrices.
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


def _adjacency_matrices(n_nodes: int, n_edges: int) -> List[Array]:
    """Produce all adjacency matrices with n_edges

                             A[i, i] = 0
                            /
     A is adjacency matrix < --->  A[i, j] = 1 iff i, j are neighbors
                            \
                            A[i, j] = 0 otherwise

    Parameters:
    -----------
    * n_nodes: Number of vertices.
    * n_edges: Number of edges.

    Returns:
    --------
    * ams: A list of adjacency matrices.
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
    * nodes: Number of agents.
    * cm_type: Consensus matrix type either `metropolis` or `laplacian`.
    * debug: If True returns a fully connected matrix

    Returns:
    --------
    * weights_list: List of matrices that guarantee consensus
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


def consensus_from_neighbors(
    all_ts_ids: str,
    neighbors: Dict[Tuple[str, str], int],
    max_edges: Union[float, int] = -1,
    to_torch: bool = True,
) -> Union[List[Array], List[th.Tensor]]:
    """Returns metropolis weights from neighbors

    Parameters:
    ----------
        * all_ts_ids: list of ts ids.
        * neighbors: keys are tuples in which elements are from tls id. to tls
        id. Values are costs or hops.
        * max_edges: maximum number of edges in consensus matrices expressed as
        a fraction of the total of neighbors or as the absolute number of
        neighbors. Use -1 to use all edges.
        * to_torch: convert weights to torch.

    Returns:
    --------
        * consensus matrix: a list of consensus weights matrices.
    """
    matrices = []
    # Converts fractional format to absolute format
    if max_edges > 0 and max_edges < 1:
        max_edges = max(round(max_edges * len(neighbors)), 1)

    k_comb = max_edges if max_edges > 0 else len(neighbors)
    for combination in it.combinations(list(neighbors.keys()), k_comb):
        matrix = np.zeros((len(all_ts_ids), len(all_ts_ids)), dtype=np.float32)
        for source, destination in combination:
            i = all_ts_ids.index(source)
            j = all_ts_ids.index(destination)
            matrix[i, j] = 1
            matrix[j, i] = 1
        matrices.append(metropolis_weights_matrix(matrix))

    if to_torch:

        def fn(x):
            return th.from_numpy(x).type(th.float32)

        matrices = [*map(fn, matrices)]
    return matrices


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


def main(target: int = 3, resco_map: str = "cologne8") -> None:
    """Performs distributed averaging on a simple graph.

    Parameters
    ----------
    target: The integer with the average the nodes should agree on.
    resco_map: Indicates a valid environment in resco. Choice ('cologne3', 'cologne8',  'ingolstaldt7', ingolstaldt21')
    """
    assert resco_map in ("cologne3", "cologne8", "ingolstaldt7", "ingolstaldt21")
    # n_edges = 2 * (n_nodes - 1)

    # adjacency = random_adjacency_matrix(n_nodes, n_edges)
    if resco_map == "cologne8":
        n_nodes = 8

        adjacency = np.zeros((n_nodes, n_nodes), dtype=int)
        adjacency[0, [3, 7]] = 1
        adjacency[1, [5, 7]] = 1
        adjacency[2, [6]] = 1
        adjacency[3, [0, 4]] = 1
        adjacency[4, [2, 3, 6]] = 1
        adjacency[5, [1, 6]] = 1
        adjacency[6, [4, 5]] = 1
        adjacency[7, [0, 1]] = 1
    elif resco_map == "ingolstaldt21":
        n_nodes = 21

        adjacency = np.zeros((n_nodes, n_nodes), dtype=int)
        adjacency[0, [1, 13, 17]] = 1
        adjacency[1, [0, 17]] = 1
        adjacency[2, [9, 13, 14, 20]] = 1
        adjacency[3, [6, 13, 20]] = 1
        adjacency[4, [17, 11]] = 1
        adjacency[5, [6, 19, 20]] = 1
        adjacency[6, [3, 5, 8]] = 1

        adjacency[7, [14, 19, 20]] = 1
        adjacency[8, [6, 11]] = 1
        adjacency[9, [2, 10, 13]] = 1
        adjacency[10, [9, 16]] = 1
        adjacency[11, [4, 8]] = 1
        adjacency[12, [15]] = 1
        adjacency[13, [0, 2, 3, 9, 20]] = 1

        adjacency[14, [2, 7, 16]] = 1
        adjacency[15, [12, 16]] = 1
        adjacency[16, [10, 14, 15]] = 1
        adjacency[17, [0, 1, 4]] = 1
        adjacency[18, [19]] = 1
        adjacency[19, [5, 7, 18]] = 1
        adjacency[20, [2, 3, 5, 7, 13]] = 1

    else:
        raise NotImplemented

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
    mse = [np.mean((x - target) ** 2)]
    n_steps = 99
    for _ in range(n_steps):
        x = C @ x
        log.append(x)
        mse.append(np.mean((x - target) ** 2))

    X = np.linspace(1, n_steps + 1, n_steps + 1)
    Y = np.stack(log)
    Z = np.stack(mse)

    # Beware that the graph must be fully connected
    plt.axhline(y=target, color=(0.2, 1.0, 0.2), linestyle="-")
    plt.suptitle("Consensus Iterations (%s, %s)" % (n_nodes, target))
    plt.ylabel("Data")
    plt.xlabel("Time")
    plt.plot(X, Y)
    plt.show()

    fig, ax = plt.subplots()
    plt.suptitle("Consensus Iterations (%s, %s)" % (n_nodes, target))
    plt.ylabel("Data")
    plt.xlabel("Time")

    ax.axhline(y=0.05, color=(0.2, 1.0, 0.2), linestyle="-")
    ax.axvline(x=np.sum(Z > 0.05).astype(int), color=(0.2, 1.0, 0.2), linestyle="-")
    ax.set_yscale("log")
    ax.plot(X, Z)
    plt.show()


if __name__ == "__main__":
    main(3, "ingolstaldt21")
