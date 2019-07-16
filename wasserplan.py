"""Wasserstein distances between districting plans."""
import networkx as nx
import cvxpy as cp
import numpy as np
from scipy.sparse import csr_matrix
from gerrychain import Partition


class Pair:
    """A pair of isomorphic districting plans to compare."""
    def __init__(self, partition_a: Partition, partition_b: Partition,
                 embedding: str = 'node', pop_col: str = None):
        """
        :param partition_a: The first GerryChain partition to compare.
        :param partition_b: The second GerryChain partition to compare.
        :param embedding: The name of the district embedding scheme to use.
            Valid embeddings are "node" (equal population assumed for
            all nodes in the dual graph) and "population" (nodes are weighted
            proportional to population).
        :param pop_col: The name of the attribute specifying a node's
            population. Required for the "population" embedding only.
        """
        if not nx.algorithms.isomorphism.is_isomorphic(partition_a.graph,
                                                       partition_b.graph):
            raise IsomorphismError('The graphs of the partitions are not '
                                   'isomorphic. Were the partitions '
                                   'generated from the same Markov chain?')
        if embedding == 'population' and not pop_col:
            raise EmbeddingError('Cannot generate population-based embeddings '
                                 'without population data. Specify a '
                                 'population column.')
        elif embedding not in ('population', 'node'):
            raise EmbeddingError(f'Unknown embedding type "{embedding}"!')

        self.embedding_type = embedding
        self.pop_col = pop_col
        self.partition_a = partition_a
        self.partition_b = partition_b

        # NetworkX graphs are not guaranteed to have integer node indices.
        # For example, nx.grid_graph generates nodes indexed by 2D
        # Cartesian coordinates. Thus, it is necessary to fix an arbitrary
        # ordering internally that maps each index to a unique integer in
        # 0..len(graph.nodes) - 1. We sort the nodes first in the hope that
        # doing so will result in a more easily interpretable ordering.
        # (For instance, on a 2x2 grid graph, we might expect the node
        #  with NetworkX label (0, 0) to map to 0 and (1, 1) to map to 4.)
        self.node_ordering = {
            node: idx
            for idx, node in enumerate(sorted(partition_a.graph.nodes))
        }
        # Similarly, we fix an arbitrary ordering for edges.
        self.edge_ordering = {
            edge: idx
            for idx, edge in enumerate(sorted(partition_a.graph.edges))
        }
        # We also fix an ordering for districts. This is primarily
        # to resolve indexing issues—by convention, the districts in a
        # districting plan are 1-indexed, but GerryChain also allows
        # 0-indexing.
        self.district_ordering = {
            district: idx
            for idx, district in enumerate(sorted(partition_a.parts.keys()))
        }

        self._a_embeddings = self.embed(partition_a)
        self._b_embeddings = self.embed(partition_b)
        self._pairwise_distances = None  # lazy-loaded

    def embed(self, partition: Partition) -> np.ndarray:
        """Embeds all districts in a partition.

        :param partition: The partition to embed.
        """
        n_districts = len(partition)
        n_nodes = len(partition.graph.nodes)
        embedding = np.zeros((n_districts, n_nodes))
        if self.embedding_type == 'node':
            for district_idx, district_label in self.district_ordering.items():
                nodes_in_district = [
                    self.node_ordering[node]
                    for node in partition.parts[district_label]
                ]
                embedding[district_idx][nodes_in_district] = 1
        elif self.embedding_type == 'population':
            for district_idx, district_label in self.district_ordering.items():
                for node_label in partition.parts[district_label]:
                    node_idx = self.node_ordering[node_label]
                    node = partition.graph.nodes[node_idx]
                    try:
                        node_pop = node[self.pop_col]
                    except KeyError:
                        raise EmbeddingError('Cannot create population '
                                             f'embedding. Node {node_label} '
                                             f'has no "{self.pop_col}" '
                                             'attribute.')
                    embedding[district_idx][node_idx] = node_pop

        # Norm so that rows sum to 1.
        return embedding / np.sum(embedding, axis=1)

    def district_distance(self, a_index: int, b_index: int) -> np.float64:
        """Calculates the 1-Wasserstein distance between districts.

        Districts are compared across plans only, as districts within
        a plan are disjoint by definition.

        :param a_index: The index of the district to compare in the
           first district (``partition_a``).
        :param b_index: The index of the district to compare in the
           second district (``partition_b``).
        """
        if self._pairwise_distances:
            # Avoid recomputation if district distances have already been
            # computed in the course of computing the plan distance.
            return self._pairwise_distances[a_index][b_index]
        return None

    @property
    def distance(self) -> np.float64:
        """Calculates the 1-Wasserstein distance between plans."""
        if not self._pairwise_distances:
            self._pairwise_distances = self.get_pairwise_distances()
        return None

    def _get_pairwise_distances(self) -> np.ndarray:
        """Generates all pairwise distances between districts.

        For a pair of districting plans with :math:`n` districts each,
        there are :math:`n^2` pairs.
        """


class EmbeddingError(Exception):
    """Raised for invalid embedding schemes."""


class IsomorphismError(Exception):
    """Raised if the graphs of a pair of partitions are not isomorphic."""
