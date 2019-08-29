"""Wasserstein distances between districting plans."""
from typing import Union, Dict, Hashable, Tuple
import networkx as nx
import cvxpy as cp
import numpy as np
from numpy import linalg as LA
from networkx.linalg.graphmatrix import incidence_matrix
from scipy.optimize import linear_sum_assignment
from gerrychain import Partition


class GenericPair:
    """A pair of districting plans to compare with a lifted distance metric.

    Subclasses are expected to implement :meth:`district_distance` and
    :meth:`indicator`. :meth:`district_distance` should calculate the distance
    between a district in partition A and a district in partition B; these
    district-level distances are lifted to a plan-level distance via
    minimum-cost matching. :meth:`indicator` should generate an indicator
    matrix for a plan that embeds each district as a vector sized according
    to the number of nodes in the graph that underlies the partitions.
    """

    def __init__(self,
                 partition_a: Partition,
                 partition_b: Partition,
                 indicator: str = 'node',
                 pop_col: str = None):
        """
        :param partition_a: The first GerryChain partition to compare.
        :param partition_b: The second GerryChain partition to compare.
        :param indicator: The name of the district indicator scheme to use.
            Valid indicators are "node" (equal population assumed for
            all nodes in the dual graph) and "population" (nodes are weighted
            proportional to population).
        :param pop_col: The name of the attribute specifying a node's
            population. Required for the "population" indicator only.
        """
        if indicator == 'population' and not pop_col:
            raise EmbeddingError('Cannot generate population-based indicators '
                                 'without population data. Specify a '
                                 'population column.')
        elif indicator not in ('population', 'node'):
            raise EmbeddingError(f'Unknown indicator type "{indicator}"!')

        self.indicator_type = indicator
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
        # We also fix an ordering for districts. This is primarily
        # to resolve indexing issues—by convention, the districts in a
        # districting plan are 1-indexed, but GerryChain also allows
        # 0-indexing.
        self.district_ordering = {
            district: idx
            for idx, district in enumerate(sorted(partition_a.parts.keys()))
        }

        self._a_indicators = self.indicators(partition_a)
        self._b_indicators = self.indicators(partition_b)
        self._pairwise_distances = None  # lazy-loaded
        self._edge_incidence = None  # lazy-loaded
        self._assignment = None  # lazy-loaded

    def indicators(self, partition: Partition) -> np.ndarray:
        """Generates indicator vectors for all districts in a partition."

        :param partition: The partition to generate indicator vectors for.
        """
        raise NotImplementedError('Indicator vectors not imlpemented. '
                                  'Do not use the base class to calculate '
                                  'distances.')

    def district_distance(self, a_label, b_label) -> np.float64:
        """Calculates the distance between districts.

        Districts are compared across plans only, as districts within
        a plan are disjoint by definition.

        :param a_label: The label of the district to compare in the
           first district (``partition_a``).
        :param b_label: The label of the district to compare in the
           second district (``partition_b``).
        """
        raise NotImplementedError('District-level distance not implemented. '
                                  'Do not use the base class to calculate '
                                  'distances.')

    @property
    def distance(self) -> np.float64:
        """Calculates the 1-Wasserstein distance between plans."""
        if self._pairwise_distances is None:
            self._pairwise_distances = self._get_pairwise_distances()
        if self._assignment is None:
            dist = self._pairwise_distances
            # pylint: disable=invalid-unary-operand-type
            a_indices, b_indices = linear_sum_assignment(dist)
            self._assignment = {
                a_index: b_index
                for a_index, b_index in zip(a_indices, b_indices)
            }

        total_dist = 0
        for a_index, b_index in self._assignment.items():
            total_dist += self._pairwise_distances[a_index][b_index]
        return total_dist

    def _get_pairwise_distances(self) -> np.ndarray:
        """Generates all pairwise distances between districts.

        For a pair of districting plans with :math:`n` districts each,
        there are :math:`n^2` pairs.
        """
        n_districts = len(self.partition_a)
        distances = np.zeros((n_districts, n_districts))
        for a_label, a_idx in self.district_ordering.items():
            for b_label, b_idx in self.district_ordering.items():
                dist = self.district_distance(a_label, b_label)
                distances[a_idx][b_idx] = dist
        return distances


class Pair(GenericPair):
    """A pair of districting plans to compare (balanced 1-Wasserstein)."""
    # The `Pair` class does not take any parameters beyond those
    # taken by `GenericPair`; no custom constructor is necessary.

    def indicators(self, partition: Partition) -> np.ndarray:
        """Returns normed indicator vectors for all districts in a partition.

        :param partition: The partition to generate indicator vectors for.
        """
        indicator = None
        if self.indicator_type == 'node':
            indicator = node_indicator(partition,
                                       self.district_ordering,
                                       self.node_ordering)
        elif self.indicator_type == 'population':
            indicator = population_indicator(partition,
                                             self.district_ordering,
                                             self.node_ordering,
                                             self.pop_col)

        # Norm so that rows sum to 1.
        return indicator / np.sum(indicator, axis=1).reshape(-1, 1)

    def district_distance(self, a_label, b_label) -> np.float64:
        """Calculates the balanced 1-Wasserstein distance between districts.

        Districts are compared across plans only, as districts within
        a plan are disjoint by definition.

        :param a_label: The label of the district to compare in the
           first district (``partition_a``).
        :param b_label: The label of the district to compare in the
           second district (``partition_b``).
        """
        a_idx = self.district_ordering[a_label]
        b_idx = self.district_ordering[b_label]
        if self._pairwise_distances:
            # Avoid recomputation if district distances have already been
            # computed in the course of computing the plan distance.
            return self._pairwise_distances[a_idx][b_idx]
        if self._edge_incidence is None:
            self._edge_incidence = incidence_matrix(self.partition_a.graph,
                                                    oriented=True)
        n_edges = self._edge_incidence.shape[1]
        edge_weights = cp.Variable(n_edges)
        diff = self._b_indicators[b_idx] - self._a_indicators[a_idx]
        objective = cp.Minimize(cp.sum(cp.abs(edge_weights)))
        conservation = (self._edge_incidence @ edge_weights) == diff
        prob = cp.Problem(objective, [conservation])
        prob.solve(solver='ECOS')  # solver recommended by Zach for big graphs
        return np.sum(np.abs(edge_weights.value))


class UnbalancedPair(Pair):
    """A pair of districting plans to compare (unbalanced 1-Wasserstein)."""
    def __init__(self,
                 partition_a: Partition,
                 partition_b: Partition,
                 slack_lambda: float,
                 slack_norm: Union[int, float, str],
                 indicator: str = 'node',
                 pop_col: str = None):
        """
        :param partition_a: The first GerryChain partition to compare.
        :param partition_b: The second GerryChain partition to compare.
        :param slack_lambda: The scale factor to use for the slack vector
           (used for calculation of distances between unbalanced partitions).
        :param slack_norm: The norm to use for the slack vector
           (used for calculation of distances between unbalanced partitions).
           Typically, this is a _p_-value for a _p_-norm, but "inf" (infinity
           norm) and "fro" (Frobenius norm) are also acceptable.
        :param indicator: The name of the district indicator scheme to use.
            Valid indicators are "node" (equal population assumed for
            all nodes in the dual graph) and "population" (nodes are weighted
            proportional to population).
        :param pop_col: The name of the attribute specifying a node's
            population. Required for the "population" indicator only.
        """
        super().__init__(partition_a=partition_a,
                         partition_b=partition_b,
                         indicator=indicator,
                         pop_col=pop_col)
        # TODO: validation of slack parameters?
        self.slack_lambda = slack_lambda
        self.slack_norm = slack_norm

    def indicators(self, partition: Partition) -> np.ndarray:
        """Returns scaled indicator vectors for all districts in a partition.

        Indicator matrices are scaled such that the rows that perfectly
        satisfy the equal population constraint (or, alternately, an
        equal number of nodes constraint) sum to 1.

        :param partition: The partition to generate indicator vectors for.
        """
        indicator = None
        target = None
        if self.indicator_type == 'node':
            target = len(partition.graph.nodes) / len(partition)
            indicator = node_indicator(partition,
                                       self.district_ordering,
                                       self.node_ordering)
        elif self.indicator_type == 'population':
            total_pop = sum(partition.graph.nodes[node][self.pop_col]
                            for node in partition.graph.nodes)
            target = total_pop / len(partition)
            indicator = population_indicator(partition,
                                             self.district_ordering,
                                             self.node_ordering,
                                             self.pop_col)
        return indicator / target

    def _district_distance(self, a_label, b_label) -> Tuple[np.float64,
                                                            np.ndarray,
                                                            np.ndarray]:
        """Calculates the unbalanced 1-Wasserstein distance between districts.

        Districts are compared across plans only, as districts within
        a plan are disjoint by definition.

        :param a_label: The label of the district to compare in the
            first district (``partition_a``).
        :param b_label: The label of the district to compare in the
            second district (``partition_b``).
        :returns: A 3-tuple with the distance as the first element,
            the
        """
        a_idx = self.district_ordering[a_label]
        b_idx = self.district_ordering[b_label]
        if self._pairwise_distances:
            # Avoid recomputation if district distances have already been
            # computed in the course of computing the plan distance.
            return self._pairwise_distances[a_idx][b_idx]
        if self._edge_incidence is None:
            self._edge_incidence = incidence_matrix(self.partition_a.graph,
                                                    oriented=True)
        n_nodes = self._edge_incidence.shape[0]
        n_edges = self._edge_incidence.shape[1]
        edge_flow = cp.Variable(n_edges)
        slack = cp.Variable(n_nodes)
        diff = self._b_indicators[b_idx] - self._a_indicators[a_idx]

        # Minimize the sum of:
        #   * The L1 norm of the edge flow vector (that is, the total flow
        #     through the entire graph)
        #   * The specified norm of the slack vector, scaled by some
        #     (positive) scale factor λ.
        # ...subject to node-level conservation of mass (slack included).
        total_flow = cp.norm(edge_flow, 1)
        scaled_slack = self.slack_lambda * cp.norm(slack, self.slack_norm)
        objective = cp.Minimize(total_flow + scaled_slack)
        conservation = (self._edge_incidence @ edge_flow) == (diff + slack)
        prob = cp.Problem(objective, [conservation])
        prob.solve()

        solved_total_flow = LA.norm(edge_flow.value, 1)
        solved_scaled_slack = self.slack_lambda * LA.norm(slack.value,
                                                          self.slack_norm)
        return (solved_total_flow + solved_scaled_slack,
                slack.value,
                edge_flow.value)


class EmbeddingError(Exception):
    """Raised for invalid indicator schemes."""


class IsomorphismError(Exception):
    """Raised if the graphs of a pair of partitions are not isomorphic."""


def population_indicator(partition: Partition,
                         district_ordering: Dict[Hashable, int],
                         node_ordering: Dict[Hashable, int],
                         pop_col: str) -> np.ndarray:
    """Returns an indicator matrix with each node weighted by its population.

    :param partition: The :class:`gerrychain.Partition` to generate an
        indicator matrix from.
    :param district_ordering: A mapping between the district labeling of the
        partition and the row indices of the indicator matrix.
    :param node_ordering: A mapping between the labeling of the nodes in the
        partition's underlying graph and the column indices of the indicator
        matrix.
    :param pop_col: The name of the population column/attribute used to
        determine population weighting. Each node of the partition's graph
        is expected to have this attribute.
    """
    n_districts = len(partition)
    n_nodes = len(partition.graph.nodes)
    indicator = np.zeros((n_districts, n_nodes))
    for district_label, district_idx in district_ordering.items():
        for node_label in partition.parts[district_label]:
            node = partition.graph.nodes[node_label]
            try:
                node_pop = node[pop_col]
            except KeyError:
                raise EmbeddingError('Cannot create population '
                                     f'indicator. Node {node_label} '
                                     f'has no "{pop_col}" attribute.')
            node_idx = node_ordering[node_label]
            indicator[district_idx][node_idx] = node_pop
    return indicator


def node_indicator(partition: Partition,
                   district_ordering: Dict[Hashable, int],
                   node_ordering: Dict[Hashable, int]) -> np.ndarray:
    """Returns an indicator matrix with a weighting of 1 for each node.

    :param partition: The :class:`gerrychain.Partition` to generate an
        indicator matrix from.
    :param district_ordering: A mapping between the district labeling of the
        partition and the row indices of the indicator matrix.
    :param node_ordering: A mapping between the labeling of the nodes in the
        partition's underlying graph and the column indices of the indicator
        matrix.
    """
    n_districts = len(partition)
    n_nodes = len(partition.graph.nodes)
    indicator = np.zeros((n_districts, n_nodes))
    for district_label, district_idx in district_ordering.items():
        nodes_in_district = [
            node_ordering[node]
            for node in partition.parts[district_label]
        ]
        indicator[district_idx][nodes_in_district] = 1
    return indicator
