#  Multi source Dijkstra + Minimax Fitness Function
# Other classes used :
#   models.city_node.CityNode  &
#   models.city_graph.CityGraph
#   Eligible ambulance nodes 
#  CityNode has no is_eligible_for_ambulance().
#   We define the rule here:
#  accessible Hospital or Ambulance Depot nodes.


import heapq
from typing import List, Dict, Tuple

from models.city_node import CityNode
from models.city_graph import CityGraph
from config import LOC_HOSPITAL, LOC_AMBULANCE


# =============================================================================
# HELPERS replicate CityGraph
# =============================================================================

def get_eligible_ambulance_nodes(graph: CityGraph) -> List[CityNode]:
    """
    Nodes where an ambulance can legally be placed:
    accessible Hospital or Ambulance Depot nodes.
    """
    return [
        n for n in graph.all_nodes()
        if n.is_accessible and n.location_type in {LOC_HOSPITAL, LOC_AMBULANCE}
    ]


def get_citizen_nodes(graph: CityGraph) -> List[CityNode]:
    """
    Demand points for the GA fitness function:
    accessible nodes with population_density > 0.
    """
    return [
        n for n in graph.all_nodes()
        if n.is_accessible and n.population_density > 0
    ]


def get_average_risk(graph: CityGraph) -> float:
    """
    Mean risk_multiplier across all nodes.
    Used by AmbulancePlacer to detect significant risk profile shifts.
    """
    nodes = list(graph.all_nodes())
    if not nodes:
        return 1.0
    return sum(n.risk_multiplier for n in nodes) / len(nodes)

# MULTI-SOURCE DIJKSTRA

def multi_source_dijkstra(
        graph: CityGraph,
        sources: List[CityNode]
) -> Dict[CityNode, float]:
    """
   shortest paths from multiple ambulance positions.
   Returns dist[node] = minimum risk-adjusted distance from any source.

effective_cost = base_weight × neighbor.risk_multiplier

    This is computed by node.effective_cost_to(neighbor)
    CityNode class.

    Args:
        graph   : Shared CityGraph (provides all_nodes context)
        sources : List of CityNode objects (ambulance positions)

    """
    dist: Dict[CityNode, float] = {}

    pq: List[Tuple[float, int, CityNode]] = []
    for source in sources:
        heapq.heappush(pq, (0.0, id(source), source))

    while pq:
        current_cost, _, current_node = heapq.heappop(pq)

        if current_node in dist:
            continue

        dist[current_node] = current_cost

        # node.get_neighbors() returns {neighbor: base_weight} dict
        for neighbor, base_weight in current_node.get_neighbors().items():
            if not neighbor.is_accessible:
                continue

            # effective_cost_to() = base_weight × neighbor.risk_multiplier
            effective_cost = current_node.effective_cost_to(neighbor)
            new_dist = current_cost + effective_cost

            if neighbor not in dist:
                heapq.heappush(pq, (new_dist, id(neighbor), neighbor))

    return dist

# MINIMAX FITNESS FUNCTION
def calculate_minimax_fitness(
        chromosome: List[CityNode],
        graph: CityGraph
) -> float:
    """
    Computes the MINIMAX fitness of one chromosome (one ambulance placement).

    Minimax definition:
        fitness = max over all citizens c of:
                      min over all ambulances a of:
                          shortest_path_dist(c, a)

    Because Multi-Source Dijkstra is used, dist[c] IS the minimum distance
    from c to its nearest ambulance in a single pass.

        fitness = max(dist[c] for all citizen nodes c)

    We MINIMISE this across GA generations.

    Returns:
        float: Worst-case response distance.
               float('inf') if any citizen is completely unreachable.
    """
    dist_from_nearest = multi_source_dijkstra(graph, chromosome)
    citizens = get_citizen_nodes(graph)

    if not citizens:
        return 0.0

    worst_case = 0.0
    for citizen in citizens:
        d = dist_from_nearest.get(citizen, float('inf'))
        if d > worst_case:
            worst_case = d

    return worst_case


# =============================================================================
# FITNESS CACHING
# =============================================================================
# The graph no longer has a .version counter (partner's CityGraph doesn't
# expose one).  We track risk changes via a simple hash of all
# risk_multiplier values across the graph — cheap and collision-resistant
# enough for this use-case.

_fitness_cache: Dict[tuple, float] = {}
_cache_graph_version: int = -1


def _compute_risk_hash(graph: CityGraph) -> int:
    """
    Use graph.version if available (new city_graph.py), otherwise
    fall back to a hash of all risk_multiplier values.
    """
    if hasattr(graph, "version"):
        return graph.version
    return hash(tuple(n.risk_multiplier for n in graph.all_nodes()))


def get_cached_fitness(chromosome: List[CityNode], graph: CityGraph) -> float:
    """
    Returns cached fitness if the graph's risk state is unchanged.
    Clears entire cache on risk change (Challenge 5 updated risk weights).

    Cache key: sorted tuple of (row, col) pairs — order-independent.
    Sorting makes [A,B,C] identical to [C,A,B].
    """
    global _fitness_cache, _cache_graph_version

    current_version = _compute_risk_hash(graph)
    if current_version != _cache_graph_version:
        _fitness_cache.clear()
        _cache_graph_version = current_version

    chrom_key = tuple(sorted((n.row, n.col) for n in chromosome))

    if chrom_key not in _fitness_cache:
        _fitness_cache[chrom_key] = calculate_minimax_fitness(chromosome, graph)

    return _fitness_cache[chrom_key]


def clear_fitness_cache():
    """Force-clears the cache. Useful for testing and manual re-triggers."""
    global _fitness_cache, _cache_graph_version
    _fitness_cache.clear()
    _cache_graph_version = -1