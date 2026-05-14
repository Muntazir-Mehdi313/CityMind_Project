# challenge5/features.py — Feature Extraction Pipeline (Phase 8)
# classes used here:
#   Uses models.city_node.CityNode  and  models.city_graph.CityGraph
#

import heapq
import numpy as np
from typing import List, Tuple, Dict

from models.city_node import CityNode
from models.city_graph import CityGraph
from config import LOC_INDUSTRIAL


def compute_industrial_proximity(graph: CityGraph) -> Dict[CityNode, float]:
    """
    For every node, computes how close it is to the nearest Industrial zone.

    Method: Multi-Source Dijkstra from ALL industrial nodes 

    Result: proximity = 1 - (dist / max_dist)
      Nodes CLOSE to industry → HIGH score (near 1.0)
      Nodes FAR from industry  → LOW score (near 0.0)

    Returns:
        Dict mapping every CityNode → industrial_proximity float [0, 1]
        All zeros if no Industrial nodes exist.
    """
    industrial_nodes = graph.nodes_of_type(LOC_INDUSTRIAL)

    if not industrial_nodes:
        print("[C5-Features] Warning: No Industrial nodes found — "
              "all industrial_proximity values set to 0.0.")
        return {node: 0.0 for node in graph.all_nodes()}

    dist_to_industry: Dict[CityNode, float] = {}
    pq = [(0.0, id(n), n) for n in industrial_nodes]
    heapq.heapify(pq)

    while pq:
        cost, _, node = heapq.heappop(pq)
        if node in dist_to_industry:
            continue
        dist_to_industry[node] = cost
        for neighbor, base_weight in node.get_neighbors().items():
            if neighbor not in dist_to_industry:
                heapq.heappush(pq, (cost + base_weight, id(neighbor), neighbor))

    finite_dists = [d for d in dist_to_industry.values() if d != float('inf')]
    max_dist = max(finite_dists) if finite_dists else 1.0
    if max_dist == 0:
        max_dist = 1.0

    proximity: Dict[CityNode, float] = {}
    for node in graph.all_nodes():
        d = dist_to_industry.get(node, float('inf'))
        proximity[node] = 0.0 if d == float('inf') else 1.0 - (d / max_dist)

    return proximity


def extract_and_normalize_features(
        graph: CityGraph
) -> Tuple[List[CityNode], np.ndarray, np.ndarray]:
    """
    Extracts [population_density, industrial_proximity] for every node
    and applies Min-Max normalization to scale both features to [0, 1].

    Returns:
        nodes_list    : ordered list of all CityNode objects
        X_normalized  : numpy array (n_nodes, 2), values in [0, 1]
        X_raw         : numpy array (n_nodes, 2), original un-scaled values
    """
    nodes_list    = list(graph.all_nodes())
    proximity_map = compute_industrial_proximity(graph)

    X_raw = np.array(
        [[node.population_density, proximity_map[node]] for node in nodes_list],
        dtype=float
    )

    feature_min   = X_raw.min(axis=0)
    feature_max   = X_raw.max(axis=0)
    feature_range = feature_max - feature_min
    feature_range[feature_range == 0] = 1.0   # avoid division by zero

    X_normalized = (X_raw - feature_min) / feature_range

    print(f"\n[C5-Features] Extracted features for {len(nodes_list)} nodes.")
    print(f"  Population density   — raw range : [{feature_min[0]:.2f}, {feature_max[0]:.2f}]")
    print(f"  Industrial proximity — raw range : [{X_raw[:,1].min():.4f}, {X_raw[:,1].max():.4f}]")
    print(f"  X_normalized shape   : {X_normalized.shape}")

    return nodes_list, X_normalized, X_raw