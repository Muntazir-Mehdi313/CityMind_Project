# =============================================================================
# algorithms/ambulance_ga.py — Challenge 3 Public Entry Point
# =============================================================================
# This file is the thin wrapper the server.py and main.py call.
# The real implementation lives in challenge3/placer.py and challenge3/ga_core.py
#
# server.py calls:   place_ambulances(graph, count=3)
# main.py calls:     place_ambulances(graph, count=3)
#
# Returns: list of CityNode objects where ambulances were placed
# =============================================================================

from models.city_graph import CityGraph
from models.city_node import CityNode
from challenge3.ga_core import run_ga, GA_CONFIG
from typing import List


def place_ambulances(graph: CityGraph, count: int = 3) -> List[CityNode]:
    """
    Public entry point for Challenge 3 — Ambulance Placement via Genetic Algorithm.

    Called by:
        - main.py at simulation startup (after Challenge 5 has set risk weights)
        - server.py /api/challenge/3/run endpoint

    Parameters
    ----------
    graph : CityGraph
        The shared city graph. Must already have:
          - location types set (Challenge 1 done)
          - roads built (Challenge 2 done)
          - risk_index values set (Challenge 5 done, ideally)

    count : int
        Number of ambulances to place. Default 3.
        Changing this at viva: update GA_CONFIG["num_ambulances"] and pass
        a modified config to run_ga().

    Returns
    -------
    List[CityNode]
        The 3 nodes where ambulances were placed.
        Each node has node.ambulance_id set (0, 1, or 2).
    """
    config = dict(GA_CONFIG)
    config["num_ambulances"] = count

    best_chromosome, best_fitness = run_ga(graph, config)

    print(f"[C3] Ambulance placement complete.")
    print(f"  Worst-case response distance : {best_fitness:.4f}")
    print(f"  Positions                    : {[(n.row, n.col) for n in best_chromosome]}")

    return best_chromosome