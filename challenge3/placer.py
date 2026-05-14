# challenge3/placer.py — AmbulancePlacer: GA Lifecycle Manager (Challenge 3)
# other classes used here :
#   Uses models.city_node.CityNode  &  models.city_graph.CityGraph

from typing import List, Tuple, Optional

from models.city_node import CityNode
from models.city_graph import CityGraph
from challenge3.ga_core import run_ga, GA_CONFIG
from challenge3.fitness import get_average_risk


class AmbulancePlacer:
    """
    Manages GA execution lifecycle for Challenge 3.
    Holds the current ambulance placement and decides when to re-run the GA
    based on flood accumulation and risk profile shifts from Challenge 5.
    """

    FLOOD_RETRIGGER_THRESHOLD = 3     # re-run after 3 accumulated floods (spec requirement)
    RISK_DELTA_THRESHOLD      = 0.15  # re-run if avg risk shifts by this much

    def __init__(self, graph: CityGraph, config: dict = GA_CONFIG, logger=None):  # FIX 3: add logger param
        self.graph  = graph
        self.config = config
        self.logger = logger  # FIX 3: store logger

        self.current_placement: List[CityNode] = []
        self.current_fitness:   float          = float('inf')

        self.flood_count_since_last_run: int   = 0
        self.last_avg_risk:              float = get_average_risk(graph)
        self.on_placement_changed: Optional[callable] = None  # Callback for D* updates

    # Public interface
    
    def initial_placement(self) -> Tuple[List[CityNode], float]:
        """
        Run the GA once at simulation startup.
        Returns the placed nodes and the achieved minimax fitness.
        """
        print("[AmbulancePlacer] Running initial placement GA...")
        self.current_placement, self.current_fitness = run_ga(self.graph, self.config)
        self.last_avg_risk              = get_average_risk(self.graph)
        self.flood_count_since_last_run = 0
        return self.current_placement, self.current_fitness

    def notify_road_flooded(self, node_u: CityNode, node_v: CityNode):
        """
        Called by Challenge 4 each time a road is blocked by a flood.
        Accumulates flood count and triggers GA re-run at threshold.
        node_u, node_v are the two endpoints of the flooded road (for logging).
        """
        self.flood_count_since_last_run += 1
        print(f"[AmbulancePlacer] Road ({node_u.row},{node_u.col})↔"
              f"({node_v.row},{node_v.col}) flooded. "
              f"Flood count: {self.flood_count_since_last_run}/"
              f"{self.FLOOD_RETRIGGER_THRESHOLD}")

        if self.flood_count_since_last_run >= self.FLOOD_RETRIGGER_THRESHOLD:
            print(f"[AmbulancePlacer] {self.flood_count_since_last_run} floods "
                  f"accumulated → triggering GA re-run.")
            self._rerun_ga(reason="road_flood_threshold")

    def notify_risk_updated(self):
        """
        Called by Challenge 5 after updating risk levels on nodes.
        Triggers GA re-run if the average risk has shifted significantly.
        """
        current_avg = get_average_risk(self.graph)
        delta       = abs(current_avg - self.last_avg_risk)

        if delta > self.RISK_DELTA_THRESHOLD:
            print(f"[AmbulancePlacer] Risk profile shifted by {delta:.3f} "
                  f"(threshold={self.RISK_DELTA_THRESHOLD}) → triggering GA re-run.")
            self.last_avg_risk = current_avg
            self._rerun_ga(reason="risk_profile_shift")
        else:
            print(f"[AmbulancePlacer] Risk delta={delta:.3f} below threshold — "
                  f"no re-run needed.")

    def get_current_placement(self) -> List[CityNode]:
        """Return the current list of ambulance-placed nodes."""
        return self.current_placement

    def get_current_fitness(self) -> float:
        """Return the current minimax fitness (worst-case response distance)."""
        return self.current_fitness

    # Internal
    
    def _rerun_ga(self, reason: str):
        """
        Clears existing ambulance assignments, re-runs GA with reduced params, stores results.
        Uses a fast config (pop=60, gen=100) so live re-triggers finish quickly.
        """
        print(f"[AmbulancePlacer] Re-run reason: {reason}")

        # FIX 2: Clear both fields on existing placements
        for node in self.current_placement:
            node.ambulance_id   = None
            node.ambulance_here = False

        # Reduced parameters for speed during live simulation re-triggers
        fast_config = dict(self.config)
        fast_config["population_size"]  = 60
        fast_config["max_generations"]  = 100
        fast_config["stagnation_limit"] = 20

        self.current_placement, self.current_fitness = run_ga(self.graph, fast_config)
        self.flood_count_since_last_run = 0

        # FIX 3: Log GA_RERUN event
        if self.logger:
            new_positions = [(n.row, n.col) for n in self.current_placement]
            self.logger.log(0, "GA_RERUN",
                f"GA re-run ({reason}). New positions: {new_positions}. "
                f"Fitness: {self.current_fitness:.4f}."
            )

        # Notify server that placement changed (so D* planner can be updated)
        if self.on_placement_changed:
            self.on_placement_changed(self.current_placement, self.current_fitness)

        print(f"[AmbulancePlacer] Re-run complete. "
              f"New fitness: {self.current_fitness:.4f}")