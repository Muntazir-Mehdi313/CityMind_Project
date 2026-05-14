# =============================================================================
# models/city_node.py — The CityNode Class
# =============================================================================
# Every cell in the grid is one CityNode instance.
# The node owns its adjacency list (neighbors dict), so there is no separate
# edge table anywhere in the system.  All reads and writes go through the
# methods below — never access _neighbors directly from outside this class.

from config import (
    LOC_EMPTY,
    RISK_LOW, RISK_MULTIPLIER,
    WEIGHT_STANDARD,
)


class CityNode:
    """
    Represents one cell in the city grid.

    Coordinates
    -----------
    row, col : int
        Zero-based grid position.  (0,0) is top-left.
        Used by A* / D* Lite as the heuristic anchor (Manhattan distance).

    Static properties  (set once by Challenge 1, never changed afterward)
    -----------------
    location_type     : str   — one of the LOC_* constants in config.py
    population_density: float — people count; input to K-Means in C5

    Dynamic properties  (updated during the simulation)
    ------------------
    risk_level        : str   — RISK_LOW / RISK_MEDIUM / RISK_HIGH  (C5 output)
    risk_multiplier   : float — derived from risk_level via RISK_MULTIPLIER dict
    is_accessible     : bool  — False when all roads to this node are blocked
    ambulance_id      : int | None — which ambulance (0/1/2) is stationed here

    Adjacency
    ---------
    _neighbors : dict[CityNode -> float]
        Maps each directly connected neighbour to its current base edge weight.
        Effective traversal cost = base_weight * destination.risk_multiplier.
        Blocked roads are REMOVED from this dict (not set to infinity) so that
        any iteration over neighbours automatically skips them.
    """

    def __init__(self, row: int, col: int):
        # --- coordinates ---
        self.row = row
        self.col = col

        # --- static properties (written once by C1) ---
        self.location_type      = LOC_EMPTY
        self.population_density = 0.0

        # --- dynamic properties (updated during simulation) ---
        self.risk_level      = RISK_LOW
        self.risk_multiplier = RISK_MULTIPLIER[RISK_LOW]
        self.is_accessible   = True
        self.ambulance_id    = None   # set by C3 GA

        # --- adjacency (written by C2, mutated by flood events) ---
        self._neighbors: dict["CityNode", float] = {}

    # Adjacency helpers


    def add_road(self, neighbor: "CityNode", weight: float = WEIGHT_STANDARD) -> None:
        """
        Create a bidirectional road between self and neighbor.
        Called exclusively by CityGraph / Challenge 2.
        Overwrites the weight if the road already exists.
        """
        self._neighbors[neighbor] = weight
        neighbor._neighbors[self] = weight

    def remove_road(self, neighbor: "CityNode") -> None:
        """
        Block the road between self and neighbor in O(1).
        Raises KeyError if the road does not exist — callers should check first
        with has_road() to avoid silent bugs.
        """
        if neighbor not in self._neighbors:
            raise KeyError(
                f"No road exists between {self} and {neighbor}. "
                "Check has_road() before calling remove_road()."
            )
        del self._neighbors[neighbor]
        del neighbor._neighbors[self]

    def has_road(self, neighbor: "CityNode") -> bool:
        """Return True if a direct (unblocked) road connects self to neighbor."""
        return neighbor in self._neighbors

    def get_base_weight(self, neighbor: "CityNode") -> float:
        """
        Return the base edge weight to neighbor (before risk multiplier).
        Raises KeyError if no road exists — intentional: callers should never
        query weight for a blocked road.
        """
        return self._neighbors[neighbor]

    def effective_cost_to(self, neighbor: "CityNode") -> float:
        """
        Return the full traversal cost to neighbor:
            base_weight × destination.risk_multiplier

        This is what A* / D* Lite use.  The risk multiplier lives on the
        destination node (per the design doc, Section 3.3).
        """
        base = self._neighbors[neighbor]
        return base * neighbor.risk_multiplier

    def get_neighbors(self) -> dict["CityNode", float]:
        """
        Return the raw neighbors dict (node -> base_weight).
        Read-only view — do not mutate the returned dict.
        Use add_road / remove_road for mutations.
        """
        return self._neighbors

    # Risk update  (called by Challenge 5 after ML classification)


    def set_risk(self, risk_level: str) -> None:
        """
        Update the node's risk level and pre-compute the multiplier.
        Pre-computing here means the router pays zero cost at traversal time.
        """
        if risk_level not in RISK_MULTIPLIER:
            raise ValueError(
                f"Invalid risk_level '{risk_level}'. "
                f"Must be one of {list(RISK_MULTIPLIER.keys())}."
            )
        self.risk_level      = risk_level
        self.risk_multiplier = RISK_MULTIPLIER[risk_level]

    # Accessibility


    def mark_inaccessible(self) -> None:
        """
        Mark this node as unreachable (e.g. all roads flooded).
        Pathfinding modules check is_accessible before expanding a node.
        """
        self.is_accessible = False

    def mark_accessible(self) -> None:
        """Restore accessibility (e.g. road cleared after a flood event)."""
        self.is_accessible = True

    # Dunder helpers

    def __repr__(self) -> str:
        return f"CityNode({self.row},{self.col})[{self.location_type}]"

    def __hash__(self) -> int:
        """Nodes are hashed by position so they work as dict keys and in sets."""
        return hash((self.row, self.col))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CityNode):
            return NotImplemented
        return self.row == other.row and self.col == other.col
