# =============================================================================
# models/city_graph.py — The CityGraph Manager
# =============================================================================
# This class owns the single 2-D array of CityNode objects.
# RULE: No other module keeps its own copy of the grid.
#       All reads and writes go through CityGraph methods.
#    single source of truth.


from collections import deque


from config import (
    GRID_ROWS, GRID_COLS,
    LOC_HOSPITAL, LOC_AMBULANCE,
    RISK_LOW, RISK_MULTIPLIER,
)
from models.city_node import CityNode


class CityGraph:
    """
    The central graph manager for the CityMind system.

    Internal layout
    ---------------
    _grid : list[list[CityNode]]
        _grid[row][col] gives the node at that position.
        Row 0 is the top row; col 0 is the leftmost column.

    Observer pattern
    ----------------
    Modules (C3, C4) can register callbacks via register_observer().
    Whenever a road is blocked or a risk level changes, all observers
    are notified automatically — this is how D* Lite hears about floods
    without polling the graph every step.
    """

    def __init__(self, rows: int = GRID_ROWS, cols: int = GRID_COLS):
        self.rows = rows
        self.cols = cols
        self.version = 0

        # Build the grid — one CityNode per cell
        self._grid: list[list[CityNode]] = [
            [CityNode(r, c) for c in range(cols)]
            for r in range(rows)
        ]

        # Observer callbacks: fn(event_type: str, payload: dict) -> None
        # event_type is one of: "road_blocked", "road_restored", "risk_updated"
        self._observers = []
        # Convenience references set after Challenge 1 runs
        self.primary_hospital = None
        self.ambulance_depot = None

    # ------------------------------------------------------------------
    # Grid access
    # ------------------------------------------------------------------

    def node(self, row: int, col: int) -> CityNode:
        """Return the node at (row, col). Raises IndexError for out-of-bounds."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError(
                f"Position ({row},{col}) is outside the {self.rows}×{self.cols} grid."
            )
        return self._grid[row][col]

    def all_nodes(self):
        """Yield every CityNode in row-major order."""
        for row in self._grid:
            yield from row

    def nodes_of_type(self, location_type: str) -> list[CityNode]:
        """Return all nodes whose location_type matches the given constant."""
        return [n for n in self.all_nodes() if n.location_type == location_type]

    # ------------------------------------------------------------------
    # Spatial neighbours (grid adjacency, NOT road adjacency)
    # ------------------------------------------------------------------

    def grid_neighbors(self, node: CityNode) -> list[CityNode]:
        """
        Return the up-to-4 grid-adjacent cells (N, S, E, W).
        Does NOT check whether a road exists — use node.get_neighbors() for that.
        Used by Challenge 1 (CSP proximity checks) and Challenge 2 (Kruskal
        candidate edge generation).
        """
        candidates = [
            (node.row - 1, node.col),  # North
            (node.row + 1, node.col),  # South
            (node.row,     node.col - 1),  # West
            (node.row,     node.col + 1),  # East
        ]
        return [
            self._grid[r][c]
            for r, c in candidates
            if 0 <= r < self.rows and 0 <= c < self.cols
        ]

    # ------------------------------------------------------------------
    # Road operations  (Challenge 2 writes; simulation events mutate)
    # ------------------------------------------------------------------

    def add_road(self, node_a: CityNode, node_b: CityNode, weight: float) -> None:
        """
        Build a road between node_a and node_b with the given base weight.
        Delegates to CityNode.add_road() and notifies observers.
        """
        node_a.add_road(node_b, weight)
        self.version += 1
        self._notify("road_added", {"a": node_a, "b": node_b, "weight": weight})

    def set_edge(self, node_a: CityNode, node_b: CityNode, weight: float) -> None:
        """
        Alias for add_road() used by Challenge 2.
        This creates or updates the bidirectional edge weight between nodes.
        """
        self.add_road(node_a, node_b, weight)

    def get_edge_weight(self, node_a: CityNode, node_b: CityNode) -> float | None:
        """
        Return the current base weight of the road between node_a and node_b.
        Returns None if no road exists.
        """
        if node_a.has_road(node_b):
            return node_a.get_base_weight(node_b)
        return None

    def remove_edge(self, node_a: CityNode, node_b: CityNode) -> None:
        """
        Remove the bidirectional road between node_a and node_b without
        firing a graph event. Used by algorithmic checks and temporary
        modifications.
        """
        if not node_a.has_road(node_b):
            raise ValueError(f"No road between {node_a} and {node_b} to remove.")
        node_a.remove_road(node_b)
        self.version += 1

    def block_road(self, node_a: CityNode, node_b: CityNode) -> None:
        """
        Block (flood / accident) the road between node_a and node_b.
        Removes the edge from both nodes' adjacency dicts and fires the
        'road_blocked' event so D* Lite can replan immediately.
        """
        if not node_a.has_road(node_b):
            raise ValueError(f"No road between {node_a} and {node_b} to block.")
        node_a.remove_road(node_b)
        self.version += 1
        self._notify("road_blocked", {"a": node_a, "b": node_b})

    def restore_road(self, node_a: CityNode, node_b: CityNode, weight: float) -> None:
        """Re-open a previously blocked road (optional — used by simulation)."""
        node_a.add_road(node_b, weight)
        self.version += 1
        self._notify("road_restored", {"a": node_a, "b": node_b, "weight": weight})

    # ------------------------------------------------------------------
    # Risk updates  (Challenge 5 writes)
    # ------------------------------------------------------------------

    def set_risk(self, node: CityNode, risk_level: str) -> None:
        """
        Update a node's risk level and notify observers.
        Challenge 5 calls this after the Decision Tree classifier runs.
        """
        node.set_risk(risk_level)
        # Mark node as recently updated so server/UI can detect risk changes
        node.just_updated = True
        self.version += 1
        self._notify("risk_updated", {"node": node, "risk_level": risk_level})

    def apply_risk_map(self, risk_map: dict[tuple[int, int], str]) -> None:
        """
        Bulk-apply a {(row,col): risk_level} mapping from Challenge 5.
        More efficient than calling set_risk() in a loop because it fires
        one summary notification instead of one per node.
        """
        for (r, c), level in risk_map.items():
            self._grid[r][c].set_risk(level)
            # Mark nodes updated in bulk so server/UI can pick them up
            self._grid[r][c].just_updated = True
        self.version += 1
        self._notify("risk_map_applied", {"risk_map": risk_map})

    def clear_just_updated_flags(self) -> None:
        """Reset the UI/server update marker after changes have been consumed."""
        for node in self.all_nodes():
            if hasattr(node, "just_updated"):
                node.just_updated = False

    # ------------------------------------------------------------------
    # Pathfinding utilities  (used by C2, C3, C4)
    # ------------------------------------------------------------------

    def bfs_hops(self, start: CityNode, max_hops: int) -> set[CityNode]:
        """
        Return all nodes reachable from start within max_hops GRID steps
        (not road steps — ignores whether a road exists).
        Used by Challenge 1 CSP to check proximity constraints:
            "every residential area must be within 3 road hops of a hospital"
        Uses grid adjacency so it works before roads are built.
        """
        visited = {start}
        frontier = deque([(start, 0)])
        while frontier:
            current, hops = frontier.popleft()
            if hops >= max_hops:
                continue
            for nb in self.grid_neighbors(current):
                if nb not in visited:
                    visited.add(nb)
                    frontier.append((nb, hops + 1))
        return visited

    def dijkstra(self, start, goal=None, use_risk=True):
        """
        Standard Dijkstra over the ROAD graph (follows _neighbors dicts).

        Parameters
        ----------
        start    : source node
        goal     : if provided, stops early when goal is settled
        use_risk : if True, uses node.effective_cost_to() (base × risk_multiplier)
                   if False, uses base weight only (for Challenge 2 path checks)

        Returns
        -------
        dist  : dict[node -> shortest distance from start]
        prev  : dict[node -> predecessor on shortest path]

        Usage
        -----
        C2 uses this (use_risk=False) to verify Hospital-Depot connectivity.
        C3 uses this (use_risk=True) to compute worst-case response distances.
        C4 (D* Lite) has its own incremental implementation — it doesn't call
        this method, but uses the same edge cost logic.
        """
        import heapq

        dist = {start: 0.0}
        prev = {start: None}
        heap = [(0.0, id(start), start)]  # (cost, tiebreak, node)

        while heap:
            cost, _, current = heapq.heappop(heap)

            if goal and current is goal:
                break

            if cost > dist.get(current, float("inf")):
                continue  # stale entry

            if not current.is_accessible:
                continue

            for neighbor, base_w in current.get_neighbors().items():
                if not neighbor.is_accessible:
                    continue
                edge_cost = (
                    current.effective_cost_to(neighbor) if use_risk else base_w
                )
                new_cost = cost + edge_cost
                if new_cost < dist.get(neighbor, float("inf")):
                    dist[neighbor] = new_cost
                    prev[neighbor] = current
                    heapq.heappush(heap, (new_cost, id(neighbor), neighbor))

        return dist, prev

    def reconstruct_path(self, prev, goal):
        if goal not in prev:
            return []
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = prev[current]
        path.reverse()
        return path

    def count_independent_paths(
        self, source: CityNode, target: CityNode
    ) -> int:
        """
        Count edge-disjoint paths between source and target using a simple
        max-flow approach (each road can carry 1 unit of flow).

        Challenge 2 uses this to verify the "two independent paths between
        Primary Hospital and Ambulance Depot" safety constraint.
        Returns 2 if the constraint is satisfied, 1 if only one path exists,
        0 if disconnected.  Stops counting at 2 (we only need to know ≥2).
        """
        def bfs_augmenting(residual: dict, s: CityNode, t: CityNode) -> list[CityNode]:
            """BFS to find an augmenting path in the residual graph."""
            visited = {s}
            queue = deque([(s, [s])])
            while queue:
                node, path = queue.popleft()
                for nb in residual.get(node, {}):
                    if nb not in visited and residual[node][nb] > 0:
                        visited.add(nb)
                        new_path = path + [nb]
                        if nb is t:
                            return new_path
                        queue.append((nb, new_path))
            return []

        # Build unit-capacity residual graph from road adjacency
        residual: dict[CityNode, dict[CityNode, int]] = {}
        for node in self.all_nodes():
            for nb in node.get_neighbors():
                residual.setdefault(node, {})[nb] = 1
                residual.setdefault(nb, {})[node] = 1

        flow = 0
        while flow < 2:
            path = bfs_augmenting(residual, source, target)
            if not path:
                break
            # Update residual along path
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                residual[u][v] -= 1
                residual[v][u] += 1
            flow += 1

        return flow

    # ------------------------------------------------------------------
    # Convenience setters  (Challenge 1 uses these)
    # ------------------------------------------------------------------

    def set_location_type(self, node: CityNode, loc_type: str) -> None:
        """Set a node's location type. Challenge 1 CSP calls this."""
        node.location_type = loc_type
        # Cache hospital and depot references for easy access by other modules
        if loc_type == LOC_HOSPITAL and self.primary_hospital is None:
            self.primary_hospital = node
        if loc_type == LOC_AMBULANCE and self.ambulance_depot is None:
            self.ambulance_depot = node

    def set_population_density(self, node: CityNode, density: float) -> None:
        """Set population density. Challenge 1 or dataset generation calls this."""
        node.population_density = density

    # ------------------------------------------------------------------
    # Observer pattern
    # ------------------------------------------------------------------

    def register_observer(self, callback) -> None:
        """
        Register a function to be called on every graph-mutating event.
        Signature: callback(event_type: str, payload: dict) -> None

        Challenge 4 (D* Lite) registers here to detect road_blocked events.
        The UI dashboard registers here to trigger canvas redraws.
        """
        self._observers.append(callback)

    def unregister_observer(self, callback) -> None:
        """Remove a previously registered observer. Safe to call if not registered."""
        if callback in self._observers:
            self._observers.remove(callback)

    def _notify(self, event_type: str, payload: dict) -> None:
        """Internal — fire all registered observers."""
        for obs in self._observers:
            obs(event_type, payload)

    # ------------------------------------------------------------------
    # Debug / introspection helpers
    # ------------------------------------------------------------------

    def road_count(self) -> int:
        """Total number of roads (each bidirectional road counted once)."""
        return sum(len(n.get_neighbors()) for n in self.all_nodes()) // 2

    def total_road_cost(self) -> float:
        """Sum of all base edge weights (each road counted once)."""
        seen = set()
        total = 0.0
        for node in self.all_nodes():
            for nb, w in node.get_neighbors().items():
                key = (min(id(node), id(nb)), max(id(node), id(nb)))
                if key not in seen:
                    seen.add(key)
                    total += w
        return total

    def summary(self) -> str:
        """One-line summary for the event log."""
        return (
            f"Grid {self.rows}×{self.cols} | "
            f"Roads: {self.road_count()} | "
            f"Total cost: {self.total_road_cost():.1f}"
        )

    def __repr__(self) -> str:
        return f"CityGraph({self.rows}×{self.cols})"
