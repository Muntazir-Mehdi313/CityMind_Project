# =============================================================================
# algorithms/dstar_lite.py — D* Lite Incremental Pathfinder  (Challenge 4)
# =============================================================================
#
# D* Lite (Koenig & Likhachev, 2002) is an incremental heuristic search that
# plans from GOAL → START and repairs the plan locally whenever the graph
# changes (road blocked, risk updated), instead of replanning from scratch.
#
# Why D* Lite for CityMind?
# -------------------------
# Ambulances drive toward the emergency.  Floods and accidents block roads
# mid-route.  A full Dijkstra/A* re-plan from scratch is O(V log V) each time.
# D* Lite repairs only the affected portions of the search tree — typically
# O(k log k) where k ≪ V — making it ideal for real-time city routing.
#
# Search direction: GOAL → START  (backwards from the emergency to the ambulance)
# Heuristic:        Manhattan distance × min_risk_multiplier  (admissible)
# Edge cost:        node.effective_cost_to(neighbor)  =  base_weight × dest.risk_multiplier
#
# Observer integration
# --------------------
# DStarLite.register_with_graph(graph) subscribes to three graph events:
#   "road_blocked"   — the blocked edge is invalidated; plan is repaired
#   "road_restored"  — the restored edge may offer a shorter path; plan is repaired
#   "risk_updated"   — all edges touching the node get new costs; plan is repaired
#   "risk_map_applied" — same, for a batch of nodes
#
# Public API
# ----------
#   planner = DStarLite(graph, start_node, goal_node)
#   planner.initialize()                    # compute initial plan
#   path = planner.get_path()               # list[CityNode] from start → goal
#   planner.move_start(new_start)           # ambulance moved; update km (k modifier)
#   planner.register_with_graph(graph)      # subscribe to graph events
#   planner.replan()                        # force full consistency pass (rarely needed)
#
# =============================================================================

from __future__ import annotations

import heapq
import math
from typing import Optional

from models.city_node import CityNode
from models.city_graph import CityGraph
from config import RISK_MULTIPLIER, RISK_LOW


# ---------------------------------------------------------------------------
# Internal priority-queue entry  (heap key, node)
# ---------------------------------------------------------------------------

class _PQEntry:
    """
    Wrapper for heap entries so CityNode objects never need __lt__.
    Entries are compared first by k1, then by k2 (standard D* Lite tie-breaking).
    """
    __slots__ = ("k1", "k2", "node", "valid")

    def __init__(self, k1: float, k2: float, node: CityNode):
        self.k1 = k1
        self.k2 = k2
        self.node = node
        self.valid = True  # set to False when a stale entry is superseded

    def __lt__(self, other: "_PQEntry") -> bool:
        return (self.k1, self.k2) < (other.k1, other.k2)


# ---------------------------------------------------------------------------
# DStarLite
# ---------------------------------------------------------------------------

class DStarLite:
    """
    D* Lite planner.  One instance per ambulance-to-emergency pair.

    Parameters
    ----------
    graph : CityGraph
        The shared city graph (single source of truth).
    start : CityNode
        The ambulance's current position.
    goal  : CityNode
        The emergency / destination node.
    """

    # The minimum risk multiplier across all risk levels — used in the
    # admissible heuristic so h(n) ≤ true cost.
    _MIN_RISK = min(RISK_MULTIPLIER.values())

    def __init__(self, graph: CityGraph, start: CityNode, goal: CityNode):
        self._graph = graph
        self._start = start
        self._goal  = goal

        # D* Lite g and rhs tables  (keyed by CityNode)
        self._g:   dict[CityNode, float] = {}
        self._rhs: dict[CityNode, float] = {}

        # km accumulates heuristic shifts when start moves
        self._km: float = 0.0

        # Priority queue: list of _PQEntry (lazy-deletion heap)
        self._heap: list[_PQEntry] = []
        # Canonical entry per node (for O(1) validity check + key lookup)
        self._entries: dict[CityNode, _PQEntry] = {}

        # Last position of start (needed for km update on move_start)
        self._last_start: CityNode = start

        self._initialized = False

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Set up tables and compute the initial plan.
        Must be called once before get_path() or any update.
        """
        # All nodes: g = rhs = ∞  except goal: rhs = 0
        for node in self._graph.all_nodes():
            self._g[node]   = math.inf
            self._rhs[node] = math.inf

        self._rhs[self._goal] = 0.0
        self._push(self._goal, self._calc_key(self._goal))

        self._compute_shortest_path()
        self._initialized = True

    def get_path(self) -> list[CityNode]:
        """
        Return the current optimal path from start → goal as a list of nodes.
        Returns an empty list if no path exists.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before get_path().")

        if self._g[self._start] == math.inf:
            return []  # unreachable

        path = [self._start]
        current = self._start

        # Greedy walk: always step to the neighbor with smallest g + edge_cost
        visited = {current}
        while current is not self._goal:
            best_neighbor: Optional[CityNode] = None
            best_cost = math.inf

            for nb, _ in current.get_neighbors().items():
                if not nb.is_accessible:
                    continue
                cost = current.effective_cost_to(nb) + self._g.get(nb, math.inf)
                if cost < best_cost:
                    best_cost = cost
                    best_neighbor = nb

            if best_neighbor is None or best_neighbor in visited:
                return []  # path broken — caller should replan or wait

            visited.add(best_neighbor)
            path.append(best_neighbor)
            current = best_neighbor

        return path

    def move_start(self, new_start: CityNode) -> None:
        """
        Call this after the ambulance moves one step.
        Updates km so heuristic keys stay consistent.
        """
        self._km += self._heuristic(self._last_start, new_start)
        self._last_start = new_start
        self._start = new_start

    def register_with_graph(self, graph: CityGraph) -> None:
        """
        Subscribe to graph mutation events.  Call once after initialize().
        The planner will self-repair whenever roads or risks change.
        """
        graph.register_observer(self._on_graph_event)

    def unregister_from_graph(self, graph: CityGraph) -> None:
        """Remove this planner's observer callback from the graph."""
        graph.unregister_observer(self._on_graph_event)

    def replan(self) -> None:
        """
        Force a full consistency pass.  Normally not needed — graph events
        trigger targeted updates.  Useful after bulk mutations.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before replan().")
        self._compute_shortest_path()

    def get_g(self, node: CityNode) -> float:
        """Return the current g-value of a node (shortest path cost from goal)."""
        return self._g.get(node, math.inf)

    def is_reachable(self) -> bool:
        """Return True if the goal is currently reachable from start."""
        return self._initialized and self._g[self._start] < math.inf

    # -----------------------------------------------------------------------
    # Graph event handler (Observer callback)
    # -----------------------------------------------------------------------

    def _on_graph_event(self, event_type: str, payload: dict) -> None:
        """Called by CityGraph._notify() whenever the graph mutates."""
        print(f"[D* OBSERVER] Received event: {event_type} with payload keys: {payload.keys()}")

        if not self._initialized:
            return

        if event_type in ("road_blocked", "road_restored"):
            # Invalidate both endpoints of the changed edge
            node_a: CityNode = payload["a"]
            node_b: CityNode = payload["b"]
            print(f"[D* OBSERVER] Road {event_type}: ({node_a.row},{node_a.col}) -> ({node_b.row},{node_b.col})")
            self._update_vertex(node_a)
            self._update_vertex(node_b)
            self._compute_shortest_path()
            print(f"[D* OBSERVER] Recomputed path after {event_type}")

        elif event_type == "risk_updated":
            # Risk change on a node affects all edges *to* that node
            node: CityNode = payload["node"]
            print(f"[D* OBSERVER] Risk updated: ({node.row},{node.col})")
            self._update_vertex(node)
            # Also update predecessors — their edge costs to `node` changed
            for pred in self._graph.grid_neighbors(node):
                if pred.has_road(node):
                    self._update_vertex(pred)
            self._compute_shortest_path()
            print(f"[D* OBSERVER] Recomputed path after risk_updated")

        elif event_type == "risk_map_applied":
            # Bulk risk update — collect all affected nodes and their neighbors
            risk_map: dict[tuple[int, int], str] = payload["risk_map"]
            affected: set[CityNode] = set()
            for (r, c) in risk_map:
                node = self._graph.node(r, c)
                affected.add(node)
                for pred in self._graph.grid_neighbors(node):
                    if pred.has_road(node):
                        affected.add(pred)
            for node in affected:
                self._update_vertex(node)
            self._compute_shortest_path()

        # road_added events (new edges after init) — treat like road_restored
        elif event_type == "road_added":
            node_a: CityNode = payload["a"]
            node_b: CityNode = payload["b"]
            self._update_vertex(node_a)
            self._update_vertex(node_b)
            self._compute_shortest_path()

    # -----------------------------------------------------------------------
    # Core D* Lite internals
    # -----------------------------------------------------------------------

    def _heuristic(self, a: CityNode, b: CityNode) -> float:
        """
        Admissible heuristic: Manhattan distance × minimum risk multiplier.
        Underestimates the true cost because risk_multiplier ≥ _MIN_RISK always.
        """
        return (abs(a.row - b.row) + abs(a.col - b.col)) * self._MIN_RISK

    def _calc_key(self, node: CityNode) -> tuple[float, float]:
        """
        D* Lite key:  [min(g,rhs) + h(start,node) + km,  min(g,rhs)]
        km shifts the heuristic as the start position moves.
        """
        min_gr = min(self._g[node], self._rhs[node])
        return (
            min_gr + self._heuristic(self._start, node) + self._km,
            min_gr,
        )

    def _push(self, node: CityNode, key: tuple[float, float]) -> None:
        """Insert or update a node's priority queue entry."""
        # Invalidate any existing entry
        old = self._entries.get(node)
        if old is not None:
            old.valid = False

        entry = _PQEntry(key[0], key[1], node)
        self._entries[node] = entry
        heapq.heappush(self._heap, entry)

    def _pop(self) -> tuple[tuple[float, float], CityNode] | None:
        """
        Pop the minimum valid entry.  Lazy-delete stale entries.
        Returns None if the heap is empty.
        """
        while self._heap:
            entry = heapq.heappop(self._heap)
            if entry.valid:
                # Invalidate so it can't be popped again
                entry.valid = False
                del self._entries[entry.node]
                return (entry.k1, entry.k2), entry.node
        return None

    def _top_key(self) -> tuple[float, float]:
        """
        Peek at the smallest key without popping.  Returns (inf, inf) if empty.
        Skips stale entries.
        """
        while self._heap:
            entry = self._heap[0]
            if entry.valid:
                return (entry.k1, entry.k2)
            heapq.heappop(self._heap)  # discard stale
        return (math.inf, math.inf)

    def _update_vertex(self, node: CityNode) -> None:
        """
        Recompute rhs for a node and (re)insert it into the queue if
        it is locally inconsistent (g ≠ rhs).

        D* Lite rhs rule:
          rhs(goal) = 0
          rhs(u)    = min over successors s of [cost(u,s) + g(s)]

        In a backwards search, "successors of u" = road neighbors of u.
        """
        if node is not self._goal:
            best = math.inf
            if node.is_accessible:
                for nb, _ in node.get_neighbors().items():
                    if nb.is_accessible:
                        c = node.effective_cost_to(nb) + self._g[nb]
                        if c < best:
                            best = c
            self._rhs[node] = best

        # Remove from queue (will be re-added if inconsistent)
        old = self._entries.get(node)
        if old is not None:
            old.valid = False
            del self._entries[node]

        if self._g[node] != self._rhs[node]:
            self._push(node, self._calc_key(node))

    def _compute_shortest_path(self) -> None:
        """
        Main D* Lite loop: process nodes from the queue until the start node
        is locally consistent and its key is optimal.

        This is identical to Algorithm 2 in Koenig & Likhachev (2002), adapted
        for an undirected grid graph with risk-weighted edge costs.
        """
        while True:
            top_key = self._top_key()
            start_key = self._calc_key(self._start)

            # Termination: start is consistent and no node in the queue has a
            # smaller key than the start's current key.
            if top_key >= start_key and self._rhs[self._start] == self._g[self._start]:
                break

            result = self._pop()
            if result is None:
                break

            k_old, u = result
            k_new = self._calc_key(u)

            if k_old < k_new:
                # Key increased (overconsistent → reinsert with updated key)
                self._push(u, k_new)

            elif self._g[u] > self._rhs[u]:
                # Overconsistent: g > rhs → set g = rhs (expand node)
                self._g[u] = self._rhs[u]
                # Update all predecessors (road neighbors in undirected graph)
                for pred in u.get_neighbors():
                    self._update_vertex(pred)

            else:
                # Underconsistent: g ≤ rhs → set g = ∞ and update u + preds
                self._g[u] = math.inf
                self._update_vertex(u)
                for pred in u.get_neighbors():
                    self._update_vertex(pred)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def run_dstar_lite(
    graph: CityGraph,
    start: CityNode,
    goal: CityNode,
    subscribe: bool = True,
) -> DStarLite:
    """
    Build, initialize, and optionally subscribe a DStarLite planner.

    Parameters
    ----------
    graph     : the shared CityGraph
    start     : ambulance / agent start node
    goal      : emergency / destination node
    subscribe : if True, register with graph for automatic replanning

    Returns
    -------
    A ready-to-use DStarLite instance.  Call planner.get_path() immediately.

    Example
    -------
    >>> planner = run_dstar_lite(graph, ambulance_node, emergency_node)
    >>> path = planner.get_path()
    >>> print([str(n) for n in path])
    """
    print(f"[D* INIT] Creating D* Lite planner: start=({start.row},{start.col}), goal=({goal.row},{goal.col}), subscribe={subscribe}")
    planner = DStarLite(graph, start, goal)
    planner.initialize()
    print(f"[D* INIT] Planner initialized, path found: {planner.get_path()[:3]}... ({len(planner.get_path())} nodes)")
    if subscribe:
        print(f"[D* INIT] Registering planner with graph for automatic updates")
        planner.register_with_graph(graph)
    return planner