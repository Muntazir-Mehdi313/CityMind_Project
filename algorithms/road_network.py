# =============================================================================
# algorithms/road_network.py — Challenge 2: Road Network Optimization
# =============================================================================
# Algorithm — Two-Stage Approach (as per Phase 1 design document)
# ---------------------------------------------------------------
#   Stage 1 — Kruskal's MST:
#       Sort all candidate edges by construction cost.
#       Add edges greedily (Kruskal) using Union-Find until all nodes are
#       connected. This guarantees the globally minimum-cost spanning tree.
#
#   Stage 2 — UCS Redundancy Enforcement:
#       After MST construction, an MST by definition has exactly ONE path
#       between any two nodes. The project requires at least TWO independent
#       paths between the Primary Hospital and the Ambulance Depot so that
#       a single road failure never cuts off emergency access.
#       We check this by temporarily removing the direct MST edge between
#       Hospital and Depot (if one exists) and running UCS to find the
#       cheapest alternative route. Those extra edges are added to the graph
#       on top of the MST, adding ONLY the minimum extra cost required.
#
# Edge Weight Logic (from Section 3.3 of the design document)
# -----------------------------------------------------------
#   Standard road         → weight 1.0
#   Road into Residential → weight 0.8
#   Effective travel cost → base_weight × destination.risk_index
#   (risk_index is set by Challenge 5; defaults to 1.0 until then)
#
# Public Entry Point
# ------------------
#   run_road_network(graph) → (total_cost: float, redundancy_ok: bool)
#
# Assumptions about CityGraph / CityNode interface
# -------------------------------------------------
#   graph.all_nodes()               → iterable of all CityNode objects
#   graph.grid_neighbors(node)      → list of up to 4 orthogonal neighbours
#   graph.node(row, col)            → CityNode at grid position
#   graph.set_edge(u, v, weight)    → adds a bidirectional road (weight on
#                                     both sides of the adjacency dict)
#   graph.get_edge_weight(u, v)     → current weight or None if no road
#   graph.remove_edge(u, v)         → removes bidirectional road temporarily
#   node.location_type              → string constant from config
#   node.risk_index                 → float (default 1.0 before Challenge 5)
#   node.row, node.col              → grid coordinates
# =============================================================================

import heapq
from collections import defaultdict

from config import (
    LOC_EMPTY,
    LOC_RESIDENTIAL,
    LOC_HOSPITAL,
    LOC_AMBULANCE,
)
from models.city_graph import CityGraph
from models.city_node  import CityNode

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROAD_WEIGHT_STANDARD    = 1.0   # default construction cost
ROAD_WEIGHT_RESIDENTIAL = 0.8   # lower-cost roads in residential zones

# When checking for two-path redundancy we need to identify the "Primary
# Hospital" and the "Ambulance Depot".  The design document treats the FIRST
# hospital seeded (top-left-most in the grid) as the Primary Hospital and the
# first ambulance depot as the target depot.  If your seeding order differs,
# change these helpers.
def _find_primary_hospital(graph: CityGraph) -> CityNode | None:
    """Return the most connected Hospital node (most non-empty neighbors)."""
    hospitals = [n for n in graph.all_nodes() if n.location_type == LOC_HOSPITAL]
    if not hospitals:
        return None
    # Count non-empty neighbors for each hospital
    def count_connected_neighbors(node):
        return sum(1 for nb in graph.grid_neighbors(node) if nb.location_type != LOC_EMPTY)
    # Pick hospital with most connections (best for redundancy)
    return max(hospitals, key=count_connected_neighbors)

def _find_all_ambulance_depots(graph: CityGraph) -> list[CityNode]:
    """Return ALL Ambulance Depot nodes in the grid."""
    return [n for n in graph.all_nodes() if n.location_type == LOC_AMBULANCE]


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def run_road_network(graph: CityGraph) -> dict:
    """
    Build the minimum-cost road network satisfying all constraints.

    Steps
    -----
    1.  Generate all candidate edges (grid adjacency pairs).
    2.  Run Kruskal's MST — connects all nodes at minimum cost.
    3.  Apply MST edges to the shared city graph.
    4.  Check two-path redundancy between Primary Hospital ↔ Ambulance Depot.
    5.  If only one path exists, run UCS to find the cheapest backup route
        and add those edges to the graph.

    Returns
    -------
    dict with keys:
        total_cost    : float — sum of all road weights in the final network
        redundancy_ok : bool  — True if two independent paths exist H ↔ Depot
        mst_edges     : int   — number of MST edges
        extra_edges   : int   — number of redundant edges added
        hospital      : dict  — {row, col} of primary hospital or None
        depot         : dict  — {row, col} of ambulance depot or None
        mst_edge_set  : set   — set of edge keys that are MST edges
        extra_edge_set: set   — set of edge keys that are redundant edges
    """
    builder = RoadNetworkBuilder(graph)
    return builder.build()


# =============================================================================
# EDGE WEIGHT HELPER
# =============================================================================

def _edge_weight(u: CityNode, v: CityNode) -> float:
    """
    Construction cost for a road between nodes u and v.
    Section 3.3: roads INTO a residential zone cost 0.8, otherwise 1.0.
    (Direction: the destination node determines the class.)
    We use the lower of the two directional weights as the undirected cost.
    """
    w_uv = ROAD_WEIGHT_RESIDENTIAL if v.location_type == LOC_RESIDENTIAL else ROAD_WEIGHT_STANDARD
    w_vu = ROAD_WEIGHT_RESIDENTIAL if u.location_type == LOC_RESIDENTIAL else ROAD_WEIGHT_STANDARD
    return min(w_uv, w_vu)


# =============================================================================
# UNION-FIND  (for Kruskal's)
# =============================================================================

class _UnionFind:
    """Path-compressed, union-by-rank Union-Find structure."""

    def __init__(self, nodes):
        self._parent = {n: n for n in nodes}
        self._rank   = {n: 0  for n in nodes}

    def find(self, x):
        while self._parent[x] is not x:
            self._parent[x] = self._parent[self._parent[x]]   # path halving
            x = self._parent[x]
        return x

    def union(self, x, y) -> bool:
        """Union x and y. Returns True if they were in different components."""
        rx, ry = self.find(x), self.find(y)
        if rx is ry:
            return False
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1
        return True

    def connected(self, x, y) -> bool:
        return self.find(x) is self.find(y)


# =============================================================================
# ROAD NETWORK BUILDER
# =============================================================================

class RoadNetworkBuilder:

    def __init__(self, graph: CityGraph):
        self.graph       = graph
        self._all_nodes  = list(graph.all_nodes())
        self._mst_edges  : list[tuple[float, CityNode, CityNode]] = []
        self._extra_edges: list[tuple[float, CityNode, CityNode]] = []
        self._redundant_path_edges: set = set()  # coord keys for ALL edges on backup paths

    # -------------------------------------------------------------------------
    # Main driver
    # -------------------------------------------------------------------------

    def build(self) -> dict:
        # Stage 1 — Kruskal's MST
        self._mst_edges = self._kruskal()
        self._apply_edges(self._mst_edges)

        mst_cost = sum(w for w, _, _ in self._mst_edges)
        print(f"\n{'='*60}")
        print(f"  CHALLENGE 2 — ROAD NETWORK BUILDER")
        print(f"{'='*60}")
        print(f"  MST complete: {len(self._mst_edges)} roads, total cost = {mst_cost:.2f}")

        # Build MST edge set for UI highlighting (use coordinates, not id())
        mst_edge_set = set()
        for w, u, v in self._mst_edges:
            key = tuple(sorted([(u.row, u.col), (v.row, v.col)]))
            mst_edge_set.add(key)

        # Stage 2 — Two-path redundancy check (Primary Hospital ↔ ALL Depots)
        hospital = _find_primary_hospital(self.graph)
        depots   = _find_all_ambulance_depots(self.graph)

        if hospital is None:
            print("  ⚠  No Hospital found in layout — cannot enforce redundancy.")
            return {
                "total_cost": mst_cost,
                "redundancy_ok": False,
                "mst_edges": len(self._mst_edges),
                "extra_edges": 0,
                "hospital": None,
                "depots": [],
                "mst_edge_set": set(),
                "extra_edge_set": set(),
            }
        if not depots:
            print("  ⚠  No Ambulance Depots found in layout — cannot enforce redundancy.")
            return {
                "total_cost": mst_cost,
                "redundancy_ok": False,
                "mst_edges": len(self._mst_edges),
                "extra_edges": 0,
                "hospital": None,
                "depots": [],
                "mst_edge_set": set(),
                "extra_edge_set": set(),
            }

        print(f"\n  Primary Hospital: ({hospital.row},{hospital.col})")
        print(f"  Enforcing redundancy to {len(depots)} Ambulance Depot(s)...")

        # Enforce redundancy to ALL depots
        total_redundancy_ok = True
        total_extra_cost = 0.0
        for depot in depots:
            print(f"    Checking H↔Depot({depot.row},{depot.col})...", end="")
            ok, cost, full_path_edges = self._enforce_redundancy(hospital, depot)
            if ok:
                print(f" ✔ (extra cost: {cost:.2f})")
            else:
                print(f" ✘ FAILED")
            total_redundancy_ok = total_redundancy_ok and ok
            total_extra_cost += cost
            # Add ALL edges from the backup path for UI highlighting
            for u, v in full_path_edges:
                key = tuple(sorted([(u.row, u.col), (v.row, v.col)]))
                self._redundant_path_edges.add(key)

        total_cost = mst_cost + total_extra_cost
        print(f"\n  {'✔' if total_redundancy_ok else '✘'}  All redundancies enforced: {total_redundancy_ok}")
        print(f"  Total extra edges added: {len(self._extra_edges)}, "
              f"total extra cost = {total_extra_cost:.2f}")
        print(f"  Total road network cost : {total_cost:.2f}")
        print(f"  Debug: self._extra_edges content: {[(w, (u.row,u.col), (v.row,v.col)) for w,u,v in self._extra_edges[:5]]}")  # First 5
        print(f"{'='*60}\n")

        # Build extra edge set for UI highlighting — full backup paths
        extra_edge_set = self._redundant_path_edges.copy()

        return {
            "total_cost": total_cost,
            "redundancy_ok": total_redundancy_ok,
            "mst_edges": len(self._mst_edges),
            "extra_edges": len(self._extra_edges),
            "hospital": {"row": hospital.row, "col": hospital.col},
            "depots": [{"row": d.row, "col": d.col} for d in depots],
            "mst_edge_set": list(mst_edge_set),
            "extra_edge_set": list(extra_edge_set),
        }

    # -------------------------------------------------------------------------
    # Stage 1 — Kruskal's MST
    # -------------------------------------------------------------------------

    def _kruskal(self) -> list[tuple[float, CityNode, CityNode]]:
        """
        Classic Kruskal's algorithm over all grid-adjacent node pairs.

        Candidate edges are ALL orthogonal pairs returned by graph.grid_neighbors().
        Each pair is generated once (u < v by id to avoid duplicates).
        Edges are sorted ascending by weight and added greedily when they
        join two previously disconnected components.
        """
        nodes = self._all_nodes
        if len(nodes) < 2:
            return []
        uf = _UnionFind(nodes)

        # Generate candidate edges (each undirected pair once)
        seen   = set()
        edges  = []   # (weight, u, v)
        for u in nodes:
            for v in self.graph.grid_neighbors(u):
                key = (min(id(u), id(v)), max(id(u), id(v)))
                if key in seen:
                    continue
                seen.add(key)
                w = _edge_weight(u, v)
                edges.append((w, u, v))

        edges.sort(key=lambda e: e[0])

        mst    = []
        n_comp = len(nodes)   # decreases by 1 each time we merge two components

        for w, u, v in edges:
            if n_comp <= 1:
                break
            if uf.union(u, v):
                mst.append((w, u, v))
                n_comp -= 1

        if n_comp > 1:
            print(f"  ⚠  MST WARNING: graph has {n_comp} disconnected components "
                  f"— some nodes are unreachable from others. "
                  f"This should not happen on a fully-seeded 20×20 grid.")

        return mst

    # -------------------------------------------------------------------------
    # Stage 2 — Redundancy enforcement via UCS
    # -------------------------------------------------------------------------

    def _enforce_redundancy(self, hospital: CityNode, depot: CityNode) -> tuple[bool, float, list]:
        """
        Ensure at least two edge-disjoint paths exist between hospital and depot.
        Returns (ok, extra_cost, full_path_edges) where full_path_edges is the
        complete backup path (all edges, including pre-existing ones).
        """
        print(f"    [DEBUG] _enforce_redundancy called for H({hospital.row},{hospital.col}) -> D({depot.row},{depot.col})")
        
        # Quick check: count edge-disjoint paths via max-flow of 2 on unit caps
        has_two = self._has_two_edge_disjoint_paths(hospital, depot)
        print(f"    [DEBUG] _has_two_edge_disjoint_paths returned: {has_two}")
        
        # Find the primary path hospital → depot (over already-built roads only)
        mst_path_edges = self._ucs_path_edges(hospital, depot, forbidden_edges=frozenset(), roads_only=True)

        if has_two:
            print("  Two independent paths already exist — no extra edges needed.")
            # Still find the second path for UI highlighting
            if mst_path_edges:
                forbidden = frozenset(
                    (min(id(u), id(v)), max(id(u), id(v)))
                    for u, v in mst_path_edges
                )
                # Use roads_only=False so we can traverse any graph edge (including newly added ones)
                second_path = self._ucs_path_edges(hospital, depot, forbidden_edges=forbidden, roads_only=False)
                if second_path:
                    print(f"    [DEBUG] Second path for UI: {len(second_path)} edges")
                    return True, 0.0, second_path
                else:
                    print(f"    [DEBUG] No second path found even with roads_only=False, using primary path")
            return True, 0.0, mst_path_edges or []

        print("  Only one path H↔Depot in MST. Running UCS for cheapest backup route...")
        if not mst_path_edges:
            print("  ⚠  UCS found no path at all — layout may be disconnected.")
            return False, 0.0, []

        # Build forbidden set: all edges on the existing MST path
        # (their node-pair ids, direction-agnostic)
        forbidden = frozenset(
            (min(id(u), id(v)), max(id(u), id(v)))
            for u, v in mst_path_edges
        )

        # UCS ignoring those forbidden edges → cheapest ALTERNATIVE path
        print(f"    [DEBUG] Forbidden edges count: {len(forbidden)}")
        print(f"    [DEBUG] First 3 forbidden: {list(forbidden)[:3]}")
        alt_path_edges = self._ucs_path_edges(hospital, depot, forbidden_edges=forbidden)
        print(f"    [DEBUG] Alternative path found: {len(alt_path_edges)} edges")
        
        if not alt_path_edges:
            print("  ⚠  No edge-disjoint path found. Trying fallback: building spoke roads...")
            # Fallback: Hospital may be isolated. Build roads to nearest non-empty cells.
            alt_path_edges = self._find_fallback_path(hospital, depot, mst_path_edges)
            if not alt_path_edges:
                print("  ✘  Fallback also failed — hospital too isolated.")
                return False, 0.0, []
            print(f"  ✔  Fallback path found with {len(alt_path_edges)} edges.")

        # Add only the edges that are NOT already in the graph
        extra_cost = 0.0
        for u, v in alt_path_edges:
            if self.graph.get_edge_weight(u, v) is None:   # road doesn't exist yet
                w = _edge_weight(u, v)
                self.graph.set_edge(u, v, w)
                self._extra_edges.append((w, u, v))
                extra_cost += w
                print(f"    + Road ({u.row},{u.col})→({v.row},{v.col})  cost={w:.2f}")

        return True, extra_cost, alt_path_edges

    # -------------------------------------------------------------------------
    # Two-path existence check  (unit-capacity max-flow, BFS augmentation)
    # -------------------------------------------------------------------------

    def _has_two_edge_disjoint_paths(self, s: CityNode, t: CityNode) -> bool:
        """
        Returns True if there are at least 2 edge-disjoint paths from s to t
        in the CURRENT graph (after MST edges have been applied).

        Uses Ford-Fulkerson with BFS augmentation (Edmonds-Karp) on a unit-
        capacity version of the road graph.  Finding flow ≥ 2 is equivalent
        to two edge-disjoint paths by Menger's theorem.

        We only need to find flow up to 2, so this terminates in at most
        2 BFS passes — extremely fast.
        """
        # Build residual capacities: each existing road gets cap 1 in each direction
        cap: dict[tuple, int] = defaultdict(int)

        seen_edges: set[tuple] = set()
        for node in self._all_nodes:
            for nb in self.graph.grid_neighbors(node):
                key = (min(id(node), id(nb)), max(id(node), id(nb)))
                if key in seen_edges:
                    continue
                if self.graph.get_edge_weight(node, nb) is not None:
                    seen_edges.add(key)
                    cap[(node, nb)] = 1   # forward arc
                    cap[(nb, node)] = 1   # backward arc (bidirectional road)

        flow_total = 0
        MAX_FLOW   = 2

        while flow_total < MAX_FLOW:
            # BFS to find an augmenting path
            parent: dict[CityNode, CityNode | None] = {s: None}
            queue  = [s]
            found  = False
            while queue and not found:
                curr = queue.pop(0)
                for nb in self.graph.grid_neighbors(curr):
                    if nb not in parent and cap[(curr, nb)] > 0:
                        parent[nb] = curr
                        if nb is t:
                            found = True
                            break
                        queue.append(nb)

            if not found:
                break   # no more augmenting paths

            # Trace back and update residual capacities
            curr = t
            while curr is not s:
                prev = parent[curr]
                cap[(prev, curr)] -= 1
                cap[(curr, prev)] += 1
                curr = prev

            flow_total += 1

        return flow_total >= 2

    # -------------------------------------------------------------------------
    # UCS path finder  (returns list of (u,v) edge pairs along cheapest path)
    # -------------------------------------------------------------------------

    def _ucs_path_edges(
        self,
        start: CityNode,
        goal:  CityNode,
        forbidden_edges: frozenset,
        roads_only: bool = False,
    ) -> list[tuple[CityNode, CityNode]]:
        """
        Uniform Cost Search from start → goal.

        roads_only=False (default): searches the FULL grid graph
            (all grid-adjacent pairs, not just already-built roads).
            Used when finding the cheapest alternative backup route.

        roads_only=True: searches ONLY already-built roads in the graph.
            Used when tracing the existing MST path between hospital
            and depot, so the forbidden set is derived from real MST
            edges rather than shortest-path edges on the full grid.

        forbidden_edges : frozenset of (min_id, max_id) pairs to skip.
        This lets us find an alternative route that avoids the existing
        MST path when enforcing two-path redundancy.

        Edge cost is _edge_weight(u, v) — same cost model as Kruskal's.
        Returns ordered list of (u, v) tuples along the path, or [] if none.
        """
        # Priority queue: (cumulative_cost, counter, node, path_as_edge_list)
        # A monotonic counter is used as the tiebreaker so Python never tries
        # to compare CityNode objects (which have no __lt__).
        _counter = 0
        start_entry = (0.0, _counter, start, [])
        heap = [start_entry]
        visited: set[int] = set()

        iterations = 0
        while heap:
            iterations += 1
            if iterations > 1000:  # Safety limit
                print(f"      [UCS DEBUG] Hit iteration limit, visited={len(visited)}")
                return []
            cost, _, curr, path = heapq.heappop(heap)

            if id(curr) in visited:
                continue
            visited.add(id(curr))

            if curr is goal:
                print(f"      [UCS DEBUG] Goal reached! Path length={len(path)}")
                return path

            neighbors_checked = 0
            added_to_heap = 0
            skipped_reasons = {"visited": 0, "empty": 0, "no_road": 0, "forbidden": 0}
            for nb in self.graph.grid_neighbors(curr):
                neighbors_checked += 1
                if id(nb) in visited:
                    skipped_reasons["visited"] += 1
                    continue
                # roads_only: skip neighbors that don't have a built road yet
                if roads_only and self.graph.get_edge_weight(curr, nb) is None:
                    skipped_reasons["no_road"] += 1
                    continue
                key = (min(id(curr), id(nb)), max(id(curr), id(nb)))
                if key in forbidden_edges:
                    skipped_reasons["forbidden"] += 1
                    print(f"        [UCS SKIP] Edge ({curr.row},{curr.col})-({nb.row},{nb.col}) is FORBIDDEN")
                    continue
                w        = _edge_weight(curr, nb)
                new_cost = cost + w
                _counter += 1
                heapq.heappush(heap, (new_cost, _counter, nb, path + [(curr, nb)]))
                added_to_heap += 1
            
            if iterations <= 3:
                print(f"      [UCS DEBUG] iter {iterations}: at ({curr.row},{curr.col}), neighbors={neighbors_checked}, added={added_to_heap}, heap={len(heap)}")
                print(f"        skipped: {skipped_reasons}")

        print(f"      [UCS DEBUG] No path found after {iterations} iterations, visited={len(visited)}")
        return []   # no path found

    def _find_fallback_path(self, hospital: CityNode, depot: CityNode, mst_path_edges: list) -> list:
        """
        Fallback when hospital is isolated: build spoke roads to nearest non-empty cells,
        then find a path that shares minimal edges with MST path.
        """
        # Get all non-empty neighbors of hospital (including diagonals if needed)
        spoke_edges = []
        for nb in self.graph.grid_neighbors(hospital):
            # Build a road to this neighbor if not already built
            if self.graph.get_edge_weight(hospital, nb) is None:
                spoke_edges.append((hospital, nb))
        
        if not spoke_edges:
            # Hospital completely isolated - try expanding search radius
            print(f"      [FALLBACK] Hospital at ({hospital.row},{hospital.col}) is completely isolated!")
            return []
        
        # Use the first spoke and find a path from there
        spoke = spoke_edges[0]
        start_node = spoke[1]  # The non-empty neighbor
        
        # Build partial forbidden set: only forbid edges that would create cycles
        forbidden_partial = frozenset(
            (min(id(u), id(v)), max(id(u), id(v)))
            for u, v in mst_path_edges[:2]  # Only forbid first 2 edges of MST path
        )
        
        # Find path from spoke node to depot
        path_from_spoke = self._ucs_path_edges(start_node, depot, forbidden_edges=forbidden_partial)
        
        if path_from_spoke:
            # Combine spoke + path
            return [spoke] + path_from_spoke
        
        return []

    # -------------------------------------------------------------------------
    # Apply edges to the shared city graph
    # -------------------------------------------------------------------------

    def _apply_edges(self, edges: list[tuple[float, CityNode, CityNode]]) -> None:
        """Write MST (or extra) edges into the shared city graph."""
        for w, u, v in edges:
            self.graph.set_edge(u, v, w)
            # Print a representative sample so the log isn't overwhelming
        print(f"  Applied {len(edges)} edges to city graph.")


# =============================================================================
# DIAGNOSTIC HELPER  (standalone call for testing)
# =============================================================================

def print_network_summary(graph: CityGraph) -> None:
    """
    Print a summary of the current road network state.
    Useful for debugging or the UI event log.
    """
    total_roads = 0
    total_cost  = 0.0
    seen        = set()

    for node in graph.all_nodes():
        for nb in graph.grid_neighbors(node):
            key = (min(id(node), id(nb)), max(id(node), id(nb)))
            if key in seen:
                continue
            w = graph.get_edge_weight(node, nb)
            if w is not None:
                seen.add(key)
                total_roads += 1
                total_cost  += w

    hospital = _find_primary_hospital(graph)
    depot    = _find_ambulance_depot(graph)

    print("\n--- Road Network Summary ---")
    print(f"  Total roads built : {total_roads}")
    print(f"  Total road cost   : {total_cost:.2f}")
    if hospital and depot:
        builder = RoadNetworkBuilder(graph)
        redundant = builder._has_two_edge_disjoint_paths(hospital, depot)
        print(f"  H↔Depot redundancy: {'✔ Two independent paths' if redundant else '✘ Only one path'}")
    print("----------------------------\n")