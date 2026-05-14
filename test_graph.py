# =============================================================================
# test_graph.py — Graph Layer Test Harness
# =============================================================================
# Run this file directly to verify CityGraph and CityNode work correctly
# BEFORE any challenge module is implemented.
#
#   python test_graph.py
#
# All tests print PASS / FAIL.  No external libraries needed.

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    GRID_ROWS, GRID_COLS,
    LOC_RESIDENTIAL, LOC_HOSPITAL, LOC_AMBULANCE, LOC_INDUSTRIAL,
    RISK_LOW, RISK_MEDIUM, RISK_HIGH,
    WEIGHT_STANDARD, WEIGHT_RESIDENTIAL,
)
from models.city_graph import CityGraph
from models.city_node import CityNode


def check(label: str, condition: bool):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}]  {label}")
    if not condition:
        # Don't raise — run all tests so the full picture is visible
        pass


def section(title: str):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# =============================================================================
def test_grid_construction():
    section("1. Grid construction")
    g = CityGraph(rows=5, cols=5)
    check("graph has 5 rows",  g.rows == 5)
    check("graph has 5 cols",  g.cols == 5)
    check("node(0,0) exists",  isinstance(g.node(0, 0), CityNode))
    check("node(4,4) exists",  isinstance(g.node(4, 4), CityNode))
    check("all_nodes yields 25 nodes", len(list(g.all_nodes())) == 25)
    try:
        g.node(5, 0)
        check("out-of-bounds raises IndexError", False)
    except IndexError:
        check("out-of-bounds raises IndexError", True)


# =============================================================================
def test_node_identity():
    section("2. Node identity and hashing")
    g = CityGraph(rows=3, cols=3)
    n1 = g.node(1, 2)
    n2 = g.node(1, 2)
    check("same position → same object", n1 is n2)
    check("nodes hashable (usable as dict key)", {n1: 99}[n1] == 99)
    check("nodes hashable (usable in set)", n1 in {n1, n2})
    check("different position → different node", g.node(0,0) != g.node(0,1))


# =============================================================================
def test_road_operations():
    section("3. Road add / remove / query")
    g = CityGraph(rows=3, cols=3)
    a = g.node(0, 0)
    b = g.node(0, 1)
    c = g.node(0, 2)

    # Add road via graph (fires observer)
    g.add_road(a, b, WEIGHT_STANDARD)
    check("a has road to b",            a.has_road(b))
    check("b has road to a (bidir)",    b.has_road(a))
    check("base weight a→b is 1.0",     a.get_base_weight(b) == WEIGHT_STANDARD)
    check("base weight b→a is 1.0",     b.get_base_weight(a) == WEIGHT_STANDARD)
    check("a has no road to c",         not a.has_road(c))

    # Block the road
    g.block_road(a, b)
    check("after block: a has no road to b", not a.has_road(b))
    check("after block: b has no road to a", not b.has_road(a))

    # Restore
    g.restore_road(a, b, WEIGHT_STANDARD)
    check("after restore: road exists again", a.has_road(b))

    # Blocking non-existent road raises
    try:
        g.block_road(a, c)
        check("blocking non-existent road raises ValueError", False)
    except ValueError:
        check("blocking non-existent road raises ValueError", True)


# =============================================================================
def test_effective_cost():
    section("4. Effective travel cost (base weight × risk multiplier)")
    g = CityGraph(rows=2, cols=2)
    a = g.node(0, 0)
    b = g.node(0, 1)
    g.add_road(a, b, WEIGHT_STANDARD)   # base = 1.0

    # Default risk is LOW → multiplier 1.0
    check("low risk: effective cost = 1.0",  a.effective_cost_to(b) == 1.0)

    g.set_risk(b, RISK_MEDIUM)
    check("medium risk: effective cost = 1.2",
          abs(a.effective_cost_to(b) - 1.2) < 1e-9)

    g.set_risk(b, RISK_HIGH)
    check("high risk: effective cost = 1.5",
          abs(a.effective_cost_to(b) - 1.5) < 1e-9)

    # Residential road base weight
    g.set_risk(b, RISK_LOW)
    g.add_road(a, b, WEIGHT_RESIDENTIAL)  # overwrites to 0.8
    check("residential road base = 0.8",
          abs(a.effective_cost_to(b) - 0.8) < 1e-9)


# =============================================================================
def test_risk_update():
    section("5. Risk level update and validation")
    g = CityGraph(rows=2, cols=2)
    n = g.node(0, 0)

    g.set_risk(n, RISK_HIGH)
    check("risk_level set to HIGH",        n.risk_level == RISK_HIGH)
    check("risk_multiplier set to 1.5",    n.risk_multiplier == 1.5)

    # apply_risk_map bulk update
    g.apply_risk_map({(0,0): RISK_LOW, (0,1): RISK_MEDIUM})
    check("bulk map: (0,0) is LOW",   g.node(0,0).risk_level == RISK_LOW)
    check("bulk map: (0,1) is MEDIUM", g.node(0,1).risk_level == RISK_MEDIUM)

    # Invalid risk level raises
    try:
        g.set_risk(n, "EXTREME")
        check("invalid risk level raises ValueError", False)
    except ValueError:
        check("invalid risk level raises ValueError", True)


# =============================================================================
def test_observer_pattern():
    section("6. Observer pattern")
    g = CityGraph(rows=2, cols=2)
    a = g.node(0, 0)
    b = g.node(0, 1)
    g.add_road(a, b, WEIGHT_STANDARD)

    events = []
    g.register_observer(lambda etype, payload: events.append(etype))

    g.block_road(a, b)
    check("block_road fires road_blocked event", "road_blocked" in events)

    g.restore_road(a, b, WEIGHT_STANDARD)
    check("restore_road fires road_restored event", "road_restored" in events)

    g.set_risk(a, RISK_HIGH)
    check("set_risk fires risk_updated event", "risk_updated" in events)


# =============================================================================
def test_dijkstra():
    section("7. Dijkstra shortest path")
    #   (0,0) -1.0- (0,1) -1.0- (0,2)
    #     |                       |
    #    1.0                     1.0
    #     |                       |
    #   (1,0) -1.0- (1,1) -1.0- (1,2)
    g = CityGraph(rows=2, cols=3)
    nodes = {(r,c): g.node(r,c) for r in range(2) for c in range(3)}

    g.add_road(nodes[0,0], nodes[0,1], 1.0)
    g.add_road(nodes[0,1], nodes[0,2], 1.0)
    g.add_road(nodes[0,0], nodes[1,0], 1.0)
    g.add_road(nodes[1,0], nodes[1,1], 1.0)
    g.add_road(nodes[1,1], nodes[1,2], 1.0)
    g.add_road(nodes[0,2], nodes[1,2], 1.0)

    dist, prev = g.dijkstra(nodes[0,0], use_risk=False)
    check("dist (0,0)→(0,2) = 2.0", dist[nodes[0,2]] == 2.0)
    check("dist (0,0)→(1,2) = 3.0", dist[nodes[1,2]] == 3.0)

    path = g.reconstruct_path(prev, nodes[0,2])
    check("path (0,0)→(0,2) starts at (0,0)", path[0] == nodes[0,0])
    check("path (0,0)→(0,2) ends at (0,2)",   path[-1] == nodes[0,2])
    check("path (0,0)→(0,2) length = 3",       len(path) == 3)

    # Block middle road → longer path
    g.block_road(nodes[0,0], nodes[0,1])
    dist2, prev2 = g.dijkstra(nodes[0,0], use_risk=False)
    check("after block: dist (0,0)→(0,2) = 4.0", dist2[nodes[0,2]] == 4.0)


# =============================================================================
def test_independent_paths():
    section("8. Edge-disjoint path counting (Hospital↔Depot safety check)")
    #   A -e1- B -e2- C      two edge-disjoint paths A→C: e1-e2 and e3-e4
    #    \               /
    #     ---e3--- D -e4-
    g = CityGraph(rows=1, cols=4)
    A, B, C, D = g.node(0,0), g.node(0,1), g.node(0,2), g.node(0,3)
    g.add_road(A, B, 1.0)
    g.add_road(B, C, 1.0)
    # Only one path so far
    check("one path A→C = 1", g.count_independent_paths(A, C) == 1)

    # Add second path A→D→C
    g.add_road(A, D, 1.0)
    g.add_road(D, C, 1.0)
    check("two paths A→C = 2", g.count_independent_paths(A, C) == 2)

    # Disconnected
    E = g.node(0,3) # reuse; test truly disconnected
    g2 = CityGraph(rows=1, cols=2)
    X, Y = g2.node(0,0), g2.node(0,1)
    check("no roads → 0 paths", g2.count_independent_paths(X, Y) == 0)


# =============================================================================
def test_bfs_hops():
    section("9. BFS hop-distance (CSP proximity check)")
    g = CityGraph(rows=5, cols=5)
    center = g.node(2, 2)
    within_1 = g.bfs_hops(center, max_hops=1)
    within_2 = g.bfs_hops(center, max_hops=2)
    within_3 = g.bfs_hops(center, max_hops=3)

    # Center + 4 neighbors = 5 nodes within 1 hop
    check("within 1 hop of center: 5 nodes", len(within_1) == 5)
    # Within 2 hops of (2,2) on a 5×5 grid = 13 nodes (diamond)
    check("within 2 hops of center: 13 nodes", len(within_2) == 13)
    # Within 3 hops of (2,2): corners like (0,0) are 4 hops away, so 25 - 4 = 21
    check("within 3 hops of center: 21 nodes", len(within_3) == 21)


# =============================================================================
def test_graph_summary():
    section("10. Graph summary and road cost")
    g = CityGraph(rows=2, cols=2)
    a, b, c, d = g.node(0,0), g.node(0,1), g.node(1,0), g.node(1,1)
    g.add_road(a, b, 1.0)
    g.add_road(a, c, 0.8)
    g.add_road(b, d, 1.0)
    g.add_road(c, d, 1.0)

    check("road count = 4", g.road_count() == 4)
    check("total cost = 3.8", abs(g.total_road_cost() - 3.8) < 1e-9)
    check("summary is a string", isinstance(g.summary(), str))


# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  CityMind — Graph Layer Tests")
    print("="*55)

    test_grid_construction()
    test_node_identity()
    test_road_operations()
    test_effective_cost()
    test_risk_update()
    test_observer_pattern()
    test_dijkstra()
    test_independent_paths()
    test_bfs_hops()
    test_graph_summary()

    print("\n" + "="*55)
    print("  All tests complete.")
    print("="*55 + "\n")
