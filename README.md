# CityMind — Project Reference

**Course:** Artificial Intelligence | BS Computer Science  
**Phase:** 2 of 3 — Implementation  
**Deadline:** 10th May, 11:59 PM  
**Team:** Ali Naqvi (24I-0828) · Muntazir Mehdi (24I-0847) · Ammar Iqbal (24-0548)

---

## Project Structure

```
CityMind_Project/
├── main.py              Entry point — initialises graph, runs all 5 challenges, launches UI
├── config.py            All global constants (grid size, location types, risk levels, colours)
├── test_graph.py        Standalone test harness — 44 tests covering the graph layer
│
├── models/
│   ├── city_node.py     CityNode class — one instance per grid cell
│   └── city_graph.py    CityGraph manager — 2-D array + all graph operations
│
├── algorithms/
│   ├── layout_csp.py    Challenge 1 — CSP backtracking layout    [Ali]
│   ├── road_network.py  Challenge 2 — Kruskal MST + UCS          [Ali]
│   ├── ambulance_ga.py  Challenge 3 — Genetic Algorithm          [Muntazir]
│   ├── routing_astar.py Challenge 4 — D* Lite dynamic router     [Ammar]
│   └── crime_ml.py      Challenge 5 — K-Means + Decision Tree    [Muntazir]
│
└── ui/
    └── dashboard.py     Tkinter canvas, overlays, event log       [Ammar]
```

---

## How to Run

```bash
# Run the full system (UI launches at the end)
python main.py

# Verify the graph layer only (no UI, no challenges needed)
python test_graph.py
```

`main.py` prints `[C1] ... not yet implemented` for any challenge stub that hasn't been filled in yet — the rest of the system still runs. You can test your module in isolation before integration.

---

## The One Rule

> **No module keeps its own copy of the graph.**

Every read and write goes through the single `CityGraph` instance created in `main.py` and passed to each challenge function. If you're tempted to copy a list of nodes into your algorithm file, use `graph.all_nodes()` or `graph.nodes_of_type(...)` instead.

---

## Graph Layer — What's Already Implemented

The `models/` layer is complete and fully tested. Here is what each challenge can call:

### Reading the grid

```python
graph.node(row, col)              # → CityNode at that position
graph.all_nodes()                 # → generator of all 400 nodes (20×20)
graph.nodes_of_type(LOC_HOSPITAL) # → list of all hospital nodes
graph.grid_neighbors(node)        # → up-to-4 adjacent cells (ignores roads)
```

### Reading a node

```python
node.location_type        # LOC_RESIDENTIAL, LOC_HOSPITAL, …  (config.py constants)
node.population_density   # float
node.risk_level           # RISK_LOW / RISK_MEDIUM / RISK_HIGH
node.risk_multiplier      # 1.0 / 1.2 / 1.5 (pre-computed, free to read)
node.is_accessible        # False when all roads to this node are blocked
node.ambulance_id         # 0 / 1 / 2 if an ambulance is here, else None
node.get_neighbors()      # dict {CityNode: base_weight} — only unblocked roads
```

### Roads (Challenge 2 writes these)

```python
graph.add_road(a, b, weight)    # build a road (bidirectional)
graph.block_road(a, b)          # flood / block — removes from both adjacency dicts
graph.restore_road(a, b, w)     # re-open a blocked road
node.has_road(other)            # bool
node.get_base_weight(other)     # float — base edge weight
node.effective_cost_to(other)   # base_weight × other.risk_multiplier  ← use this in pathfinding
```

### Risk (Challenge 5 writes these)

```python
graph.set_risk(node, RISK_HIGH)              # single node
graph.apply_risk_map({(r,c): level, …})      # bulk update after ML run
```

### Pathfinding utilities

```python
# BFS on grid adjacency (no roads needed) — used by CSP proximity checks
graph.bfs_hops(start_node, max_hops=3)       # → set of reachable nodes

# Dijkstra on road graph
dist, prev = graph.dijkstra(start, goal=None, use_risk=True)
path = graph.reconstruct_path(prev, goal)    # → ordered list of CityNode

# Edge-disjoint path count — Challenge 2 safety check
graph.count_independent_paths(hospital, depot)  # → int (stops counting at 2)
```

### Observer pattern — how D* Lite hears about floods

```python
def my_callback(event_type: str, payload: dict) -> None:
    if event_type == "road_blocked":
        replan(payload["a"], payload["b"])

graph.register_observer(my_callback)
```

Event types: `road_added`, `road_blocked`, `road_restored`, `risk_updated`, `risk_map_applied`

---

## Config Constants (import from `config.py`)

```python
from config import (
    GRID_ROWS, GRID_COLS,           # 20, 20
    LOC_EMPTY, LOC_RESIDENTIAL, LOC_HOSPITAL,
    LOC_SCHOOL, LOC_INDUSTRIAL, LOC_POWER_PLANT, LOC_AMBULANCE,
    RISK_LOW, RISK_MEDIUM, RISK_HIGH,
    RISK_MULTIPLIER,                # dict: level → float
    WEIGHT_STANDARD,                # 1.0
    WEIGHT_RESIDENTIAL,             # 0.8
    SIM_STEPS,                      # 20
    COLOR_NODE,                     # dict: loc_type → hex colour (for UI)
    COLOR_RISK,                     # dict: risk_level → hex colour (for UI)
)
```

Never hardcode `20` for grid size. Always use `GRID_ROWS` / `GRID_COLS`.

---

## Challenge Contract — What Each Function Must Do

| File | Function / Class | Must call | Must return |
|---|---|---|---|
| `layout_csp.py` | `run_layout(graph)` | `graph.set_location_type()`, `graph.set_population_density()` | `bool` (True = valid, False = fallback) |
| `road_network.py` | `build_roads(graph)` | `graph.add_road()` | `float` (total cost) |
| `ambulance_ga.py` | `place_ambulances(graph, count=3)` | sets `node.ambulance_id` | `list[CityNode]` |
| `routing_astar.py` | `DStarRouter(graph)` | `graph.register_observer()` | — |
| `crime_ml.py` | `run_crime_pipeline(graph)` | `graph.apply_risk_map()` | `dict[(row,col) → risk_level]` |
| `dashboard.py` | `Dashboard(graph)` | `graph.register_observer()` | — |

Stubs with the correct signatures are already in each file. Fill in the body.

---

## Effective Travel Cost — The Formula

All pathfinding (D* Lite in C4, Dijkstra in C3) must use:

```
cost(A → B) = base_weight(A,B) × B.risk_multiplier
```

Call `node.effective_cost_to(neighbor)` — it does this in one line. The risk multiplier is pre-computed on the node so the router pays no extra cost at traversal time.

---

## Current Status

| Component | Status |
|---|---|
| `config.py` | ✅ Complete |
| `models/city_node.py` | ✅ Complete — 44 tests passing |
| `models/city_graph.py` | ✅ Complete — 44 tests passing |
| `algorithms/layout_csp.py` | 🔲 Stub — Ali |
| `algorithms/road_network.py` | 🔲 Stub — Ali |
| `algorithms/ambulance_ga.py` | 🔲 Stub — Muntazir |
| `algorithms/routing_astar.py` | 🔲 Stub — Ammar |
| `algorithms/crime_ml.py` | 🔲 Stub — Muntazir |
| `ui/dashboard.py` | 🔲 Stub — Ammar |
