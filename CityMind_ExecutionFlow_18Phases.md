# CityMind Urban Intelligence System
## Master Execution Flow — 18-Phase Implementation Plan
### Complete Build Guide for Muntazir Mehdi (24I-0847) & Group

---

> **How to Use This Document:**
> This document divides the entire CityMind project into **18 sequential phases**. Each phase is self-contained and buildable in one sitting. Every phase tells you exactly: what to build, how to build it, what the code looks like, how to test it, and what "done" means before moving to the next phase. Work through them in order. Never skip a phase — each phase's output is the foundation the next phase stands on.

---

## Project-Wide Constants & File Structure

Before Phase 1 begins, create your project folder structure:

```
citymind/
├── city/
│   ├── __init__.py
│   ├── node.py              ← CityNode class (Phase 1)
│   ├── graph.py             ← CityGraph class (Phase 2)
│   └── events.py            ← Flood/event system (Phase 14)
├── challenge1/
│   └── layout.py            ← CSP layout planner (Ali's)
├── challenge2/
│   └── roads.py             ← Kruskal MST + UCS (Ali's)
├── challenge3/
│   ├── __init__.py
│   ├── ga_core.py           ← GA engine (YOUR Phase 7-9)
│   ├── fitness.py           ← Dijkstra + minimax (YOUR Phase 6)
│   └── placer.py            ← AmbulancePlacer manager (YOUR Phase 10)
├── challenge4/
│   └── router.py            ← D* Lite router (Ammar's)
├── challenge5/
│   ├── __init__.py
│   ├── features.py          ← Feature extraction (YOUR Phase 11)
│   ├── clustering.py        ← K-Means (YOUR Phase 12)
│   ├── dataset.py           ← Synthetic data gen (YOUR Phase 13)
│   ├── classifier.py        ← Decision Tree (YOUR Phase 14)
│   └── monitor.py           ← Dynamic learning loop (YOUR Phase 15)
├── simulation/
│   ├── __init__.py
│   ├── loop.py              ← 20-step sim loop (Phase 16)
│   └── logger.py            ← Event log (Phase 16)
├── ui/
│   └── interface.py         ← Grid visualizer (Phase 17)
└── main.py                  ← Entry point (Phase 18)
```

---

## PHASE 1 — Build the CityNode Foundation

### What This Phase Is

The `CityNode` is the single most important class in the entire project. It is the atom that every other component — your GA, your ML model, the router, the layout planner — operates on. If this class is wrong, every phase after it is wrong. Build it carefully, test it thoroughly, and get sign-off from your group before anyone writes any other code.

### What You Are Building

A Python class `CityNode` that represents one grid cell in the city. It holds all static properties (type, location, population) and all dynamic properties (risk, accessibility, emergency count, ambulance presence).

### Why It Must Be Built First

Every other challenge reads from and writes to this object. The graph is just a 2D array of `CityNode` objects connected by their `neighbors` dictionaries. Challenge 1 sets `location_type`. Challenge 2 sets `neighbors`. Challenge 3 (your GA) sets `ambulance_here`. Challenge 5 (your ML) sets `risk_index`, `predicted_risk`, `total_emergencies`. Challenge 4 sets `is_accessible`. If any field is missing or mis-typed, the dependent challenge silently breaks.

### Implementation

```python
# FILE: city/node.py

class CityNode:
    """
    Represents one cell in the CityMind grid.
    This is the shared state atom used by ALL five challenges.
    
    OWNERSHIP RULES — who writes what:
      row, col, location_type, population_density  ← Challenge 1 (Ali)
      neighbors                                     ← Challenge 2 (Ali)
      is_accessible                                 ← Challenge 4 (Ammar)
      risk_index, predicted_risk, total_emergencies ← Challenge 5 (YOU)
      ambulance_here                                ← Challenge 3 (YOU)
    """

    # All valid location types — do not use strings directly, use these constants
    TYPE_RESIDENTIAL    = "Residential"
    TYPE_HOSPITAL       = "Hospital"
    TYPE_SCHOOL         = "School"
    TYPE_INDUSTRIAL     = "Industrial"
    TYPE_POWER_PLANT    = "Power Plant"
    TYPE_AMBULANCE_DEPOT = "Ambulance Depot"

    ALL_TYPES = {
        TYPE_RESIDENTIAL, TYPE_HOSPITAL, TYPE_SCHOOL,
        TYPE_INDUSTRIAL, TYPE_POWER_PLANT, TYPE_AMBULANCE_DEPOT
    }

    # Risk multiplier lookup — used by Challenge 5 and Challenge 3
    RISK_MULTIPLIERS = {
        "High":   1.5,
        "Medium": 1.2,
        "Low":    1.0
    }

    def __init__(self, row: int, col: int,
                 location_type: str = "Residential",
                 population_density: float = 0.0):

        # ── SPATIAL IDENTITY (set by Challenge 1, immutable after layout) ──
        self.row = row
        self.col = col
        self.location_type = location_type
        self.population_density = population_density

        # ── GRAPH CONNECTIVITY (set by Challenge 2) ──
        # Format: { neighbor_CityNode : base_edge_weight (float) }
        # Standard road weight = 1.0 | Residential road = 0.8
        self.neighbors: dict = {}

        # ── DYNAMIC STATE — Challenge 4 writes is_accessible ──
        self.is_accessible: bool = True

        # ── CHALLENGE 5 FIELDS (Muntazir writes these) ──
        # risk_index is the cost MULTIPLIER applied when entering this node
        self.risk_index: float = 1.0          # 1.0=Low, 1.2=Medium, 1.5=High
        self.predicted_risk: str = "Low"      # String label matching RISK_MULTIPLIERS keys
        self.total_emergencies: float = 0.0   # Cumulative emergency count (float for neighbor spillover)

        # ── CHALLENGE 3 FIELD (Muntazir writes this) ──
        self.ambulance_here: bool = False     # True when GA solution places ambulance here

    # ─────────────────────────────────────────────────
    # GRAPH OPERATIONS
    # ─────────────────────────────────────────────────

    def add_neighbor(self, other: 'CityNode', base_weight: float = 1.0):
        """
        Creates a bidirectional road between this node and other.
        Called by Challenge 2 (road builder) during MST construction.
        """
        self.neighbors[other] = base_weight
        other.neighbors[self] = base_weight

    def get_effective_weight_to(self, neighbor: 'CityNode') -> float:
        """
        Returns the EFFECTIVE (risk-adjusted) travel cost from this node
        to the given neighbor. This is the formula used by BOTH:
          - Challenge 3 Dijkstra fitness function
          - Challenge 4 D* Lite router

        Formula: base_weight × destination.risk_index
        
        The risk multiplier is applied at the DESTINATION because the
        difficulty is in entering a high-risk zone, not leaving it.
        """
        if neighbor not in self.neighbors:
            return float('inf')
        return self.neighbors[neighbor] * neighbor.risk_index

    def block_road_to(self, other: 'CityNode'):
        """
        Simulates a road flood/blockage. Removes bidirectional connection.
        Called by Challenge 4 event system.
        O(1) — change is immediately visible to ALL modules reading neighbors.
        """
        self.neighbors.pop(other, None)
        other.neighbors.pop(self, None)

    def has_road_to(self, other: 'CityNode') -> bool:
        return other in self.neighbors

    # ─────────────────────────────────────────────────
    # UTILITY
    # ─────────────────────────────────────────────────

    def manhattan_distance(self, other: 'CityNode') -> int:
        """Used as admissible heuristic in A* / D* Lite."""
        return abs(self.row - other.row) + abs(self.col - other.col)

    def is_citizen_node(self) -> bool:
        """
        Returns True if this node has population (a 'demand point' for GA fitness).
        Used by Challenge 3 to build the citizen_nodes list.
        """
        return self.population_density > 0 and self.is_accessible

    def is_eligible_for_ambulance(self) -> bool:
        """
        Returns True if an ambulance can be placed here.
        Per Phase 1 design: only Depot or Hospital nodes.
        """
        return (self.is_accessible and
                self.location_type in {self.TYPE_AMBULANCE_DEPOT, self.TYPE_HOSPITAL})

    def __repr__(self):
        return (f"Node({self.row},{self.col}|{self.location_type[:3].upper()}|"
                f"pop={self.population_density:.1f}|risk={self.risk_index}|"
                f"emerg={self.total_emergencies:.1f})")

    def __hash__(self):
        # Nodes are uniquely identified by their grid position
        return hash((self.row, self.col))

    def __eq__(self, other):
        return isinstance(other, CityNode) and self.row == other.row and self.col == other.col
```

### How to Test Phase 1 is Done

```python
# PHASE 1 TEST — run this in a scratch file, all assertions must pass

from city.node import CityNode

# Test 1: Basic creation
n1 = CityNode(0, 0, "Residential", 100.0)
n2 = CityNode(0, 1, "Hospital", 50.0)
assert n1.risk_index == 1.0
assert n1.ambulance_here == False
assert n1.is_accessible == True
assert n1.total_emergencies == 0.0

# Test 2: Road connection
n1.add_neighbor(n2, 1.0)
assert n2 in n1.neighbors
assert n1 in n2.neighbors            # Bidirectional

# Test 3: Effective weight with risk
n2.risk_index = 1.5                  # High risk
assert n1.get_effective_weight_to(n2) == 1.5   # 1.0 * 1.5

# Test 4: Block road
n1.block_road_to(n2)
assert n2 not in n1.neighbors
assert n1 not in n2.neighbors
assert n1.get_effective_weight_to(n2) == float('inf')

# Test 5: Citizen check
n3 = CityNode(1, 0, "Industrial", 0.0)
assert n1.is_citizen_node() == True    # Has population
assert n3.is_citizen_node() == False   # No population

print("PHASE 1: ALL TESTS PASSED ✓")
```

### Phase 1 Done Criteria

- [ ] `CityNode` class created in `city/node.py`
- [ ] All 9 fields present with correct types and default values
- [ ] `add_neighbor()` creates bidirectional connection
- [ ] `get_effective_weight_to()` multiplies base weight by destination `risk_index`
- [ ] `block_road_to()` removes bidirectional connection
- [ ] `is_citizen_node()` and `is_eligible_for_ambulance()` return correct booleans
- [ ] `__hash__` and `__eq__` use `(row, col)` so nodes work in sets and dict keys
- [ ] All 5 test assertions pass

---

## PHASE 2 — Build the CityGraph Container

### What This Phase Is

The `CityGraph` is the wrapper around your 2D array of `CityNode` objects. It is the object passed to every challenge. It owns the grid, exposes traversal helpers, and is the "single source of truth" the project statement mandates.

### What You Are Building

A `CityGraph` class that holds the 2D grid, provides helper iteration methods, and maintains a version counter that your Challenge 3 fitness cache checks to know when to invalidate.

### Implementation

```python
# FILE: city/graph.py

from city.node import CityNode
from typing import List, Iterator, Optional

class CityGraph:
    """
    The shared city graph — single source of truth for the entire system.
    
    Wraps a 2D list of CityNode objects.
    Every module receives a reference to the SAME CityGraph instance.
    No module may copy this object.
    """

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols

        # The primary data structure: 2D array of CityNode objects
        # Access: self.grid[row][col]
        self.grid: List[List[CityNode]] = [
            [CityNode(r, c) for c in range(cols)]
            for r in range(rows)
        ]

        # Version counter — incremented by Challenge 5 whenever risk_index changes.
        # Challenge 3 fitness cache checks this to decide whether to recompute.
        self.version: int = 0

        # Primary hospital reference — set by Challenge 2 after layout
        self.primary_hospital: Optional[CityNode] = None
        # Primary ambulance depot reference — set by Challenge 2
        self.primary_depot: Optional[CityNode] = None

    # ─────────────────────────────────────────────────
    # NODE ACCESS
    # ─────────────────────────────────────────────────

    def get_node(self, row: int, col: int) -> CityNode:
        """Safe access with bounds checking."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row][col]
        raise IndexError(f"Node ({row},{col}) out of bounds for {self.rows}×{self.cols} grid.")

    def all_nodes(self) -> Iterator[CityNode]:
        """Iterates over every node in the grid, row by row."""
        for row in self.grid:
            for node in row:
                yield node

    def nodes_of_type(self, location_type: str) -> List[CityNode]:
        """Returns all nodes of the given location type."""
        return [n for n in self.all_nodes() if n.location_type == location_type]

    def accessible_nodes(self) -> List[CityNode]:
        """Returns all nodes where is_accessible == True."""
        return [n for n in self.all_nodes() if n.is_accessible]

    def citizen_nodes(self) -> List[CityNode]:
        """
        Returns all nodes that have population (demand points for GA fitness).
        Filters out inaccessible nodes automatically.
        """
        return [n for n in self.all_nodes() if n.is_citizen_node()]

    def eligible_ambulance_nodes(self) -> List[CityNode]:
        """
        Returns all nodes where an ambulance can legally be placed.
        Per Phase 1 design: Depot or Hospital nodes that are accessible.
        """
        return [n for n in self.all_nodes() if n.is_eligible_for_ambulance()]

    def industrial_nodes(self) -> List[CityNode]:
        """Returns all Industrial zone nodes — used by Challenge 5 proximity computation."""
        return self.nodes_of_type(CityNode.TYPE_INDUSTRIAL)

    # ─────────────────────────────────────────────────
    # SPATIAL HELPERS
    # ─────────────────────────────────────────────────

    def grid_neighbors_of(self, node: CityNode) -> List[CityNode]:
        """
        Returns the up to 4 grid-adjacent nodes (N, S, E, W).
        Note: this is GRID adjacency (physical proximity), NOT road adjacency.
        Road adjacency is node.neighbors — set by Challenge 2.
        Used by Challenge 1 (layout constraint checking) and Challenge 2 (road building).
        """
        candidates = [
            (node.row - 1, node.col),
            (node.row + 1, node.col),
            (node.row, node.col - 1),
            (node.row, node.col + 1),
        ]
        result = []
        for r, c in candidates:
            if 0 <= r < self.rows and 0 <= c < self.cols:
                result.append(self.grid[r][c])
        return result

    # ─────────────────────────────────────────────────
    # GRAPH VERSION (used by Challenge 3 cache)
    # ─────────────────────────────────────────────────

    def increment_version(self):
        """
        Called by Challenge 5 after updating risk_index on any node.
        Signals to Challenge 3's fitness cache that all cached values are stale.
        """
        self.version += 1

    # ─────────────────────────────────────────────────
    # STATISTICS (for UI and event log)
    # ─────────────────────────────────────────────────

    def total_road_cost(self) -> float:
        """
        Computes total base edge weight across all roads.
        Each undirected edge counted once.
        Used by UI statistics panel.
        """
        seen_pairs = set()
        total = 0.0
        for node in self.all_nodes():
            for neighbor, weight in node.neighbors.items():
                pair = tuple(sorted([(node.row, node.col), (neighbor.row, neighbor.col)]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    total += weight
        return total

    def blocked_road_count(self) -> int:
        """
        Counts how many node pairs that were once connected are now disconnected
        due to flood events. Used by UI statistics panel.
        Requires a snapshot of initial connections — see simulation/loop.py.
        """
        # Placeholder — actual implementation needs initial edge set stored at startup
        return 0

    def average_risk(self) -> float:
        """
        Returns mean risk_index across all nodes.
        Used by AmbulancePlacer to detect when risk profile has shifted enough
        to re-run the GA.
        """
        nodes = list(self.all_nodes())
        return sum(n.risk_index for n in nodes) / len(nodes)

    def __repr__(self):
        return f"CityGraph({self.rows}×{self.cols}, v={self.version})"
```

### Phase 2 Done Criteria

- [ ] `CityGraph(rows, cols)` creates a 2D grid of `CityNode` objects
- [ ] `all_nodes()`, `citizen_nodes()`, `eligible_ambulance_nodes()` return correct subsets
- [ ] `industrial_nodes()` returns all `TYPE_INDUSTRIAL` nodes
- [ ] `increment_version()` increments `self.version`
- [ ] `average_risk()` returns the mean of all `node.risk_index` values
- [ ] `grid_neighbors_of()` returns only in-bounds adjacent nodes (not road-connected neighbors)

---

## PHASE 3 — Coordinate with Teammates: Graph Contract Review

### What This Phase Is

This is NOT a coding phase. It is a **team synchronization meeting**. Before Ali writes Challenge 1 code and Ammar writes Challenge 4 code, everyone must agree on the exact interface they will use with the graph. You (Muntazir) call this meeting because your Challenge 3 and Challenge 5 depend on fields that Ali and Ammar set.

### What You Discuss and Agree On

Sit together (or on a call) and agree on each item below. Write the decisions down.

**Agreement 1: Grid Size**
- Decide: Will the grid be 15×15, 20×20, or larger?
- Recommendation: **20×20** (400 nodes, manageable performance, large enough to be meaningful).
- Write in your shared notes: `GRID_SIZE = 20`

**Agreement 2: Location Type Counts**
- How many of each type will Challenge 1 place?
- Recommendation: Residential=150, Hospital=3, School=5, Industrial=4, Power Plant=2, Ambulance Depot=3.
- The 3 Ambulance Depots are the eligible placement positions for your GA.

**Agreement 3: Population Density Range**
- Residential nodes: population_density between 50 and 500 (randomly assigned by Challenge 1).
- Hospital: 30–100 (staff + patients).
- School: 200–800 (student body).
- Industrial: 10–50 (workers).
- Power Plant: 5–20.
- Ambulance Depot: 0 (no resident population — it's a facility).

**Agreement 4: Base Edge Weights**
- Challenge 2 (Ali) sets road weights. Standard road = 1.0. Residential road = 0.8.
- You confirm: Challenge 3 and Challenge 5 NEVER modify `neighbors` dict values directly.
- Only `block_road_to()` on CityNode removes neighbors. Only Challenge 2 adds them.

**Agreement 5: The `is_accessible` Flag**
- Challenge 4 (Ammar) owns this. When a flood event cuts off a node completely, Ammar sets `node.is_accessible = False`.
- You confirm: Your GA re-builds its `eligible_nodes` list and `citizen_nodes` list fresh from the live graph whenever it runs. No cached lists that miss newly inaccessible nodes.

**Agreement 6: The `total_emergencies` field**
- You own this. Ammar's router calls `crime_stats_manager.notify_emergency(node)` (your function) whenever the medical team dispatches to a civilian. You handle the increment internally.
- Ammar does NOT increment `total_emergencies` directly.

**Deliverable of Phase 3:** A written one-page agreement (can be a WhatsApp message thread or document) with all 6 agreements confirmed by all 3 members.

### Phase 3 Done Criteria

- [ ] Grid size agreed and written down
- [ ] Node count per type agreed and written down
- [ ] Population density ranges per type agreed
- [ ] Ali knows the base edge weight rules
- [ ] Ammar knows to call your `notify_emergency()` from his router
- [ ] Everyone has the latest `city/node.py` and `city/graph.py` files

---

## PHASE 4 — Build the Dijkstra Engine (Core of Challenge 3 Fitness)

### What This Phase Is

Before writing any GA code, you need the core algorithm that the GA depends on: Multi-Source Dijkstra. This algorithm answers the question "given a set of ambulance positions, what is the shortest risk-adjusted distance from every node in the city to its nearest ambulance?" The GA's fitness function calls this algorithm for every chromosome it evaluates.

### Why Multi-Source Dijkstra (Not 3 × Single-Source)

With 3 ambulances, the naive approach runs Dijkstra 3 times (once per ambulance) and then takes the minimum across the 3 results. Multi-Source Dijkstra initializes the priority queue with all 3 ambulance nodes simultaneously at cost 0. The result is mathematically identical but computed in a single pass — approximately 3× faster. For a population of 100 chromosomes evaluated over 200 generations, this saves 200 × 100 × 2 = 40,000 Dijkstra traversals.

### Implementation

```python
# FILE: challenge3/fitness.py

import heapq
from typing import List, Dict
from city.node import CityNode
from city.graph import CityGraph


def multi_source_dijkstra(graph: CityGraph, sources: List[CityNode]) -> Dict[CityNode, float]:
    """
    Computes shortest paths from multiple source nodes simultaneously.
    
    Returns dist[node] = minimum risk-adjusted distance from any source to node.
    
    The effective edge weight used here is:
        base_weight × destination.risk_index
    
    This means high-risk zones are 'heavier' to travel through.
    When Challenge 5 updates risk_index on any node, the next call to this
    function automatically reflects the updated weights — no re-initialization needed.
    
    Args:
        graph:   The shared CityGraph (used only to know which nodes exist)
        sources: List of CityNode objects representing ambulance positions
    
    Returns:
        Dictionary mapping every reachable CityNode to its shortest distance
        from the nearest source. Unreachable nodes are absent from the dict.
    """
    dist: Dict[CityNode, float] = {}

    # Initialize priority queue with all sources at cost 0
    # Heap entry format: (cost, node_id_for_tiebreak, node_object)
    # We use id(node) as tiebreaker because CityNode objects are not orderable
    pq = []
    for source in sources:
        heapq.heappush(pq, (0.0, id(source), source))

    while pq:
        current_cost, _, current_node = heapq.heappop(pq)

        # Already settled this node with a shorter or equal path
        if current_node in dist:
            continue

        # Settle this node
        dist[current_node] = current_cost

        # Explore neighbors
        for neighbor, base_weight in current_node.neighbors.items():
            # Skip nodes that are completely inaccessible (flooded off the map)
            if not neighbor.is_accessible:
                continue

            # CRITICAL: apply risk multiplier from Challenge 5
            # effective_cost = how hard it is to ENTER the neighbor node
            effective_cost = base_weight * neighbor.risk_index
            new_dist = current_cost + effective_cost

            if neighbor not in dist:
                heapq.heappush(pq, (new_dist, id(neighbor), neighbor))

    return dist


def calculate_minimax_fitness(chromosome: List[CityNode], graph: CityGraph) -> float:
    """
    Computes the Minimax fitness of a chromosome.
    
    Minimax means: what is the WORST-CASE response distance?
    i.e., for the citizen who is FARTHEST from ANY ambulance, how far are they?
    
    We want to MINIMIZE this value across GA generations.
    
    Step-by-step:
    1. Run Multi-Source Dijkstra from all 3 ambulance positions simultaneously
    2. For each citizen node, their "coverage distance" = dist[citizen]
       (this is already the distance to the NEAREST ambulance, from multi-source)
    3. Find the MAXIMUM coverage distance across all citizens
    4. That maximum IS the fitness — lower is better
    
    Returns:
        float: The worst-case distance. float('inf') if any citizen is unreachable.
    """
    # Build the distance map from all ambulance positions
    dist_from_nearest = multi_source_dijkstra(graph, chromosome)

    # Get all citizen nodes (demand points)
    citizens = graph.citizen_nodes()

    if not citizens:
        return 0.0  # Edge case: no citizens → trivially optimal

    worst_case = 0.0
    for citizen in citizens:
        d = dist_from_nearest.get(citizen, float('inf'))
        if d > worst_case:
            worst_case = d

    return worst_case


# ── FITNESS CACHING ──────────────────────────────────────────────────────────
# Running Dijkstra for every chromosome every generation is expensive.
# If the graph hasn't changed (same version), a chromosome's fitness is stable.

_fitness_cache: Dict[tuple, float] = {}
_cache_graph_version: int = -1


def get_cached_fitness(chromosome: List[CityNode], graph: CityGraph) -> float:
    """
    Returns cached fitness if the graph version hasn't changed since last compute.
    Otherwise recomputes and caches.
    
    Cache key = (sorted tuple of node (row,col) pairs, graph.version)
    Sorting makes the key order-independent — [A,B,C] same as [C,A,B].
    """
    global _fitness_cache, _cache_graph_version

    # Invalidate entire cache if graph version changed (C5 updated risk weights)
    if graph.version != _cache_graph_version:
        _fitness_cache.clear()
        _cache_graph_version = graph.version

    # Build a hashable, order-independent key from chromosome
    chrom_key = tuple(sorted((n.row, n.col) for n in chromosome))

    if chrom_key not in _fitness_cache:
        _fitness_cache[chrom_key] = calculate_minimax_fitness(chromosome, graph)

    return _fitness_cache[chrom_key]
```

### How to Test Phase 4

```python
# PHASE 4 TEST
from city.node import CityNode
from city.graph import CityGraph
from challenge3.fitness import multi_source_dijkstra, calculate_minimax_fitness

# Build a tiny 3×3 graph manually
graph = CityGraph(3, 3)

# Connect nodes in a line: (0,0) — (0,1) — (0,2)
graph.grid[0][0].add_neighbor(graph.grid[0][1], 1.0)
graph.grid[0][1].add_neighbor(graph.grid[0][2], 1.0)

# Set populations
graph.grid[0][0].population_density = 100.0
graph.grid[0][0].location_type = "Residential"
graph.grid[0][2].population_density = 100.0
graph.grid[0][2].location_type = "Residential"

# Place ambulance at center
ambulance = graph.grid[0][1]
ambulance.location_type = "Ambulance Depot"

dist = multi_source_dijkstra(graph, [ambulance])
assert dist[graph.grid[0][0]] == 1.0   # One hop left
assert dist[graph.grid[0][2]] == 1.0   # One hop right

# Fitness = max(1.0, 1.0) = 1.0
fitness = calculate_minimax_fitness([ambulance], graph)
assert fitness == 1.0

# Now add risk to node (0,0)
graph.grid[0][0].risk_index = 1.5  # High risk
dist2 = multi_source_dijkstra(graph, [ambulance])
# Cost entering (0,0) from (0,1) = 1.0 * 1.5 = 1.5
assert dist2[graph.grid[0][0]] == 1.5

print("PHASE 4: ALL TESTS PASSED ✓")
```

### Phase 4 Done Criteria

- [ ] `multi_source_dijkstra()` initializes all sources at cost 0 in same priority queue
- [ ] `effective_cost = base_weight × neighbor.risk_index` inside the loop
- [ ] Inaccessible nodes (`is_accessible == False`) are skipped
- [ ] `calculate_minimax_fitness()` returns `max` of all citizen distances
- [ ] Returns `float('inf')` correctly when a citizen is unreachable
- [ ] Cache invalidates when `graph.version` changes

---

## PHASE 5 — Build GA Utility Functions

### What This Phase Is

Before the main GA loop, you need the four building-block functions that the loop calls: population initialization, tournament selection, crossover, and mutation. Build and test each one independently before assembling the loop.

### Implementation

```python
# FILE: challenge3/ga_core.py  (first section — utility functions)

import random
from typing import List, Tuple
from city.node import CityNode
from city.graph import CityGraph


# ── POPULATION INITIALIZATION ─────────────────────────────────────────────────

def initialize_population(graph: CityGraph, population_size: int = 100) -> List[List[CityNode]]:
    """
    Creates the initial random population.
    
    Each chromosome = list of 3 distinct CityNode objects from eligible positions.
    Uses random.sample() which guarantees no duplicates within a chromosome.
    
    Raises ValueError if fewer than 3 eligible positions exist.
    """
    eligible = graph.eligible_ambulance_nodes()

    if len(eligible) < 3:
        raise ValueError(
            f"Cannot create population: only {len(eligible)} eligible ambulance "
            f"positions on the grid. Need at least 3."
        )

    population = []
    for _ in range(population_size):
        chromosome = random.sample(eligible, 3)
        population.append(chromosome)

    return population


# ── SELECTION ─────────────────────────────────────────────────────────────────

def tournament_selection(population: List[List[CityNode]],
                          fitness_scores: List[float],
                          tournament_size: int = 5) -> List[CityNode]:
    """
    Tournament selection: pick k random chromosomes, return the fittest one.
    
    Why not Roulette Wheel:
    - Roulette requires fitness-proportionate probability, which breaks down
      when fitness values vary widely (e.g., after a risk spike from Challenge 5).
    - Tournament is robust to scale changes — only the ranking matters.
    - Tournament maintains diversity better for large populations.
    
    Lower fitness = better (we minimize worst-case distance).
    """
    # Randomly pick tournament_size competitors
    indices = random.sample(range(len(population)), min(tournament_size, len(population)))

    # Winner = index with LOWEST fitness value
    winner_idx = min(indices, key=lambda i: fitness_scores[i])
    return list(population[winner_idx])  # Return a copy


# ── CROSSOVER ─────────────────────────────────────────────────────────────────

def crossover(parent1: List[CityNode],
               parent2: List[CityNode],
               eligible_nodes: List[CityNode]) -> Tuple[List[CityNode], List[CityNode]]:
    """
    Single-point crossover at index 1.
    
    child1 = [parent1[0], parent2[1], parent2[2]]
    child2 = [parent2[0], parent1[1], parent1[2]]
    
    After splitting, duplicates may arise (parent1[0] == parent2[1] etc.)
    repair_chromosome() fixes these by replacing duplicates with random eligible nodes.
    """
    child1_raw = [parent1[0]] + parent2[1:]
    child2_raw = [parent2[0]] + parent1[1:]

    child1 = repair_chromosome(child1_raw, eligible_nodes)
    child2 = repair_chromosome(child2_raw, eligible_nodes)

    return child1, child2


def repair_chromosome(chromosome: List[CityNode],
                       eligible_nodes: List[CityNode]) -> List[CityNode]:
    """
    Fixes a chromosome that has duplicate nodes by replacing duplicates
    with randomly chosen eligible nodes not already in the chromosome.
    
    This keeps all chromosomes valid after every crossover without needing
    a separate validity check.
    """
    seen_ids = set()
    result = []

    # Keep unique nodes first
    for node in chromosome:
        if id(node) not in seen_ids:
            seen_ids.add(id(node))
            result.append(node)

    # Fill remaining slots from eligible nodes not already used
    unused = [n for n in eligible_nodes if id(n) not in seen_ids]
    random.shuffle(unused)

    while len(result) < 3:
        if not unused:
            raise RuntimeError(
                "repair_chromosome: not enough eligible nodes to fill chromosome. "
                "Increase grid size or eligible positions."
            )
        node = unused.pop()
        result.append(node)
        seen_ids.add(id(node))

    return result


# ── MUTATION ──────────────────────────────────────────────────────────────────

def mutate(chromosome: List[CityNode],
           eligible_nodes: List[CityNode],
           mutation_rate: float = 0.15) -> List[CityNode]:
    """
    Random node swap mutation.
    
    With probability mutation_rate:
      - Choose one of the 3 ambulance positions at random
      - Replace it with a random eligible node NOT already in the chromosome
    
    This is the simplest effective mutation for placement problems.
    The replacement is always a valid node, so no repair needed after mutation.
    """
    if random.random() > mutation_rate:
        return chromosome  # No mutation this time

    idx_to_replace = random.randint(0, 2)
    current_ids = {id(n) for n in chromosome}

    # Candidates: eligible nodes not already in the chromosome
    candidates = [n for n in eligible_nodes if id(n) not in current_ids]

    if not candidates:
        return chromosome  # All eligible nodes already used — cannot mutate

    chromosome[idx_to_replace] = random.choice(candidates)
    return chromosome


def adaptive_mutation_rate(base_rate: float,
                            gens_without_improvement: int,
                            stagnation_threshold: int = 20) -> float:
    """
    Doubles the mutation rate if stuck in a local optimum.
    Reverts to base_rate once the stagnation clears.
    
    Prevents premature convergence on large grids where the initial
    random population may settle quickly into a mediocre plateau.
    """
    if gens_without_improvement > stagnation_threshold:
        return min(base_rate * 2.0, 0.40)  # Cap at 40% to avoid random walk
    return base_rate
```

### Phase 5 Done Criteria

- [ ] `initialize_population()` uses `random.sample` (no duplicates)
- [ ] `tournament_selection()` returns a COPY of the winner (not the original reference)
- [ ] `crossover()` correctly splits at index 1 and calls `repair_chromosome()`
- [ ] `repair_chromosome()` fills duplicates from unused eligible nodes
- [ ] `mutate()` only replaces with nodes NOT already in the chromosome
- [ ] `adaptive_mutation_rate()` caps at 40% and only activates after stagnation threshold

---

## PHASE 6 — Assemble the Main GA Loop

### What This Phase Is

With the building blocks ready (fitness from Phase 4, utility functions from Phase 5), assemble the main generational loop. This is the heart of Challenge 3.

### Implementation

```python
# FILE: challenge3/ga_core.py  (second section — main loop)
# Add this below the utility functions from Phase 5

from challenge3.fitness import get_cached_fitness

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

GA_CONFIG = {
    "population_size":    100,
    "max_generations":    200,
    "tournament_size":    5,
    "crossover_rate":     0.85,
    "mutation_rate":      0.15,
    "elite_count":        2,
    "stagnation_limit":   30,
    "num_ambulances":     3,
}


# ── MAIN GA LOOP ──────────────────────────────────────────────────────────────

def run_ga(graph: CityGraph, config: dict = GA_CONFIG) -> Tuple[List[CityNode], float]:
    """
    Runs the full Genetic Algorithm for ambulance placement.
    
    Returns:
        best_chromosome: list of 3 CityNode objects — the optimal placement found
        best_fitness:    float — the worst-case citizen-to-ambulance distance
    
    Side effects:
        Sets node.ambulance_here = True on the 3 nodes in best_chromosome.
        Clears node.ambulance_here = False on any previously placed ambulances.
    """
    print("\n" + "="*55)
    print("CHALLENGE 3: Genetic Algorithm — Ambulance Placement")
    print("="*55)

    # ── SETUP ────────────────────────────────────────────────────────────────
    eligible_nodes = graph.eligible_ambulance_nodes()
    print(f"  Eligible positions: {len(eligible_nodes)}")
    print(f"  Citizen nodes (demand): {len(graph.citizen_nodes())}")
    print(f"  Population: {config['population_size']} | Max generations: {config['max_generations']}")

    # Clear any existing ambulance placements from a previous GA run
    for node in graph.all_nodes():
        node.ambulance_here = False

    # ── INITIALIZATION ───────────────────────────────────────────────────────
    population = initialize_population(graph, config["population_size"])

    best_chromosome: List[CityNode] = None
    best_fitness: float = float('inf')
    gens_no_improve: int = 0

    # ── GENERATIONAL LOOP ────────────────────────────────────────────────────
    for generation in range(1, config["max_generations"] + 1):

        # STEP 1: Evaluate entire population
        fitness_scores = [
            get_cached_fitness(chrom, graph)
            for chrom in population
        ]

        # STEP 2: Track best individual this generation
        gen_best_idx = min(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
        gen_best_fitness = fitness_scores[gen_best_idx]

        if gen_best_fitness < best_fitness:
            best_fitness = gen_best_fitness
            best_chromosome = list(population[gen_best_idx])
            gens_no_improve = 0
        else:
            gens_no_improve += 1

        # STEP 3: Log progress every 20 generations
        if generation % 20 == 0 or generation == 1:
            pos_str = str([(n.row, n.col) for n in best_chromosome])
            print(f"  Gen {generation:3d} | Best: {best_fitness:.4f} | "
                  f"No-improve: {gens_no_improve} | Positions: {pos_str}")

        # STEP 4: Early stopping if converged
        if gens_no_improve >= config["stagnation_limit"]:
            print(f"  [GA] Converged early at generation {generation}.")
            break

        # STEP 5: Build next generation
        # Sort population by fitness (ascending = best first)
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
        next_generation = []

        # Elitism: carry the best ELITE_COUNT chromosomes unchanged
        for i in range(config["elite_count"]):
            next_generation.append(list(population[sorted_indices[i]]))

        # Adaptive mutation rate (increases if stuck)
        current_mutation_rate = adaptive_mutation_rate(
            config["mutation_rate"], gens_no_improve
        )

        # Selection → Crossover → Mutation to fill rest of population
        while len(next_generation) < config["population_size"]:

            # Select two parents via tournament
            p1 = tournament_selection(population, fitness_scores, config["tournament_size"])
            p2 = tournament_selection(population, fitness_scores, config["tournament_size"])

            # Crossover with probability CROSSOVER_RATE
            if random.random() < config["crossover_rate"]:
                c1, c2 = crossover(p1, p2, eligible_nodes)
            else:
                c1, c2 = list(p1), list(p2)  # Pass through unchanged

            # Mutate
            c1 = mutate(c1, eligible_nodes, current_mutation_rate)
            c2 = mutate(c2, eligible_nodes, current_mutation_rate)

            next_generation.append(c1)
            if len(next_generation) < config["population_size"]:
                next_generation.append(c2)

        population = next_generation

    # ── APPLY RESULT TO GRAPH ─────────────────────────────────────────────────
    for node in best_chromosome:
        node.ambulance_here = True

    print(f"\n  ✓ GA Complete.")
    print(f"  Worst-case response distance: {best_fitness:.4f}")
    print(f"  Ambulance positions: {[(n.row, n.col) for n in best_chromosome]}")
    print("="*55 + "\n")

    return best_chromosome, best_fitness
```

### Phase 6 Done Criteria

- [ ] GA loop runs for `max_generations` or until stagnation limit, whichever comes first
- [ ] Elitism copies top 2 chromosomes BEFORE building new generation
- [ ] All ambulance_here flags cleared before new placement applied
- [ ] Progress logged every 20 generations with position information
- [ ] Returns both the chromosome (list of nodes) and the fitness value

---

## PHASE 7 — Build the AmbulancePlacer Manager (Challenge 3 Re-trigger System)

### What This Phase Is

The GA runs once at startup. But the project requires it to re-run when conditions change: roads flood, risk weights shift. The `AmbulancePlacer` class is the manager that monitors these events and decides when a re-run is warranted.

### Implementation

```python
# FILE: challenge3/placer.py

from challenge3.ga_core import run_ga, GA_CONFIG
from city.graph import CityGraph
from typing import List
from city.node import CityNode


class AmbulancePlacer:
    """
    Manages GA execution lifecycle for Challenge 3.
    
    Other modules notify this manager of events. The manager decides
    whether the event warrants a full GA re-run.
    
    Public interface (called by other challenges):
        initial_placement()          ← call once at simulation start
        notify_road_flooded()        ← Challenge 4 calls after each flood event
        notify_risk_updated()        ← Challenge 5 calls after updating risk_index
        get_current_placement()      ← UI and logger call to display positions
    """

    FLOOD_RETRIGGER_THRESHOLD = 3    # Re-run after this many accumulated flood events
    RISK_DELTA_THRESHOLD = 0.15      # Re-run if avg risk shifts by more than this

    def __init__(self, graph: CityGraph, config: dict = GA_CONFIG):
        self.graph = graph
        self.config = config

        self.current_placement: List[CityNode] = []
        self.current_fitness: float = float('inf')

        self.flood_count_since_last_run: int = 0
        self.last_avg_risk: float = graph.average_risk()

    def initial_placement(self) -> tuple:
        """
        Called once at simulation startup, AFTER Challenge 5 has set all risk_index values.
        Runs the full GA and stores the best placement.
        """
        print("[AmbulancePlacer] Running initial placement GA...")
        self.current_placement, self.current_fitness = run_ga(self.graph, self.config)
        self.last_avg_risk = self.graph.average_risk()
        self.flood_count_since_last_run = 0
        return self.current_placement, self.current_fitness

    def notify_road_flooded(self, node_u: CityNode, node_v: CityNode):
        """
        Called by Challenge 4 router whenever a road is blocked by a flood event.
        Accumulates flood count. Triggers GA re-run when threshold is reached.
        """
        self.flood_count_since_last_run += 1

        if self.flood_count_since_last_run >= self.FLOOD_RETRIGGER_THRESHOLD:
            print(f"[AmbulancePlacer] {self.flood_count_since_last_run} floods accumulated "
                  f"→ triggering GA re-run.")
            self._rerun_ga(reason="road_flood_threshold")

    def notify_risk_updated(self):
        """
        Called by Challenge 5 after updating risk_index on graph nodes.
        Re-runs GA if the average risk profile has shifted significantly.
        """
        current_avg = self.graph.average_risk()
        delta = abs(current_avg - self.last_avg_risk)

        if delta > self.RISK_DELTA_THRESHOLD:
            print(f"[AmbulancePlacer] Risk profile shifted by {delta:.3f} "
                  f"(threshold={self.RISK_DELTA_THRESHOLD}) → triggering GA re-run.")
            self.last_avg_risk = current_avg
            self._rerun_ga(reason="risk_profile_shift")
        else:
            print(f"[AmbulancePlacer] Risk delta={delta:.3f} below threshold — no re-run needed.")

    def _rerun_ga(self, reason: str):
        """Internal: clears old placement and re-runs GA."""
        # Clear old placement markers
        for node in self.current_placement:
            node.ambulance_here = False

        # Re-run with reduced parameters for speed (this is a live re-trigger)
        fast_config = dict(self.config)
        fast_config["population_size"] = 60
        fast_config["max_generations"] = 100

        self.current_placement, self.current_fitness = run_ga(self.graph, fast_config)
        self.flood_count_since_last_run = 0
        print(f"[AmbulancePlacer] Re-run complete (reason={reason}). "
              f"New fitness: {self.current_fitness:.4f}")

    def get_current_placement(self) -> List[CityNode]:
        return self.current_placement

    def get_current_fitness(self) -> float:
        return self.current_fitness
```

### Phase 7 Done Criteria

- [ ] `initial_placement()` runs the GA and stores result in `self.current_placement`
- [ ] `notify_road_flooded()` accumulates count and triggers at threshold 3
- [ ] `notify_risk_updated()` compares current vs last average risk
- [ ] `_rerun_ga()` clears old `ambulance_here` flags before re-running
- [ ] Re-run uses reduced parameters (pop=60, gen=100) for speed

---

## PHASE 8 — Build Feature Extraction for Challenge 5

### What This Phase Is

Challenge 5 starts with reading the city graph and computing features for each node. The two features are: `population_density` (already on each node) and `industrial_proximity` (computed from graph distance to nearest Industrial node). This phase builds the feature extraction pipeline.

### Implementation

```python
# FILE: challenge5/features.py

import numpy as np
from typing import List, Tuple, Dict
from city.node import CityNode
from city.graph import CityGraph
from challenge3.fitness import multi_source_dijkstra  # Reuse your Dijkstra


def compute_industrial_proximity(graph: CityGraph) -> Dict[CityNode, float]:
    """
    For every node, computes how close it is to the nearest Industrial zone.
    
    Method: Multi-Source Dijkstra from ALL industrial nodes simultaneously.
    Result: dist[node] = shortest risk-adjusted distance to nearest Industrial.
    
    Then invert: proximity = 1 - (dist / max_dist)
    So nodes CLOSE to industry get HIGH proximity scores.
    Nodes FAR from industry get LOW proximity scores.
    
    Why not Manhattan distance?
    - Manhattan ignores the actual road network.
    - A node 2 blocks away but separated by a wall has Manhattan dist=2
      but actual road distance >> 2.
    - Using Dijkstra on the actual graph gives meaningful proximity.
    
    Returns:
        Dict mapping every CityNode to its industrial_proximity float [0, 1]
    """
    industrial_nodes = graph.industrial_nodes()

    if not industrial_nodes:
        # No industrial zones — all proximity values are 0 (no industrial influence)
        return {node: 0.0 for node in graph.all_nodes()}

    # Multi-source Dijkstra from all industrial nodes
    # Note: we use base weights only (not risk-adjusted) for proximity computation
    # because we want geographic distance, not travel difficulty
    dist_to_industry: Dict[CityNode, float] = {}
    import heapq
    pq = [(0.0, id(n), n) for n in industrial_nodes]
    import heapq as hq
    hq.heapify(pq)

    while pq:
        cost, _, node = hq.heappop(pq)
        if node in dist_to_industry:
            continue
        dist_to_industry[node] = cost
        for neighbor, base_weight in node.neighbors.items():
            if neighbor not in dist_to_industry:
                hq.heappush(pq, (cost + base_weight, id(neighbor), neighbor))

    # Find max finite distance for normalization
    finite_dists = [d for d in dist_to_industry.values() if d != float('inf')]
    max_dist = max(finite_dists) if finite_dists else 1.0

    proximity = {}
    for node in graph.all_nodes():
        d = dist_to_industry.get(node, float('inf'))
        if d == float('inf') or max_dist == 0:
            proximity[node] = 0.0
        else:
            proximity[node] = 1.0 - (d / max_dist)  # Closer → higher value

    return proximity


def extract_and_normalize_features(graph: CityGraph) -> Tuple[List[CityNode], np.ndarray, np.ndarray]:
    """
    Extracts [population_density, industrial_proximity] for every node.
    Applies Min-Max normalization to scale both features to [0, 1].
    
    Returns:
        nodes_list:    ordered list of all CityNode objects
        X_normalized:  numpy array shape (n_nodes, 2), values in [0, 1]
        X_raw:         numpy array shape (n_nodes, 2), original values (for interpretation)
    
    The nodes_list preserves the ordering so that nodes_list[i] corresponds
    to X_normalized[i] — critical for the classifier to write predictions back.
    """
    nodes_list = list(graph.all_nodes())
    proximity_map = compute_industrial_proximity(graph)

    # Build raw feature matrix
    X_raw = np.array(
        [[node.population_density, proximity_map[node]] for node in nodes_list],
        dtype=float
    )

    # Min-Max normalization: scaled = (x - min) / (max - min)
    feature_min = X_raw.min(axis=0)   # Shape (2,)
    feature_max = X_raw.max(axis=0)   # Shape (2,)
    feature_range = feature_max - feature_min

    # Protect against zero range (all values identical → set range to 1 to avoid division by zero)
    feature_range[feature_range == 0] = 1.0

    X_normalized = (X_raw - feature_min) / feature_range

    print(f"[C5-Features] Extracted features for {len(nodes_list)} nodes.")
    print(f"  Population density range: [{feature_min[0]:.1f}, {feature_max[0]:.1f}]")
    print(f"  Industrial proximity range: [{X_raw[:,1].min():.4f}, {X_raw[:,1].max():.4f}]")

    return nodes_list, X_normalized, X_raw
```

### Phase 8 Done Criteria

- [ ] `compute_industrial_proximity()` uses Dijkstra (not Manhattan)
- [ ] Uses base weights (not risk-adjusted) for geographic proximity
- [ ] Handles case where no Industrial nodes exist (returns all-zero dict)
- [ ] `extract_and_normalize_features()` returns `nodes_list`, `X_normalized`, `X_raw`
- [ ] Min-Max normalization with zero-range protection
- [ ] Feature ranges printed for verification

---

## PHASE 9 — Build K-Means Clustering (Challenge 5 Phase A)

### What This Phase Is

K-Means groups the city's neighborhoods into clusters based on their feature similarity. This is the **unsupervised** step — no labels needed. The clusters become "socio-economic profiles" that inform the synthetic data generation in the next phase.

### Implementation

```python
# FILE: challenge5/clustering.py

import numpy as np
from typing import Tuple, Dict, List
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from city.node import CityNode


def run_kmeans_clustering(X_normalized: np.ndarray,
                           k_range: Tuple[int, int] = (2, 6)) -> Tuple[int, np.ndarray, KMeans]:
    """
    Runs K-Means for each k in k_range and selects the best using Silhouette Score.
    
    Silhouette Score measures: how similar is each point to its own cluster
    compared to other clusters? Range: [-1, 1]. Higher = better clusters.
    
    We test k=2 through k=5 and pick the k with the highest Silhouette Score.
    This is more principled than just hardcoding k=3.
    
    Returns:
        best_k:         the optimal number of clusters
        cluster_labels: numpy array of shape (n_nodes,) with cluster ID per node
        best_model:     fitted KMeans object (use best_model.cluster_centers_ for centroids)
    """
    best_k = k_range[0]
    best_score = -1.0
    best_model = None

    print("\n[C5-Clustering] Searching for optimal k:")
    for k in range(k_range[0], k_range[1]):
        model = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = model.fit_predict(X_normalized)

        # Silhouette requires at least 2 clusters with at least 1 point each
        if len(set(labels)) < 2:
            print(f"  k={k}: Skipped (degenerate clustering)")
            continue

        score = silhouette_score(X_normalized, labels)
        print(f"  k={k}: Silhouette Score = {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_model = model

    cluster_labels = best_model.labels_
    print(f"  → Best k = {best_k} (Silhouette = {best_score:.4f})\n")

    return best_k, cluster_labels, best_model


def interpret_clusters(best_model: KMeans,
                        best_k: int) -> Dict[int, str]:
    """
    Assigns human-readable socio-economic profile labels to each cluster
    based on the position of its centroid in normalized feature space.
    
    Feature space axes:
      X-axis (feature 0): population_density (normalized)
      Y-axis (feature 1): industrial_proximity (normalized)
    
    Quadrant mapping (from Phase 1 design + Muntazir's real-world observation):
      High pop  + High prox  → "Urban Industrial"  (high crime risk potential)
      Low pop   + High prox  → "Industrial Fringe"  (medium risk — workers, less oversight)
      High pop  + Low prox   → "Residential Core"   (low risk — community presence)
      Low pop   + Low prox   → "Rural Peripheral"   (low risk — sparse, low activity)
    """
    centroids = best_model.cluster_centers_  # Shape: (k, 2)
    cluster_profiles: Dict[int, str] = {}

    print("[C5-Clustering] Cluster Profiles:")
    for cluster_id in range(best_k):
        pop_norm = centroids[cluster_id][0]
        ind_norm = centroids[cluster_id][1]

        if ind_norm >= 0.5:
            profile = "Urban Industrial" if pop_norm >= 0.5 else "Industrial Fringe"
        else:
            profile = "Residential Core" if pop_norm >= 0.5 else "Rural Peripheral"

        cluster_profiles[cluster_id] = profile
        print(f"  Cluster {cluster_id}: '{profile}' "
              f"(pop={pop_norm:.2f}, prox={ind_norm:.2f})")

    return cluster_profiles
```

### Phase 9 Done Criteria

- [ ] K-Means tested for k in range [2, 5]
- [ ] Silhouette Score computed for each k and printed
- [ ] Best k selected automatically (not hardcoded)
- [ ] Cluster profiles assigned based on centroid quadrant
- [ ] All 4 possible profiles ("Urban Industrial", "Industrial Fringe", "Residential Core", "Rural Peripheral") handled

---

## PHASE 10 — Build Synthetic Dataset Generator (Challenge 5 Phase B)

### What This Phase Is

This is the most creatively important phase of Challenge 5. You generate labeled training data from scratch, using logic YOU justify. The rule system you build here directly determines what the Decision Tree learns about crime risk.

### The Rule System (Muntazir's Design)

Based on the Phase 1 design document observation: *"areas close to industrial zones tend to have higher crime rates due to economic pressure, transient populations, and lower policing levels."*

```
PRIMARY RULE (industrial proximity drives risk):
  ind >= 0.7 AND pop >= 0.4          → HIGH
  ind >= 0.5 OR  pop >= 0.7          → MEDIUM
  otherwise                          → LOW

SECONDARY MODIFIER (location type adjusts prediction):
  TYPE = Industrial or Power Plant   → bump UP one level
  TYPE = School AND ind >= 0.5       → bump UP one level (vulnerable population near industry)
  TYPE = Hospital or Depot           → bump DOWN one level (presence of emergency services = safer)

NOISE (realism + prevents perfect overfit):
  10% random chance to shift label by ±1 level
```

### Implementation

```python
# FILE: challenge5/dataset.py

import random
import numpy as np
import pandas as pd
from typing import List, Dict
from city.node import CityNode


LOCATION_TYPE_ENCODING = {
    CityNode.TYPE_RESIDENTIAL:    0,
    CityNode.TYPE_HOSPITAL:       1,
    CityNode.TYPE_SCHOOL:         2,
    CityNode.TYPE_INDUSTRIAL:     3,
    CityNode.TYPE_POWER_PLANT:    4,
    CityNode.TYPE_AMBULANCE_DEPOT: 5,
}

RISK_SCALE = ["Low", "Medium", "High"]


def _bump(risk: str, direction: int) -> str:
    """Moves risk up (+1) or down (-1) one level, clamped to [Low, High]."""
    idx = RISK_SCALE.index(risk)
    return RISK_SCALE[max(0, min(2, idx + direction))]


def assign_ground_truth_risk(nodes_list: List[CityNode],
                               X_normalized: np.ndarray,
                               cluster_labels: np.ndarray,
                               cluster_profiles: Dict[int, str]) -> List[dict]:
    """
    Assigns a Ground Truth risk label to every node using the rule system
    designed by Muntazir Mehdi (Phase 1 document).
    
    Returns a list of dicts — one per node — ready for DataFrame construction.
    Each dict has: population_density, industrial_proximity,
                   location_type_encoded, cluster_id, risk_label
    """
    dataset = []
    label_counts = {"Low": 0, "Medium": 0, "High": 0}

    for i, node in enumerate(nodes_list):
        pop = float(X_normalized[i][0])       # Normalized [0,1]
        ind = float(X_normalized[i][1])       # Normalized [0,1]
        cluster_id = int(cluster_labels[i])
        loc_type = node.location_type

        # ── PRIMARY RULE ──────────────────────────────────────────────────
        if ind >= 0.7 and pop >= 0.4:
            risk = "High"
        elif ind >= 0.5 or pop >= 0.7:
            risk = "Medium"
        else:
            risk = "Low"

        # ── SECONDARY MODIFIER ────────────────────────────────────────────
        if loc_type in (CityNode.TYPE_INDUSTRIAL, CityNode.TYPE_POWER_PLANT):
            risk = _bump(risk, +1)   # Industrial areas are inherently risky
        elif loc_type == CityNode.TYPE_SCHOOL and ind >= 0.5:
            risk = _bump(risk, +1)   # School near industry — vulnerable population
        elif loc_type in (CityNode.TYPE_HOSPITAL, CityNode.TYPE_AMBULANCE_DEPOT):
            risk = _bump(risk, -1)   # Emergency services presence reduces effective risk

        # ── NOISE INJECTION (10%) ─────────────────────────────────────────
        if random.random() < 0.10:
            direction = random.choice([-1, 1])
            risk = _bump(risk, direction)

        label_counts[risk] += 1

        dataset.append({
            "population_density":    pop,
            "industrial_proximity":  ind,
            "location_type_encoded": LOCATION_TYPE_ENCODING.get(loc_type, 0),
            "cluster_id":            cluster_id,
            "risk_label":            risk,
        })

    print(f"[C5-Dataset] Synthetic dataset generated ({len(dataset)} samples):")
    for label, count in label_counts.items():
        pct = 100 * count / len(dataset)
        print(f"  {label:6s}: {count:4d} ({pct:.1f}%)")

    return dataset


def build_training_dataframe(dataset: List[dict]) -> pd.DataFrame:
    """Converts dataset list to pandas DataFrame for sklearn."""
    df = pd.DataFrame(dataset)
    return df
```

### Phase 10 Done Criteria

- [ ] Primary rule implemented (thresholds 0.7 and 0.5 for ind, 0.4 and 0.7 for pop)
- [ ] All 3 secondary modifier conditions implemented
- [ ] 10% noise injection applied with `_bump()`
- [ ] Label distribution printed to verify no extreme imbalance (if one class < 10%, reconsider thresholds)
- [ ] Returns a list of dicts with exactly 5 keys matching `FEATURES` list in classifier

---

## PHASE 11 — Build the Decision Tree Classifier (Challenge 5 Phase C)

### What This Phase Is

Train the Decision Tree on the synthetic dataset, apply predictions to the graph, and provide the interface for re-training. This is the supervised learning step of Challenge 5.

### Implementation

```python
# FILE: challenge5/classifier.py

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Tuple
from city.node import CityNode
from city.graph import CityGraph
from challenge5.dataset import LOCATION_TYPE_ENCODING


class CrimeRiskClassifier:
    """
    Decision Tree classifier for crime risk prediction.
    Trains on synthetic data, predicts for all nodes, applies risk multipliers
    to the shared city graph.
    
    Key design choice: Decision Tree over KNN for viva-explainability.
    Can trace any prediction to an explicit IF-THEN rule in the tree.
    """

    FEATURES = [
        "population_density",
        "industrial_proximity",
        "location_type_encoded",
        "cluster_id",
    ]
    TARGET = "risk_label"

    RISK_MULTIPLIERS = {
        "High":   1.5,
        "Medium": 1.2,
        "Low":    1.0,
    }

    # Hyperparameters — justified in Dev Manual §2.4
    HYPERPARAMS = dict(
        max_depth=5,           # Prevents overfitting on ~400 node synthetic dataset
        min_samples_split=4,   # Need 4 samples to justify a split (noise tolerance)
        criterion="gini",      # Standard Gini impurity — slightly faster than entropy
        class_weight="balanced",  # Compensates for unequal High/Medium/Low counts
        random_state=42,       # Reproducibility for demo/viva
    )

    def __init__(self):
        self.model: DecisionTreeClassifier = None
        self.label_encoder = LabelEncoder()
        self.training_df: pd.DataFrame = None
        self.version: int = 0   # Incremented on each re-train

    def train(self, df: pd.DataFrame):
        """
        Trains the Decision Tree on the provided DataFrame.
        Prints accuracy and classification report.
        Prints tree rules (first 2000 chars) for viva preparation.
        """
        self.training_df = df.copy()

        X = df[self.FEATURES].values
        y_raw = df[self.TARGET].values
        y = self.label_encoder.fit_transform(y_raw)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = DecisionTreeClassifier(**self.HYPERPARAMS)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n[C5-Classifier] Decision Tree trained (v{self.version + 1}):")
        print(f"  Training samples: {len(X_train)} | Test samples: {len(X_test)}")
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

        # Print tree rules for viva — this is your explainability proof
        rules = export_text(self.model, feature_names=self.FEATURES)
        print("[Decision Tree Rules (first 2000 chars)]:")
        print(rules[:2000])

        self.version += 1

    def predict_all_nodes(self,
                           nodes_list: List[CityNode],
                           X_normalized: np.ndarray,
                           cluster_labels: np.ndarray) -> Dict[CityNode, str]:
        """
        Runs the trained classifier on all city nodes.
        Returns a dict: node → risk_label ("High"/"Medium"/"Low")
        """
        if self.model is None:
            raise RuntimeError("Call train() before predict_all_nodes().")

        predictions = {}
        for i, node in enumerate(nodes_list):
            features = [
                float(X_normalized[i][0]),
                float(X_normalized[i][1]),
                float(LOCATION_TYPE_ENCODING.get(node.location_type, 0)),
                float(cluster_labels[i]),
            ]
            encoded = self.model.predict([features])[0]
            label = self.label_encoder.inverse_transform([encoded])[0]
            predictions[node] = label

        return predictions

    def apply_predictions_to_graph(self, predictions: Dict[CityNode, str], graph: CityGraph):
        """
        Writes risk_index and predicted_risk to each node in the shared graph.
        Increments graph.version to invalidate Challenge 3's fitness cache.
        """
        changed = 0
        for node, label in predictions.items():
            new_risk = self.RISK_MULTIPLIERS[label]
            if node.risk_index != new_risk:
                changed += 1
            node.risk_index = new_risk
            node.predicted_risk = label

        # Signal all caches to invalidate
        graph.increment_version()

        print(f"[C5-Classifier] Risk indices applied: {changed} nodes changed. "
              f"Graph v→{graph.version}.")
```

### Phase 11 Done Criteria

- [ ] `train()` prints accuracy AND classification report (per-class precision/recall/F1)
- [ ] `export_text()` prints tree rules — save this output before viva
- [ ] `predict_all_nodes()` builds feature vector with same 4 fields as training data
- [ ] `apply_predictions_to_graph()` writes to BOTH `risk_index` AND `predicted_risk`
- [ ] `graph.increment_version()` called after every apply

---

## PHASE 12 — Build the Dynamic Learning Loop (Challenge 5 Phase D)

### What This Phase Is

The most advanced feature of your contribution. The `CrimeStatsManager` monitors real simulation events, detects when the ML model's predictions are wrong, and re-trains the classifier with corrected data. This is what turns your system from a static predictor into a living intelligence.

### Implementation

```python
# FILE: challenge5/monitor.py

import pandas as pd
from typing import List
from city.node import CityNode
from city.graph import CityGraph
from challenge5.classifier import CrimeRiskClassifier
from challenge3.placer import AmbulancePlacer
from challenge5.dataset import LOCATION_TYPE_ENCODING, RISK_SCALE


class CrimeStatsManager:
    """
    Monitors the live simulation and detects when predicted risk labels
    diverge from actual observed behavior (concept drift).
    
    When drift is detected, re-trains the classifier on augmented data
    and propagates updated risk weights through the entire system.
    
    Called from the main simulation loop.
    """

    DRIFT_WINDOW = 5
    LOW_RISK_EMERGENCY_THRESHOLD   = 3    # "Low" node with 3+ emergencies → drifted
    MEDIUM_RISK_EMERGENCY_THRESHOLD = 6   # "Medium" node with 6+ emergencies → drifted

    def __init__(self,
                 graph: CityGraph,
                 classifier: CrimeRiskClassifier,
                 ambulance_placer: AmbulancePlacer):
        self.graph = graph
        self.classifier = classifier
        self.ambulance_placer = ambulance_placer

        self.simulation_step: int = 0

        # These are stored at pipeline init — needed for re-training
        self.nodes_list = None
        self.X_normalized = None
        self.cluster_labels = None

    def set_pipeline_data(self, nodes_list, X_normalized, cluster_labels):
        """Called once after Phase A-C to store data for re-training."""
        self.nodes_list = nodes_list
        self.X_normalized = X_normalized
        self.cluster_labels = cluster_labels

    def notify_emergency(self, node: CityNode):
        """
        Called by Challenge 4 router whenever the medical team dispatches
        to a civilian location.
        
        Increments emergency counter on the target node.
        Also adds a fractional spillover to immediate road neighbors
        (crime presence tends to affect surrounding areas).
        """
        node.total_emergencies += 1.0

        # Spillover effect on road neighbors
        for neighbor in node.neighbors:
            neighbor.total_emergencies += 0.3

    def step(self):
        """
        Called by the main simulation loop at every simulation step.
        Triggers drift check every DRIFT_WINDOW steps.
        """
        self.simulation_step += 1

        if self.simulation_step % self.DRIFT_WINDOW == 0:
            self._check_for_drift()

    def _check_for_drift(self):
        """
        Scans all nodes for concept drift:
        - A "Low Risk" node with >= LOW_RISK_EMERGENCY_THRESHOLD emergencies
        - A "Medium Risk" node with >= MEDIUM_RISK_EMERGENCY_THRESHOLD emergencies
        """
        drifted = []

        for node in self.graph.all_nodes():
            pred = node.predicted_risk
            actual = node.total_emergencies

            is_drifted = (
                (pred == "Low"    and actual >= self.LOW_RISK_EMERGENCY_THRESHOLD) or
                (pred == "Medium" and actual >= self.MEDIUM_RISK_EMERGENCY_THRESHOLD)
            )

            if is_drifted:
                drifted.append(node)
                print(f"  [DRIFT] Node ({node.row},{node.col}): predicted '{pred}' "
                      f"but {actual:.1f} emergencies observed.")

        if drifted:
            print(f"\n[CrimeStatsManager] Step {self.simulation_step}: "
                  f"{len(drifted)} drifted nodes detected → re-training.")
            self._retrain_with_corrections(drifted)

    def _retrain_with_corrections(self, drifted_nodes: List[CityNode]):
        """
        Augments training data with corrected labels for drifted nodes,
        re-trains the Decision Tree, re-applies predictions to graph,
        and notifies the ambulance placer.
        """
        if self.nodes_list is None:
            raise RuntimeError("set_pipeline_data() must be called before drift can be handled.")

        # Build correction records
        corrections = []
        for node in drifted_nodes:
            emergencies = node.total_emergencies
            # Determine corrected label based on how severe the drift is
            if emergencies >= self.MEDIUM_RISK_EMERGENCY_THRESHOLD:
                corrected_label = "High"
            else:
                corrected_label = "Medium"

            node_idx = self.nodes_list.index(node)
            corrections.append({
                "population_density":    float(self.X_normalized[node_idx][0]),
                "industrial_proximity":  float(self.X_normalized[node_idx][1]),
                "location_type_encoded": float(LOCATION_TYPE_ENCODING.get(node.location_type, 0)),
                "cluster_id":            float(self.cluster_labels[node_idx]),
                "risk_label":            corrected_label,
            })

        # Augment original training data: add corrections with 3× weight
        correction_df = pd.DataFrame(corrections)
        augmented_df = pd.concat(
            [self.classifier.training_df] + [correction_df] * 3,
            ignore_index=True
        )

        print(f"  Training data: {len(self.classifier.training_df)} → {len(augmented_df)} rows")

        # Re-train
        self.classifier.train(augmented_df)

        # Re-predict all nodes and apply
        predictions = self.classifier.predict_all_nodes(
            self.nodes_list, self.X_normalized, self.cluster_labels
        )
        self.classifier.apply_predictions_to_graph(predictions, self.graph)

        # Update predicted_risk label on each node
        for node, label in predictions.items():
            node.predicted_risk = label

        # Notify ambulance placer — may trigger GA re-run
        self.ambulance_placer.notify_risk_updated()

        print(f"  [Step {self.simulation_step}] Re-train complete. "
              f"Graph v={self.graph.version}.")
```

### Phase 12 Done Criteria

- [ ] `notify_emergency()` increments `total_emergencies` on target node AND 0.3 spillover to road neighbors
- [ ] `step()` triggers `_check_for_drift()` every `DRIFT_WINDOW` steps
- [ ] Drift detection checks BOTH Low→Medium/High and Medium→High transitions
- [ ] Re-training augments (not replaces) original training data
- [ ] Corrections added 3× to give them sufficient weight against original data
- [ ] After re-train: `apply_predictions_to_graph()` called → `ambulance_placer.notify_risk_updated()` called

---

## PHASE 13 — Build the Challenge 5 Pipeline Entry Point

### What This Phase Is

Wire all the Challenge 5 pieces (Phases 8-12) into a single function that the simulation loop calls once at startup.

### Implementation

```python
# FILE: challenge5/__init__.py  or  challenge5/pipeline.py

from challenge5.features import extract_and_normalize_features
from challenge5.clustering import run_kmeans_clustering, interpret_clusters
from challenge5.dataset import assign_ground_truth_risk, build_training_dataframe
from challenge5.classifier import CrimeRiskClassifier
from challenge5.monitor import CrimeStatsManager
from city.graph import CityGraph
from challenge3.placer import AmbulancePlacer


def run_crime_risk_pipeline(graph: CityGraph,
                              ambulance_placer: AmbulancePlacer) -> CrimeStatsManager:
    """
    Full Challenge 5 pipeline. Call ONCE before the simulation loop.
    
    Execution Order:
      Phase A: Feature extraction + K-Means clustering
      Phase B: Synthetic dataset generation
      Phase C: Decision Tree training + initial graph risk update
      Phase D: CrimeStatsManager setup (monitor runs inside sim loop)
    
    Returns:
        CrimeStatsManager ready for Phase D — call .step() and
        .notify_emergency() from inside the simulation loop.
    """
    print("\n" + "="*55)
    print("CHALLENGE 5: Crime Risk Prediction Pipeline")
    print("="*55)

    # ── PHASE A ──────────────────────────────────────────────────────────────
    print("\n[Phase A] Feature Extraction & K-Means Clustering")
    nodes_list, X_normalized, X_raw = extract_and_normalize_features(graph)
    best_k, cluster_labels, kmeans_model = run_kmeans_clustering(X_normalized)
    cluster_profiles = interpret_clusters(kmeans_model, best_k)

    # Write cluster_id to each node (useful for UI heatmap)
    for i, node in enumerate(nodes_list):
        node.cluster_id = int(cluster_labels[i])   # Optional convenience field

    # ── PHASE B ──────────────────────────────────────────────────────────────
    print("\n[Phase B] Synthetic Dataset Generation")
    dataset = assign_ground_truth_risk(nodes_list, X_normalized, cluster_labels, cluster_profiles)
    df = build_training_dataframe(dataset)

    # ── PHASE C ──────────────────────────────────────────────────────────────
    print("\n[Phase C] Decision Tree Training & Graph Update")
    classifier = CrimeRiskClassifier()
    classifier.train(df)

    predictions = classifier.predict_all_nodes(nodes_list, X_normalized, cluster_labels)
    classifier.apply_predictions_to_graph(predictions, graph)

    for node, label in predictions.items():
        node.predicted_risk = label

    # ── PHASE D SETUP ────────────────────────────────────────────────────────
    print("\n[Phase D] Initializing Dynamic Learning Monitor")
    monitor = CrimeStatsManager(graph, classifier, ambulance_placer)
    monitor.set_pipeline_data(nodes_list, X_normalized, cluster_labels)

    print("\n[Challenge 5] Pipeline complete.")
    print(f"  Graph risk version: {graph.version}")
    print("="*55 + "\n")

    return monitor
```

### Phase 13 Done Criteria

- [ ] All 4 phases execute in the correct order
- [ ] `cluster_id` written to each node as a convenience field
- [ ] `predicted_risk` written to each node after initial classification
- [ ] `CrimeStatsManager` constructed with correct references to graph, classifier, and ambulance_placer
- [ ] Returns the monitor object (not void)

---

## PHASE 14 — Build the Simulation Loop (20-Step Engine)

### What This Phase Is

Assemble the 20-step simulation loop that ties all 5 challenges together. This loop is what the professor will watch during your viva demo.

### Implementation

```python
# FILE: simulation/loop.py

import random
from city.graph import CityGraph
from challenge3.placer import AmbulancePlacer
from challenge4.router import DStarLiteRouter       # Ammar's module
from challenge5.pipeline import run_crime_risk_pipeline
from challenge1.layout import build_city_layout      # Ali's module
from challenge2.roads import build_road_network      # Ali's module
from simulation.logger import EventLogger


def run_simulation(grid_rows: int = 20, grid_cols: int = 20) -> CityGraph:
    """
    Full 20-step CityMind simulation.
    
    Startup sequence:
      Step 0a: Challenge 1 — build city layout (CSP)
      Step 0b: Challenge 2 — build road network (Kruskal + UCS)
      Step 0c: Challenge 5 — run crime risk pipeline (ML)
      Step 0d: Challenge 3 — initial ambulance placement (GA)
    
    Simulation steps 1-20:
      Each step: generate random event → route team → update ML → log
    """
    logger = EventLogger()

    # ── STARTUP ──────────────────────────────────────────────────────────────
    print("╔══════════════════════════════════════╗")
    print("║   CityMind Urban Intelligence System  ║")
    print("╚══════════════════════════════════════╝\n")

    # Challenge 1: Build layout
    graph = CityGraph(grid_rows, grid_cols)
    build_city_layout(graph)                  # Ali's function
    logger.log(0, "LAYOUT", f"City layout initialized: {grid_rows}×{grid_cols} grid.")

    # Challenge 2: Build roads
    build_road_network(graph)                 # Ali's function
    logger.log(0, "ROADS", f"Road network built. Total cost: {graph.total_road_cost():.2f}")

    # Challenge 5: Crime risk pipeline
    ambulance_placer = AmbulancePlacer(graph)
    crime_monitor = run_crime_risk_pipeline(graph, ambulance_placer)
    logger.log(0, "RISK", f"Crime risk predictions applied. Graph v={graph.version}")

    # Challenge 3: Initial ambulance placement
    placement, fitness = ambulance_placer.initial_placement()
    positions = [(n.row, n.col) for n in placement]
    logger.log(0, "AMBULANCE", f"Initial GA placement: {positions}. Worst-case dist: {fitness:.4f}")

    # Challenge 4: Router setup
    civilians = _generate_civilian_list(graph)  # Random set of Residential nodes to visit
    router = DStarLiteRouter(graph, civilians)  # Ammar's class

    # ── SIMULATION STEPS ─────────────────────────────────────────────────────
    for step in range(1, 21):
        print(f"\n{'─'*40}")
        print(f"  SIMULATION STEP {step}/20")
        print(f"{'─'*40}")

        # Generate random event
        event_type = random.choice(["flood", "emergency", "emergency", "quiet"])
        # "emergency" is twice as likely as "flood" or "quiet"

        if event_type == "flood":
            # Pick a random accessible edge and flood it
            flooded_pair = _pick_random_road(graph)
            if flooded_pair:
                u, v = flooded_pair
                u.block_road_to(v)
                logger.log(step, "FLOOD", f"Road ({u.row},{u.col})↔({v.row},{v.col}) flooded.")
                print(f"  [EVENT] FLOOD: Road ({u.row},{u.col})↔({v.row},{v.col}) blocked.")

                # Notify router (D* Lite replans)
                router.notify_edge_blocked(u, v)

                # Notify ambulance placer
                ambulance_placer.notify_road_flooded(u, v)

        elif event_type == "emergency":
            # Route to next civilian
            result = router.route_to_next_civilian()
            if result:
                civilian, path, cost = result
                logger.log(step, "ROUTE",
                           f"Team routed to civilian at ({civilian.row},{civilian.col}). "
                           f"Path cost: {cost:.4f}. Hops: {len(path)}.")
                print(f"  [EVENT] EMERGENCY: Routed to ({civilian.row},{civilian.col}). "
                      f"Cost={cost:.4f}")

                # Notify crime monitor (this is the key integration!)
                crime_monitor.notify_emergency(civilian)
            else:
                logger.log(step, "ROUTE", "No reachable civilians remaining.")
                print("  [EVENT] All civilians reached or isolated.")

        else:  # quiet
            logger.log(step, "QUIET", "No events this step.")
            print("  [EVENT] Quiet step.")

        # Challenge 5 step — may detect drift and re-train
        crime_monitor.step()

        # Log current system state
        logger.log(step, "STATE",
                  f"Ambulance positions: {[(n.row,n.col) for n in ambulance_placer.get_current_placement()]}. "
                  f"Worst-case dist: {ambulance_placer.get_current_fitness():.4f}. "
                  f"Graph v={graph.version}.")

    print("\n╔══════════════════════════════════════╗")
    print("║          SIMULATION COMPLETE          ║")
    print("╚══════════════════════════════════════╝")
    logger.print_summary()

    return graph


def _generate_civilian_list(graph: CityGraph, count: int = 8):
    """Randomly selects residential nodes to serve as trapped civilians."""
    residential = [n for n in graph.all_nodes()
                   if n.location_type == CityNode.TYPE_RESIDENTIAL and n.is_accessible]
    return random.sample(residential, min(count, len(residential)))


def _pick_random_road(graph: CityGraph):
    """Picks a random accessible road edge to flood."""
    candidates = []
    for node in graph.all_nodes():
        for neighbor in node.neighbors:
            if id(node) < id(neighbor):  # Count each undirected edge once
                candidates.append((node, neighbor))
    return random.choice(candidates) if candidates else None
```

### Phase 14 Done Criteria

- [ ] Startup sequence: C1 → C2 → C5 → C3 (in this order)
- [ ] Challenge 5 runs BEFORE Challenge 3 initial placement (so GA uses risk-weighted distances)
- [ ] Each sim step generates one of: flood, emergency, or quiet
- [ ] `crime_monitor.notify_emergency(civilian)` called for every routed emergency
- [ ] `crime_monitor.step()` called at end of every simulation step
- [ ] All events logged with step number, type, and description

---

## PHASE 15 — Build the Event Logger

### What This Phase Is

The event log is a graded component (visible in the UI and counted in the rubric). Build it as a clean class that stores entries and formats them for display.

### Implementation

```python
# FILE: simulation/logger.py

from typing import List
from dataclasses import dataclass, field
from datetime import datetime


EVENT_COLORS = {
    "LAYOUT":    "gray",
    "ROADS":     "gray",
    "RISK":      "orange",
    "AMBULANCE": "green",
    "FLOOD":     "red",
    "ROUTE":     "blue",
    "DRIFT":     "orange",
    "RETRAIN":   "orange",
    "GA_RERUN":  "green",
    "QUIET":     "lightgray",
    "STATE":     "darkgray",
}


@dataclass
class LogEntry:
    step: int
    event_type: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


class EventLogger:
    """
    Stores and formats all simulation events.
    Used by the UI to populate the scrollable event log panel.
    """

    def __init__(self):
        self.entries: List[LogEntry] = []

    def log(self, step: int, event_type: str, message: str):
        entry = LogEntry(step=step, event_type=event_type, message=message)
        self.entries.append(entry)
        # Also print to console immediately
        color = EVENT_COLORS.get(event_type, "white")
        print(f"  [{entry.timestamp}][Step {step:2d}][{event_type:9s}] {message}")

    def get_recent(self, n: int = 20) -> List[LogEntry]:
        """Returns the n most recent log entries."""
        return self.entries[-n:]

    def get_by_type(self, event_type: str) -> List[LogEntry]:
        return [e for e in self.entries if e.event_type == event_type]

    def print_summary(self):
        print("\n── EVENT SUMMARY ──────────────────────────────")
        for etype in ["FLOOD", "ROUTE", "DRIFT", "RETRAIN", "GA_RERUN"]:
            count = len(self.get_by_type(etype))
            print(f"  {etype:10s}: {count} events")
        print(f"  TOTAL ENTRIES: {len(self.entries)}")
```

### Phase 15 Done Criteria

- [ ] `LogEntry` dataclass with step, event_type, message, timestamp
- [ ] `log()` appends entry AND prints to console immediately
- [ ] `get_recent(n)` returns last n entries for UI scrolling
- [ ] Color mapping defined for every event type
- [ ] `print_summary()` shows counts by event type

---

## PHASE 16 — Build the Visual Interface

### What This Phase Is

The UI is 15 marks in the rubric. It must show the city grid with 3 overlay toggles: road network, ambulance coverage, and crime risk heatmap. Plus a live event log panel.

### Architecture Decision

Use Python with `tkinter` (built-in, no install needed) or `pygame`. HTML/CSS/JS is also allowed per the professor's announcement. This guide uses tkinter for simplicity. If your group prefers HTML/JS, the grid data can be exported as JSON and rendered in a web page.

### Minimal Interface Specification

```python
# FILE: ui/interface.py  (structure only — visual implementation varies by toolkit)

"""
UI Layout:
┌─────────────────────────────────────────────────────────────────┐
│  CityMind Urban Intelligence System  [Init] [Step] [Run] [Reset]│  ← Top Bar
│                                             Step: 0/20           │
├──────────────────────────────────────┬──────────────────────────┤
│                                      │ ☑ Road Network           │
│         CITY GRID CANVAS             │ ☑ Ambulance Coverage     │  ← Right
│           (20 × 20)                  │ ☑ Crime Risk Heatmap     │    Panel
│                                      │──────────────────────────│
│   Color key:                         │ Total Road Cost: 42.3    │
│   ■ Residential  ■ Hospital          │ Worst-Case Dist: 8.4     │
│   ■ School       ■ Industrial        │ Blocked Roads: 2         │
│   ■ Depot        ■ Power Plant       │ GA Version: 1            │
│                                      │ Risk Model v: 2          │
│   🚑 = Ambulance                     │                          │
├──────────────────────────────────────┴──────────────────────────┤
│ EVENT LOG (scrollable)                                           │
│ [14:22:01][Step  7][FLOOD    ] Road (3,4)↔(3,5) flooded.        │
│ [14:22:01][Step  7][ROUTE    ] Team routed to (12,8). Cost=6.2  │
│ [14:22:02][Step 10][DRIFT    ] Node (7,1): predicted Low, 4 emg │
└─────────────────────────────────────────────────────────────────┘

Node Colors (on grid):
  Residential:    #90CAF9  (light blue)
  Hospital:       #A5D6A7  (light green)
  School:         #FFF59D  (light yellow)
  Industrial:     #BCAAA4  (brown-gray)
  Power Plant:    #CE93D8  (light purple)
  Ambulance Depot:#FF8A65  (orange)

Risk Heatmap Overlay (when enabled):
  Low risk:    node fill stays at type color
  Medium risk: yellow tint border (3px)
  High risk:   red tint border (3px) + slight red fill blend

Ambulance Coverage Overlay (when enabled):
  Draw a circle of radius = current best_fitness around each ambulance node
  or color-code nodes by which ambulance is nearest

Road Network Overlay (when enabled):
  Draw lines between connected neighbors
  Blocked roads shown as dashed red lines
"""
```

### Phase 16 Done Criteria

- [ ] Grid rendered with correct colors per location type
- [ ] Ambulance positions marked with a distinct icon (e.g., "🚑" or filled circle)
- [ ] Crime risk heatmap overlay toggleable (color/border indicates risk level)
- [ ] Ambulance coverage overlay toggleable (shows coverage radius or zone coloring)
- [ ] Road network overlay toggleable (edges drawn between connected nodes)
- [ ] Event log panel shows last 20 entries, color-coded by event type
- [ ] Statistics panel shows total road cost, worst-case distance, blocked road count
- [ ] [Step] button advances simulation by 1 step

---

## PHASE 17 — Integration Testing: Connect All Challenges

### What This Phase Is

This is the first time all 5 challenges run together. The goal is to find and fix integration bugs before the demo. Work through each integration point systematically.

### Integration Test Checklist

**Test 1: Graph version propagation**
```python
# After C5 updates risk, GA cache should be stale
initial_version = graph.version
crime_monitor._retrain_with_corrections([some_node])
assert graph.version == initial_version + 1
assert fitness_cache is empty (all keys will have new version)
print("Integration Test 1: PASSED ✓")
```

**Test 2: Flood → GA re-trigger**
```python
# Flood 3 roads, confirm GA re-runs
for i in range(3):
    ambulance_placer.notify_road_flooded(node_a, node_b)
# Check: ambulance_placer.flood_count_since_last_run should be 0 after 3rd flood
assert ambulance_placer.flood_count_since_last_run == 0  # Reset after re-run
print("Integration Test 2: PASSED ✓")
```

**Test 3: ML drift → risk update → GA uses new weights**
```python
# Manually trigger drift
node = graph.grid[5][5]
node.total_emergencies = 10
node.predicted_risk = "Low"
crime_monitor._check_for_drift()
# After retrain, node.risk_index should be 1.2 or 1.5
assert node.risk_index > 1.0
print("Integration Test 3: PASSED ✓")
```

**Test 4: Blocked road → no inf fitness**
```python
# Block a road into an ambulance depot
depot_node = graph.eligible_ambulance_nodes()[0]
for neighbor in list(depot_node.neighbors.keys()):
    depot_node.block_road_to(neighbor)
# GA should handle this gracefully (depot unreachable → not placed there ideally)
placement, fitness = run_ga(graph)
assert fitness != float('inf') or len(graph.citizen_nodes()) == 0
print("Integration Test 4: PASSED ✓")
```

### Phase 17 Done Criteria

- [ ] All 4 integration tests pass without exception
- [ ] No module holds a stale copy of node data (all reads go to live `node` attributes)
- [ ] `crime_monitor.notify_emergency()` correctly called from simulation loop
- [ ] `ambulance_placer.notify_road_flooded()` correctly called from simulation loop
- [ ] Full 20-step simulation runs without uncaught exceptions

---

## PHASE 18 — Viva Preparation & Demo Hardening

### What This Phase Is

Your code works. Now you must make it unbreakable for the demo and be able to explain everything from first principles.

### Demo Hardening Checklist

**Error handling — wrap these in try/except with informative messages:**
- `run_ga()` if fewer than 3 eligible nodes exist
- `dijkstra_from_source()` if graph has no connected nodes
- `run_crime_risk_pipeline()` if no Industrial nodes exist (proximity = all zeros)
- `CrimeStatsManager._retrain_with_corrections()` if `nodes_list` is None

**Demo script — practice this exact sequence:**
1. Launch `main.py`. Show city grid rendered.
2. Point out the color legend (location types).
3. Enable Crime Risk Heatmap. Show red/yellow/green nodes.
4. Point out one Industrial node and the red (High Risk) nodes near it — explain the model learned this.
5. Click Step. Show an emergency route drawn in the event log.
6. Click Step several more times until a flood occurs. Show the event log entry.
7. If GA re-triggers, point it out in the console output.
8. Click Step until drift is detected. Show the re-train output in console.
9. Toggle Ambulance Coverage overlay. Show coverage zones.
10. Narrate: "The ML model updated the risk weights, which forced the GA to re-evaluate the ambulance placement with updated travel costs."

### First-Principles Viva Scripts (Memorize)

**If asked "Explain the GA fitness function from scratch":**
> "For a given placement of 3 ambulances, I run Multi-Source Dijkstra from all 3 simultaneously. This gives me the shortest risk-adjusted distance from every node to its nearest ambulance. Then I find the maximum of those distances — the citizen who is hardest to reach. That maximum is the fitness score. I want to minimize it across generations, because minimizing the worst case guarantees no citizen is left dangerously far from help."

**If asked "How does the ML update affect the ambulance placement?":**
> "Challenge 5 writes updated risk_index values directly onto CityNode objects. The GA's Dijkstra function reads these when computing edge weights: effective_cost equals base_weight times destination.risk_index. So when a node goes from risk 1.0 to 1.5, every path through that node becomes 50% more expensive. The GA's fitness function automatically uses this heavier graph on its next call. Additionally, I increment graph.version, which invalidates my fitness cache. Then I call ambulance_placer.notify_risk_updated() which, if the average risk shifted by more than 0.15, triggers a full GA re-run to find better placements under the new cost landscape."

**If asked "Why re-train on augmented data instead of replacing training data?":**
> "Replacing would cause catastrophic forgetting — the model would lose all the patterns it learned from the original synthetic data. The original data represents the baseline geographic risk patterns (industry proximity driving crime). The augmented corrections represent real simulation events. I want the model to know both. By adding corrections with 3× weight, I make them influential without wiping out the foundation. This is conceptually similar to how transfer learning works in deep learning."

### Phase 18 Done Criteria

- [ ] All error handling added and tested with bad inputs
- [ ] Demo script rehearsed at least twice from start to finish
- [ ] All 8 viva justification answers from the Dev Manual memorized
- [ ] Can walk through `calculate_minimax_fitness()` line by line on a whiteboard
- [ ] Can draw the Decision Tree decision path for one example node
- [ ] Live modification scenarios tested: change ambulances from 3 to 5, change High multiplier from 1.5 to 2.0

---

## Summary: Phase Dependencies & Timeline

```
WEEK 1 (Design, already done):
  Phase 1: CityNode class
  Phase 2: CityGraph class
  Phase 3: Team coordination meeting

WEEK 2 (Implementation — build in this order):
  Day 1:  Phase 4 (Dijkstra engine)
  Day 1:  Phase 5 (GA utility functions)
  Day 2:  Phase 6 (GA main loop)
  Day 2:  Phase 7 (AmbulancePlacer manager)
  Day 3:  Phase 8 (Feature extraction)
  Day 3:  Phase 9 (K-Means clustering)
  Day 4:  Phase 10 (Synthetic dataset generator)
  Day 4:  Phase 11 (Decision Tree classifier)
  Day 5:  Phase 12 (Dynamic learning loop)
  Day 5:  Phase 13 (Pipeline entry point)
  Day 6:  Phase 14 (Simulation loop)
  Day 6:  Phase 15 (Event logger)
  Day 7:  Phase 16 (Visual interface)

WEEK 3 (Demo & Defense):
  Day 1:  Phase 17 (Integration testing)
  Day 2–3: Phase 18 (Viva preparation + demo hardening)
  Day 4–5: Group final rehearsal + submission
```

---

## Quick Reference: Your Writes vs Reads

| Node Field | You WRITE (which phase) | You READ (which phase) |
|---|---|---|
| `risk_index` | Phase 11, 12 (classifier) | Phase 6 (GA Dijkstra) |
| `predicted_risk` | Phase 11, 12 | Phase 12 (drift check) |
| `total_emergencies` | Phase 12 (notify_emergency) | Phase 12 (drift check) |
| `ambulance_here` | Phase 6 (GA result), Phase 7 (re-trigger clear) | Phase 16 (UI display) |
| `location_type` | NEVER (Ali owns this) | Phases 8, 10, 11 |
| `population_density` | NEVER (Ali owns this) | Phases 8, 10 |
| `neighbors` | NEVER (Ali/Ammar own this) | Phase 4 (Dijkstra), Phase 8 |
| `is_accessible` | NEVER (Ammar owns this) | Phase 4 (skip inaccessible) |

---

*Document End*

**Version:** 1.0  
**Author:** Execution Flow Plan for Muntazir Mehdi (24I-0847)  
**Project:** CityMind Urban Intelligence System  
**Deadline:** 10th May, 11:59 PM
