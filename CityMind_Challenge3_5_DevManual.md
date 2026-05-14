# CityMind Urban Intelligence System
## Developer Manual: Challenge 3 & Challenge 5
### Implementation Reference for Muntazir Mehdi (24I-0847)

---

> **Document Purpose:** This is the complete, implementation-ready technical specification for Challenge 3 (Genetic Algorithm Ambulance Placement) and Challenge 5 (Crime Risk Prediction & ML Integration). Every design decision documented here is aligned with the Phase 1 design document submitted by Group 24I-0548 / 24I-0828 / 24I-0847 and with the CityMind Project Statement. This document is your single source of truth for coding, testing, and viva preparation.

---

## Table of Contents

1. [System Context & Shared Graph Contract](#1-system-context--shared-graph-contract)
2. [Part 1 — Challenge 3: Genetic Algorithm for Ambulance Placement](#part-1--challenge-3-genetic-algorithm-for-ambulance-placement)
   - [1.1 Problem Restatement](#11-problem-restatement)
   - [1.2 Search Space Analysis](#12-search-space-analysis)
   - [1.3 Chromosome (State) Representation](#13-chromosome-state-representation)
   - [1.4 Population Initialization](#14-population-initialization)
   - [1.5 The Fitness Function — Minimax Logic](#15-the-fitness-function--minimax-logic)
   - [1.6 Selection Strategy](#16-selection-strategy)
   - [1.7 Crossover Operations](#17-crossover-operations)
   - [1.8 Mutation Operations](#18-mutation-operations)
   - [1.9 Elitism & Generational Loop](#19-elitism--generational-loop)
   - [1.10 Dynamic Re-Trigger Logic](#110-dynamic-re-trigger-logic)
   - [1.11 Complexity Management & Performance](#111-complexity-management--performance)
   - [1.12 Full GA Implementation Blueprint](#112-full-ga-implementation-blueprint)
   - [1.13 Developer Checklist — Challenge 3](#113-developer-checklist--challenge-3)
3. [Part 2 — Challenge 5: Crime Risk Prediction & ML Integration](#part-2--challenge-5-crime-risk-prediction--ml-integration)
   - [2.1 Pipeline Overview](#21-pipeline-overview)
   - [2.2 Phase A: Feature Engineering & K-Means Clustering](#22-phase-a-feature-engineering--k-means-clustering)
   - [2.3 Phase B: Synthetic Dataset Generation](#23-phase-b-synthetic-dataset-generation)
   - [2.4 Phase C: Supervised Classification — Decision Tree](#24-phase-c-supervised-classification--decision-tree)
   - [2.5 Phase D: The Dynamic Learning Loop](#25-phase-d-the-dynamic-learning-loop)
   - [2.6 Developer Checklist — Challenge 5](#26-developer-checklist--challenge-5)
4. [Part 3 — System Integration & Shared Graph](#part-3--system-integration--shared-graph)
   - [3.1 The Risk Multiplier Logic](#31-the-risk-multiplier-logic)
   - [3.2 Synchronization Protocol](#32-synchronization-protocol)
   - [3.3 Data Structures: CityNode & CrimeStatsManager](#33-data-structures-citynode--crimestatsmanager)
   - [3.4 Integration Flowchart (Textual)](#34-integration-flowchart-textual)
   - [3.5 Developer Checklist — Integration](#35-developer-checklist--integration)
5. [Part 4 — Viva Defense & Live Modification Prep](#part-4--viva-defense--live-modification-prep)
   - [4.1 "Why" Justification Bank](#41-why-justification-bank)
   - [4.2 Live Modification Scenarios](#42-live-modification-scenarios)
   - [4.3 First-Principles Walkthrough Scripts](#43-first-principles-walkthrough-scripts)
6. [Appendix A: Complete Pseudocode Reference](#appendix-a-complete-pseudocode-reference)
7. [Appendix B: Python Code Skeleton](#appendix-b-python-code-skeleton)

---

## 1. System Context & Shared Graph Contract

### 1.1 What You MUST Know Before Touching Challenge 3 or 5

Before writing a single line of your Challenge 3 or Challenge 5 code, you must understand the **contract** that the shared city graph enforces. This contract was established in Phase 1 and is the architectural foundation of the whole project.

**The Golden Rule:** You do NOT own the graph. You READ from it and WRITE specific fields to it. You never create your own copy.

```
The CityGraph object is the SINGLE SOURCE OF TRUTH.
Challenge 3 READS:  node positions, edge weights, risk multipliers (set by C5)
Challenge 3 WRITES: ambulance assignment markers on nodes
Challenge 5 READS:  population_density, location_type, industrial_proximity per node
Challenge 5 WRITES: risk_index on each node (this is your write territory)
```

### 1.2 Graph Fields You Depend On

The following fields on every `CityNode` object are relevant to your work. Ammar (Challenge 4 / Graph module) owns their initialization, but you read and write them:

| Field | Type | Owner (sets it) | You (read/write) | Description |
|---|---|---|---|---|
| `row, col` | int | C1 (layout) | READ | Grid coordinates — used in Manhattan distance heuristic |
| `location_type` | str | C1 (layout) | READ | "Residential", "Hospital", "Industrial", etc. |
| `population_density` | float | C1 (layout) | READ | Number of occupants — primary ML feature |
| `risk_index` | float | **YOU (C5)** | READ + WRITE | Multiplier: 1.0, 1.2, or 1.5 — affects edge weights |
| `is_accessible` | bool | C4 (router) | READ | If False, skip this node in GA fitness calculation |
| `neighbors` | dict{node: weight} | C2 (roads) | READ | Adjacency list — this is the weighted graph for Dijkstra |
| `total_emergencies` | int | **YOU (C5)** | READ + WRITE | Counter you add — tracks real-time incident density |
| `ambulance_here` | bool | **YOU (C3)** | READ + WRITE | Set True when a chromosome places an ambulance here |

### 1.3 The Effective Edge Weight Formula

This formula is the backbone of both your challenges. Every time a pathfinding step queries the cost of moving from node `u` to node `v`:

```
effective_weight(u, v) = base_edge_weight(u,v) × destination_risk_multiplier(v)
```

Where:
- `base_edge_weight(u,v)` = `u.neighbors[v]` — set by Challenge 2 (Kruskal's MST). Standard roads = 1.0, residential roads = 0.8.
- `destination_risk_multiplier(v)` = `v.risk_index` — set by your ML classifier in Challenge 5.

**Critical implication:** When your ML model updates `risk_index` on any node, the graph immediately becomes "heavier" or "lighter" at those locations. Challenge 4's router and your GA's fitness function will automatically compute different costs on the next call — because they both use this same formula against the same node objects. No manual synchronization needed.

---

## Part 1 — Challenge 3: Genetic Algorithm for Ambulance Placement

---

### 1.1 Problem Restatement

From the project statement: *"The city has three ambulances. These ambulances need to be positioned at locations on the grid such that no citizen is unreasonably far from help. The objective is to minimize the worst-case response time."*

Decoded, this is a **Minimax Facility Placement Problem**:
- You have a set of **demand points** (citizen nodes — all Residential, School, and Hospital nodes).
- You have a set of **candidate positions** (all accessible nodes — or restricted to Depot/Hospital nodes per your Phase 1 decision, which must be documented).
- You must place exactly **3 ambulances** at candidate positions.
- The **quality** of a placement is measured by the **maximum** over all demand points of the **minimum** distance from that demand point to its nearest ambulance.
- You want to **minimize** this maximum. Hence: **minimax**.

**Formally:**
```
Minimize:  max over all citizens c of  min over all ambulances a of  dist(c, a)
```

This is NP-hard for large graphs. Exact solution requires evaluating C(N,3) combinations which is infeasible. The GA finds a near-optimal solution efficiently.

---

### 1.2 Search Space Analysis

| Grid Size | Eligible Positions (N) | C(N,3) Combinations | Exhaustive Feasible? |
|---|---|---|---|
| 5×5 = 25 | ~15 | 455 | Yes (trivially) |
| 10×10 = 100 | ~60 | 34,220 | Borderline |
| 15×15 = 225 | ~130 | 366,080 | No — too slow |
| 20×20 = 400 | ~220 | 1,752,780 | Absolutely not |

For a realistic CityMind grid (likely 15×15 or 20×20), exhaustive search would require evaluating **hundreds of thousands to over a million** configurations, running Dijkstra for each. This is the justification for GA.

**GA Advantage:** With a population of 100 chromosomes and 200 generations, you evaluate at most **20,000 configurations** — roughly 100x fewer — while using crossover and mutation to intelligently explore the space rather than brute-force it.

---

### 1.3 Chromosome (State) Representation

A **chromosome** is one candidate solution — one assignment of 3 ambulances to positions.

#### Definition
```python
# A chromosome is a Python list of exactly 3 CityNode objects
# Each CityNode is a valid, accessible position in the graph
chromosome = [node_A, node_B, node_C]

# Example (by node IDs for readability):
chromosome = [graph.grid[2][5], graph.grid[8][1], graph.grid[14][11]]
```

#### Constraints on a Valid Chromosome
1. **No duplicates:** All 3 nodes must be distinct positions.
2. **All accessible:** `node.is_accessible == True` for all 3 positions.
3. **Valid placement zone:** Per your Phase 1 design, you documented that ambulances are placed at Depot or Hospital nodes. Enforce this here. If your group decided ambulances can be at ANY node, remove this constraint but document the change.

```python
def is_valid_chromosome(chromosome: list) -> bool:
    """
    Returns True if a chromosome satisfies all placement constraints.
    """
    # Check no duplicate nodes
    if len(set(id(n) for n in chromosome)) != 3:
        return False
    # Check all nodes are accessible
    if not all(n.is_accessible for n in chromosome):
        return False
    # Check placement zone constraint (adjust if your group allows any node)
    allowed_types = {"Ambulance Depot", "Hospital"}
    if not all(n.location_type in allowed_types for n in chromosome):
        return False
    return True
```

#### Why This Representation is Powerful
- **Always valid by construction.** If you initialize chromosomes by sampling from the eligible node list, and mutation only swaps to other eligible nodes, you will **never produce an invalid chromosome**. There is no "repair" step needed — every individual in your population is a legal solution.
- **No encoding/decoding overhead.** You work directly with `CityNode` objects — no integer-to-node lookup table needed.

---

### 1.4 Population Initialization

```python
import random

def initialize_population(graph, population_size: int = 100) -> list:
    """
    Creates the initial population for the GA.
    
    Args:
        graph: The shared CityGraph object
        population_size: Number of chromosomes in the population (default 100)
    
    Returns:
        A list of chromosomes, each being a list of 3 CityNode objects
    """
    # Step 1: Collect all eligible positions
    eligible_nodes = [
        node 
        for row in graph.grid 
        for node in row
        if node.is_accessible and node.location_type in {"Ambulance Depot", "Hospital"}
    ]
    
    if len(eligible_nodes) < 3:
        raise ValueError(
            f"Cannot place 3 ambulances: only {len(eligible_nodes)} eligible positions exist."
        )
    
    population = []
    
    # Step 2: Generate random chromosomes
    attempts = 0
    while len(population) < population_size:
        # random.sample guarantees no duplicates
        chromosome = random.sample(eligible_nodes, 3)
        population.append(chromosome)
        attempts += 1
        if attempts > population_size * 10:
            raise RuntimeError("Too few eligible nodes to generate diverse population.")
    
    return population
```

**Checklist for Population Initialization:**
- [ ] Eligible nodes are refreshed from the live graph (not a cached copy)
- [ ] `random.sample` is used (not `random.choices`) to prevent duplicate nodes per chromosome
- [ ] Population size is configurable (default 100, increase to 200 for larger grids)
- [ ] A meaningful error is raised if fewer than 3 eligible positions exist

---

### 1.5 The Fitness Function — Minimax Logic

This is the **heart** of Challenge 3. Get this right and everything else follows.

#### Objective (Recap)
```
fitness(chromosome) = max over all citizens c of  min over all ambulances a of  shortest_path_dist(c, a)
```

We want to **minimize** this value across generations.

#### What Counts as a "Citizen"?
For the fitness calculation, "citizens" are all nodes with `population_density > 0`. In practice, this includes:
- All `Residential` nodes
- All `School` nodes (students as occupants)
- All `Hospital` nodes (patients + staff)

Nodes with no population (Industrial, Power Plants, pure road junctions) can optionally be included but their inclusion doesn't change the minimax result significantly and increases computation time.

```python
def get_citizen_nodes(graph) -> list:
    """Returns all nodes that have a population (demand points)."""
    return [
        node
        for row in graph.grid
        for node in row
        if node.population_density > 0 and node.is_accessible
    ]
```

#### Dijkstra From Each Ambulance Position

The core sub-problem: given one ambulance at position `a`, what is the shortest path to every citizen?

```python
import heapq

def dijkstra_from_source(graph, source_node) -> dict:
    """
    Runs Dijkstra from source_node on the weighted graph.
    Returns a dict mapping each reachable node -> shortest distance.
    
    IMPORTANT: Uses effective_weight = base_weight × destination.risk_index
    This ensures risk multipliers from Challenge 5 are respected.
    """
    dist = {}
    # Priority queue: (cost, node_id, node)
    pq = [(0.0, id(source_node), source_node)]
    
    while pq:
        current_cost, _, current_node = heapq.heappop(pq)
        
        # Skip if already found a shorter path
        if current_node in dist:
            continue
        
        dist[current_node] = current_cost
        
        # Explore neighbors
        for neighbor, base_weight in current_node.neighbors.items():
            if not neighbor.is_accessible:
                continue  # Skip blocked nodes
            
            # THE KEY LINE: apply risk multiplier from Challenge 5
            effective_cost = base_weight * neighbor.risk_index
            new_cost = current_cost + effective_cost
            
            if neighbor not in dist:
                heapq.heappush(pq, (new_cost, id(neighbor), neighbor))
    
    return dist
```

#### The Full Fitness Function

```python
def calculate_fitness(chromosome: list, graph) -> float:
    """
    Calculates the minimax fitness of a chromosome.
    
    Lower fitness = better (we are minimizing worst-case response time).
    
    Returns: the maximum shortest-path distance from any citizen to their
             nearest ambulance. Returns float('inf') if any citizen is
             completely unreachable.
    
    Steps:
    1. Run Dijkstra once from each of the 3 ambulance positions
    2. For each citizen, find the minimum distance across all 3 Dijkstra results
    3. Return the maximum of those minimum distances
    """
    citizen_nodes = get_citizen_nodes(graph)
    
    if not citizen_nodes:
        return 0.0  # Edge case: no citizens
    
    # Step 1: Run Dijkstra from each ambulance position
    # This gives us 3 distance maps: dist_maps[i][node] = distance from ambulance i
    dist_maps = []
    for ambulance_node in chromosome:
        dist_map = dijkstra_from_source(graph, ambulance_node)
        dist_maps.append(dist_map)
    
    # Step 2 & 3: Find worst-case distance
    worst_case_distance = 0.0
    
    for citizen in citizen_nodes:
        # Find the nearest ambulance to this citizen
        min_dist_to_nearest_ambulance = float('inf')
        
        for dist_map in dist_maps:
            d = dist_map.get(citizen, float('inf'))
            if d < min_dist_to_nearest_ambulance:
                min_dist_to_nearest_ambulance = d
        
        # The worst-case is the citizen who is farthest from ANY ambulance
        if min_dist_to_nearest_ambulance > worst_case_distance:
            worst_case_distance = min_dist_to_nearest_ambulance
    
    return worst_case_distance
```

#### Fitness Optimization Note: Multi-Source Dijkstra (Advanced)

Running Dijkstra 3 separate times per chromosome evaluation is correct but can be slow for large populations. An optimization is **Multi-Source Dijkstra**: initialize the priority queue with all 3 ambulance positions simultaneously (each at cost 0), then run a single Dijkstra pass. The result gives `dist[node]` = minimum distance from the nearest ambulance.

```python
def multi_source_dijkstra(graph, sources: list) -> dict:
    """
    Optimized: runs Dijkstra from multiple sources simultaneously.
    Returns dist[node] = distance to nearest source.
    3x faster than running single-source Dijkstra 3 times separately.
    """
    dist = {}
    pq = []
    
    # Initialize with all ambulance positions at cost 0
    for source in sources:
        heapq.heappush(pq, (0.0, id(source), source))
    
    while pq:
        current_cost, _, current_node = heapq.heappop(pq)
        
        if current_node in dist:
            continue
        dist[current_node] = current_cost
        
        for neighbor, base_weight in current_node.neighbors.items():
            if not neighbor.is_accessible:
                continue
            effective_cost = base_weight * neighbor.risk_index
            new_cost = current_cost + effective_cost
            if neighbor not in dist:
                heapq.heappush(pq, (new_cost, id(neighbor), neighbor))
    
    return dist


def calculate_fitness_optimized(chromosome: list, graph) -> float:
    """
    Optimized fitness using multi-source Dijkstra.
    Equivalent result to calculate_fitness() but ~3x faster.
    """
    citizen_nodes = get_citizen_nodes(graph)
    if not citizen_nodes:
        return 0.0
    
    dist_from_nearest = multi_source_dijkstra(graph, chromosome)
    
    worst_case = max(
        dist_from_nearest.get(citizen, float('inf'))
        for citizen in citizen_nodes
    )
    
    return worst_case
```

**Use `calculate_fitness_optimized` in production.** The pedagogical version (`calculate_fitness`) is clearer for viva explanations; the optimized version is what runs in your code.

#### Fitness Caching (Critical for Performance)

```python
# Cache Dijkstra results to avoid recomputation for unchanged graphs
_fitness_cache = {}
_graph_version = 0  # Increment this when graph weights change

def get_fitness_cached(chromosome_key: tuple, chromosome: list, graph) -> float:
    """
    chromosome_key: tuple of node IDs (hashable representation)
    Invalidate cache by incrementing _graph_version when risk weights update.
    """
    global _fitness_cache, _graph_version
    
    cache_key = (chromosome_key, _graph_version)
    if cache_key in _fitness_cache:
        return _fitness_cache[cache_key]
    
    fitness = calculate_fitness_optimized(chromosome, graph)
    _fitness_cache[cache_key] = fitness
    return fitness
```

---

### 1.6 Selection Strategy

#### Tournament Selection (Recommended)

For each parent needed, run a **tournament**: randomly pick `k` chromosomes from the population, return the one with the best (lowest) fitness.

```python
def tournament_selection(population: list, fitness_scores: list, tournament_size: int = 5) -> list:
    """
    Selects one parent chromosome via tournament selection.
    
    Args:
        population: List of chromosomes
        fitness_scores: Parallel list of fitness values (lower = better)
        tournament_size: How many candidates compete (default 5)
    
    Returns:
        The winning chromosome (list of 3 nodes)
    """
    # Randomly select tournament candidates
    indices = random.sample(range(len(population)), tournament_size)
    
    # Find the one with lowest fitness (best solution)
    winner_idx = min(indices, key=lambda i: fitness_scores[i])
    
    return population[winner_idx]
```

#### Why Tournament Selection over Roulette Wheel?

| Property | Tournament Selection | Roulette Wheel Selection |
|---|---|---|
| **Handles equal fitness** | Works — random tiebreak | Fails — equal fitness = equal probability = no selection pressure |
| **Handles very large fitness differences** | Works — winner takes all | Fails — high-fitness individuals dominate, diversity lost |
| **Computation** | O(k) per selection | O(n) per selection (need total fitness sum) |
| **Parameter control** | `tournament_size` controls pressure | `scaling` needed for control |
| **Viva explainability** | Very easy to explain | Requires explaining fitness-proportionate weighting |

**Viva Answer:** "We chose Tournament Selection because our fitness values (response distances) can vary widely as the grid evolves, especially after Challenge 5 updates risk weights. Roulette Wheel breaks down under large variance. Tournament Selection maintains selection pressure regardless of the absolute fitness scale."

---

### 1.7 Crossover Operations

Crossover produces two children by combining parts of two parent chromosomes.

#### Single-Point Crossover (Recommended for this problem)

```python
def crossover(parent1: list, parent2: list, eligible_nodes: list) -> tuple:
    """
    Performs single-point crossover between two parent chromosomes.
    
    Strategy:
    - Split each parent at index 1 (crossover point)
    - Child 1 = parent1[0] + parent2[1:]
    - Child 2 = parent2[0] + parent1[1:]
    - Repair duplicates if any node appears twice
    
    Returns:
        Tuple (child1, child2), each a list of 3 valid, distinct nodes
    """
    # Crossover point is after index 0
    child1 = [parent1[0]] + parent2[1:]
    child2 = [parent2[0]] + parent1[1:]
    
    child1 = repair_chromosome(child1, eligible_nodes)
    child2 = repair_chromosome(child2, eligible_nodes)
    
    return child1, child2


def repair_chromosome(chromosome: list, eligible_nodes: list) -> list:
    """
    Repairs a chromosome that may have duplicate nodes after crossover.
    Replaces duplicates with random unused eligible nodes.
    """
    seen = set()
    result = []
    
    for node in chromosome:
        if id(node) not in seen:
            seen.add(id(node))
            result.append(node)
    
    # Fill missing slots with random eligible nodes not already used
    available = [n for n in eligible_nodes if id(n) not in seen]
    
    while len(result) < 3:
        if not available:
            raise RuntimeError("Not enough eligible nodes to repair chromosome.")
        replacement = random.choice(available)
        result.append(replacement)
        available.remove(replacement)
        seen.add(id(replacement))
    
    return result
```

#### Crossover Probability

Apply crossover with probability `CROSSOVER_RATE = 0.85`. If two parents are not crossed over, they pass directly to the next generation (with possible mutation).

```python
CROSSOVER_RATE = 0.85  # 85% of the time, perform crossover

def maybe_crossover(parent1, parent2, eligible_nodes):
    if random.random() < CROSSOVER_RATE:
        return crossover(parent1, parent2, eligible_nodes)
    else:
        return list(parent1), list(parent2)  # Pass through unchanged
```

---

### 1.8 Mutation Operations

Mutation introduces new genetic material to prevent the population from converging prematurely to a local optimum.

#### Random Node Swap Mutation

```python
MUTATION_RATE = 0.15  # 15% per chromosome

def mutate(chromosome: list, eligible_nodes: list, mutation_rate: float = MUTATION_RATE) -> list:
    """
    Mutates a chromosome by randomly replacing one ambulance position
    with a different eligible node.
    
    With probability mutation_rate:
    - Pick one of the 3 ambulance positions at random
    - Replace it with a random eligible node not already in the chromosome
    
    Returns:
        The (possibly mutated) chromosome
    """
    if random.random() > mutation_rate:
        return chromosome  # No mutation this time
    
    # Choose which ambulance to relocate
    idx_to_mutate = random.randint(0, 2)
    
    # Find nodes not already in the chromosome
    current_ids = {id(n) for n in chromosome}
    available = [n for n in eligible_nodes if id(n) not in current_ids]
    
    if not available:
        return chromosome  # Cannot mutate — no alternatives
    
    # Replace with a random available node
    chromosome[idx_to_mutate] = random.choice(available)
    return chromosome
```

#### Adaptive Mutation Rate (Advanced Feature — Documents Well)

If the population's best fitness has not improved in `STAGNATION_THRESHOLD` generations, temporarily increase the mutation rate to escape local optima:

```python
def adaptive_mutation_rate(base_rate: float, generations_without_improvement: int,
                            stagnation_threshold: int = 20) -> float:
    """
    Doubles mutation rate if stuck for too long, then resets.
    """
    if generations_without_improvement > stagnation_threshold:
        return min(base_rate * 2, 0.4)  # Cap at 40%
    return base_rate
```

---

### 1.9 Elitism & Generational Loop

**Elitism:** Always copy the best `ELITE_COUNT` chromosomes to the next generation unchanged. This guarantees the best solution found so far is never lost.

```python
ELITE_COUNT = 2       # Top 2 chromosomes survive unchanged
POPULATION_SIZE = 100
MAX_GENERATIONS = 200
STAGNATION_LIMIT = 30  # Re-trigger if no improvement after 30 generations
```

#### Complete Generational Loop Pseudocode

```
FUNCTION run_genetic_algorithm(graph):

    1. COLLECT eligible_nodes from graph (accessible Depot/Hospital nodes)
    2. COLLECT citizen_nodes from graph (Residential/School/Hospital nodes with population > 0)
    3. population = initialize_population(graph, POPULATION_SIZE)
    4. best_chromosome = None
    5. best_fitness = infinity
    6. generations_without_improvement = 0
    
    FOR generation = 1 to MAX_GENERATIONS:
    
        // --- EVALUATION PHASE ---
        fitness_scores = []
        FOR each chromosome in population:
            f = calculate_fitness_optimized(chromosome, graph)
            fitness_scores.append(f)
        
        // --- TRACK BEST ---
        gen_best_idx = index of minimum in fitness_scores
        gen_best_fitness = fitness_scores[gen_best_idx]
        
        IF gen_best_fitness < best_fitness:
            best_fitness = gen_best_fitness
            best_chromosome = copy of population[gen_best_idx]
            generations_without_improvement = 0
        ELSE:
            generations_without_improvement += 1
        
        // --- EARLY STOPPING ---
        IF generations_without_improvement >= STAGNATION_LIMIT:
            LOG "GA converged at generation {generation}"
            BREAK
        
        // --- BUILD NEXT GENERATION ---
        next_population = []
        
        // Elitism: carry over top ELITE_COUNT chromosomes
        sorted_indices = sort fitness_scores ascending
        FOR i = 0 to ELITE_COUNT - 1:
            next_population.append(copy of population[sorted_indices[i]])
        
        // Adaptive mutation rate
        mut_rate = adaptive_mutation_rate(MUTATION_RATE, generations_without_improvement)
        
        // Fill rest of population through selection + crossover + mutation
        WHILE len(next_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)
            child1, child2 = maybe_crossover(parent1, parent2, eligible_nodes)
            child1 = mutate(child1, eligible_nodes, mut_rate)
            child2 = mutate(child2, eligible_nodes, mut_rate)
            next_population.append(child1)
            IF len(next_population) < POPULATION_SIZE:
                next_population.append(child2)
        
        population = next_population
        
        // --- LOG PROGRESS ---
        IF generation % 10 == 0:
            LOG f"Generation {generation}: Best Fitness = {best_fitness:.4f}"
    
    // --- APPLY RESULT TO GRAPH ---
    FOR each node in best_chromosome:
        node.ambulance_here = True
    
    LOG f"GA Complete. Best worst-case distance = {best_fitness:.4f}"
    LOG f"Ambulance positions: {[str(n) for n in best_chromosome]}"
    
    RETURN best_chromosome, best_fitness
```

---

### 1.10 Dynamic Re-Trigger Logic

The GA does not run only once. It must re-run when the city state changes enough to invalidate the current placement.

#### Trigger Events

```python
class AmbulancePlacer:
    """Manages GA execution and tracks re-trigger conditions."""
    
    def __init__(self, graph):
        self.graph = graph
        self.current_placement = None
        self.current_fitness = float('inf')
        self.flood_count_since_last_run = 0
        self.last_risk_profile = None
        
        # Trigger thresholds
        self.FLOOD_TRIGGER_THRESHOLD = 3    # Re-run after 3 road floods
        self.RISK_CHANGE_THRESHOLD = 0.15   # Re-run if average risk changes by 15%
    
    def notify_road_flooded(self, node_u, node_v):
        """Called by Challenge 4 router when a road is blocked."""
        self.flood_count_since_last_run += 1
        
        if self.flood_count_since_last_run >= self.FLOOD_TRIGGER_THRESHOLD:
            self._log_trigger("Road flood threshold reached")
            self._rerun_ga()
    
    def notify_risk_updated(self):
        """Called by Challenge 5 ML module after updating risk indices."""
        current_profile = self._compute_risk_profile()
        
        if self.last_risk_profile is None:
            self.last_risk_profile = current_profile
            return
        
        # Check if average risk has changed significantly
        delta = abs(current_profile - self.last_risk_profile)
        if delta > self.RISK_CHANGE_THRESHOLD:
            self._log_trigger(f"Risk profile shifted by {delta:.3f}")
            self.last_risk_profile = current_profile
            self._rerun_ga()
    
    def _compute_risk_profile(self) -> float:
        """Returns the average risk_index across all nodes."""
        nodes = [n for row in self.graph.grid for n in row]
        return sum(n.risk_index for n in nodes) / len(nodes)
    
    def _rerun_ga(self):
        """Clears current placement, re-runs GA, updates graph."""
        # Clear old placement markers
        if self.current_placement:
            for node in self.current_placement:
                node.ambulance_here = False
        
        # Re-run GA
        self.current_placement, self.current_fitness = run_genetic_algorithm(self.graph)
        self.flood_count_since_last_run = 0
    
    def _log_trigger(self, reason: str):
        print(f"[AMBULANCE GA] Re-triggered: {reason}")
```

#### Trigger Summary Table

| Trigger Event | Threshold | Who Calls It | Priority |
|---|---|---|---|
| Road flooded | 3 floods accumulated | Challenge 4 router | High |
| Risk weights updated by ML | Avg risk change > 0.15 | Challenge 5 ML module | Medium |
| Simulation step boundary | Every 5 simulation steps | Simulation loop | Low |
| Manual override (viva demo) | Immediate | You (manual call) | Immediate |

---

### 1.11 Complexity Management & Performance

#### Time Complexity Breakdown

| Operation | Complexity | Notes |
|---|---|---|
| Initialize population | O(P) | P = population size (100) |
| Single fitness evaluation | O((V + E) log V) | One Dijkstra on V nodes, E edges |
| Fitness evaluation (full population) | O(P × (V + E) log V) | The bottleneck |
| Tournament selection | O(k × P) | k = tournament size = 5 |
| Crossover + mutation | O(N) | N = eligible nodes |
| **Total per generation** | **O(P × (V+E) log V)** | Dijkstra dominates |
| **Full GA (G generations)** | **O(G × P × (V+E) log V)** | G=200, P=100 |

For a 20×20 grid (V≈400 nodes, E≈800 edges), one generation takes approximately:
- 100 fitness evaluations × one Dijkstra each = 100 × O(400 × log 400) ≈ fast (milliseconds)

#### Optimization Strategies

1. **Use Multi-Source Dijkstra** (already described in §1.5) — reduces 3 Dijkstra calls to 1 per fitness evaluation.

2. **Cache citizen_nodes list** — recompute only when a road event changes accessibility.

3. **Cache fitness scores** — if a chromosome from the previous generation appears unchanged in the new generation (elitism), reuse its fitness score.

4. **Parallel fitness evaluation** (optional, impressive for viva):
```python
from concurrent.futures import ThreadPoolExecutor

def evaluate_population_parallel(population, graph):
    with ThreadPoolExecutor(max_workers=4) as executor:
        fitness_scores = list(executor.map(
            lambda chrom: calculate_fitness_optimized(chrom, graph),
            population
        ))
    return fitness_scores
```

5. **Reduce population for re-triggers** — When the GA re-runs after a minor risk change, use `POPULATION_SIZE = 50` and `MAX_GENERATIONS = 100` instead of full parameters. Only run full GA on initial placement.

---

### 1.12 Full GA Implementation Blueprint

```python
# ============================================================
# CHALLENGE 3: AMBULANCE PLACEMENT — COMPLETE FLOW
# ============================================================

# CONFIGURATION
GA_CONFIG = {
    "population_size": 100,
    "max_generations": 200,
    "tournament_size": 5,
    "crossover_rate": 0.85,
    "mutation_rate": 0.15,
    "elite_count": 2,
    "stagnation_limit": 30,
    "num_ambulances": 3,
}

# MAIN ENTRY POINT
def run_ga_ambulance_placement(graph, config=GA_CONFIG):
    """
    Main entry point for Challenge 3.
    Call this from the simulation loop or AmbulancePlacer._rerun_ga()
    
    Returns:
        best_placement: list of 3 CityNode objects
        best_fitness: float (worst-case response distance)
    """
    # 1. Setup
    eligible_nodes = get_eligible_nodes(graph)
    
    # 2. Initialize
    population = initialize_population(graph, config["population_size"])
    
    best_chromosome = None
    best_fitness = float('inf')
    gens_no_improve = 0
    
    # 3. Generational loop
    for gen in range(1, config["max_generations"] + 1):
        
        # Evaluate
        fitness_scores = [
            calculate_fitness_optimized(chrom, graph)
            for chrom in population
        ]
        
        # Track global best
        gen_best_idx = min(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
        if fitness_scores[gen_best_idx] < best_fitness:
            best_fitness = fitness_scores[gen_best_idx]
            best_chromosome = list(population[gen_best_idx])
            gens_no_improve = 0
        else:
            gens_no_improve += 1
        
        # Log
        if gen % 20 == 0 or gen == 1:
            print(f"  Gen {gen:3d} | Best Fitness: {best_fitness:.4f} | No-Improve: {gens_no_improve}")
        
        # Early stop
        if gens_no_improve >= config["stagnation_limit"]:
            print(f"  [GA] Early stop at generation {gen}")
            break
        
        # Build next generation
        sorted_pop = [population[i] for i in sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])]
        next_gen = [list(c) for c in sorted_pop[:config["elite_count"]]]  # Elitism
        
        mut_rate = adaptive_mutation_rate(config["mutation_rate"], gens_no_improve)
        
        while len(next_gen) < config["population_size"]:
            p1 = tournament_selection(population, fitness_scores, config["tournament_size"])
            p2 = tournament_selection(population, fitness_scores, config["tournament_size"])
            c1, c2 = maybe_crossover(p1, p2, eligible_nodes)
            c1 = mutate(c1, eligible_nodes, mut_rate)
            c2 = mutate(c2, eligible_nodes, mut_rate)
            next_gen.append(c1)
            if len(next_gen) < config["population_size"]:
                next_gen.append(c2)
        
        population = next_gen
    
    # 4. Apply result
    for node in best_chromosome:
        node.ambulance_here = True
    
    print(f"\n[CHALLENGE 3] GA Final Result:")
    print(f"  Worst-case response distance: {best_fitness:.4f}")
    print(f"  Ambulance positions: {[f'({n.row},{n.col})' for n in best_chromosome]}")
    
    return best_chromosome, best_fitness
```

---

### 1.13 Developer Checklist — Challenge 3

**State Representation:**
- [ ] Chromosome is a list of exactly 3 `CityNode` objects
- [ ] Chromosome only contains nodes where `is_accessible == True`
- [ ] Chromosome only contains Depot or Hospital nodes (or justified alternative)
- [ ] No duplicate nodes in any chromosome

**Fitness Function:**
- [ ] Uses Multi-Source Dijkstra for performance
- [ ] Applies `effective_weight = base_weight × neighbor.risk_index`
- [ ] Returns `float('inf')` for unreachable citizens gracefully
- [ ] Correctly computes `max(min(dist))` — NOT `min(max)` or `sum`

**GA Operations:**
- [ ] Tournament selection with configurable `tournament_size`
- [ ] Crossover with duplicate repair via `repair_chromosome()`
- [ ] Mutation swaps to nodes NOT already in chromosome
- [ ] Elitism preserves top 2 chromosomes

**Dynamic Re-trigger:**
- [ ] `AmbulancePlacer.notify_road_flooded()` is hooked into Challenge 4's flood event
- [ ] `AmbulancePlacer.notify_risk_updated()` is hooked into Challenge 5's model update
- [ ] Old `ambulance_here` flags are cleared before re-running GA

**Performance:**
- [ ] Fitness cache invalidated when `_graph_version` increments
- [ ] `citizen_nodes` list is cached and only refreshed on accessibility changes
- [ ] GA terminates early on stagnation (no infinite loops)

**Integration:**
- [ ] After GA completes, `node.ambulance_here = True` on the 3 selected nodes
- [ ] Event log entry written: `f"[Step {step}] Ambulance GA placed units at {positions}"`

---

## Part 2 — Challenge 5: Crime Risk Prediction & ML Integration

---

### 2.1 Pipeline Overview

Challenge 5 is a **4-phase machine learning pipeline** that flows from raw graph data to live risk updates on the shared city graph.

```
Phase A: Unsupervised Clustering (K-Means)
         ↓
         Assigns each node to a "Socio-Economic Profile" cluster
         
Phase B: Synthetic Dataset Generation
         ↓
         Uses cluster labels + features to assign "Ground Truth" risk levels
         Produces: a labeled DataFrame for supervised training
         
Phase C: Supervised Classification (Decision Tree)
         ↓
         Trains on synthetic data
         Predicts: High / Medium / Low risk per node
         Writes: risk_index multiplier onto each node in shared graph
         
Phase D: Dynamic Learning Loop
         ↓
         Monitors real-time emergencies per node
         Detects: when a node's actual behavior contradicts its predicted label
         Re-trains: the Decision Tree on augmented data
         Updates: risk_index on graph → triggers GA re-evaluation
```

---

### 2.2 Phase A: Feature Engineering & K-Means Clustering

#### What Features to Use

The project statement specifies: *"Group the city's neighborhoods into clusters based on their population density and industrial proximity."*

| Feature | Source | Normalization | Meaning |
|---|---|---|---|
| `population_density` | `node.population_density` | Min-Max scale to [0, 1] | How many people occupy this location |
| `industrial_proximity` | Computed from graph | Min-Max scale to [0, 1] | How close this node is to an Industrial zone |

#### Computing Industrial Proximity

This is NOT stored on the node initially — you must compute it from the graph structure.

```python
def compute_industrial_proximity(graph) -> dict:
    """
    For each node, computes its industrial_proximity = inverse of shortest
    path distance to the nearest Industrial zone.
    
    Higher proximity = closer to industry = higher value.
    
    Returns:
        dict mapping node -> industrial_proximity (float in [0, 1] before normalization)
    """
    # Find all industrial nodes
    industrial_nodes = [
        node for row in graph.grid for node in row
        if node.location_type == "Industrial"
    ]
    
    if not industrial_nodes:
        # Fallback: all nodes get proximity 0
        return {node: 0.0 for row in graph.grid for node in row}
    
    # Multi-source Dijkstra from all industrial nodes simultaneously
    # Result: dist[node] = shortest distance to nearest industrial zone
    dist_to_industry = multi_source_dijkstra(graph, industrial_nodes)
    
    # Convert distance to proximity (closer = higher value)
    # Handle unreachable nodes (dist = infinity)
    max_dist = max(
        (d for d in dist_to_industry.values() if d != float('inf')),
        default=1.0
    )
    
    proximity = {}
    for row in graph.grid:
        for node in row:
            d = dist_to_industry.get(node, float('inf'))
            if d == float('inf'):
                proximity[node] = 0.0  # Unreachable — no industrial influence
            else:
                proximity[node] = 1.0 - (d / max_dist)  # Invert: closer = higher
    
    return proximity
```

#### Feature Normalization

```python
import numpy as np

def extract_and_normalize_features(graph) -> tuple:
    """
    Extracts population_density and industrial_proximity for every node,
    normalizes both features to [0, 1] using Min-Max scaling.
    
    Returns:
        nodes_list: list of CityNode objects (preserves ordering)
        X_normalized: numpy array of shape (n_nodes, 2)
    """
    nodes_list = [node for row in graph.grid for node in row]
    
    # Compute industrial proximity for all nodes
    proximity_map = compute_industrial_proximity(graph)
    
    # Build raw feature matrix
    raw_features = np.array([
        [node.population_density, proximity_map[node]]
        for node in nodes_list
    ], dtype=float)
    
    # Min-Max normalization: (x - min) / (max - min)
    feature_min = raw_features.min(axis=0)
    feature_max = raw_features.max(axis=0)
    
    # Avoid division by zero if all values are the same
    feature_range = feature_max - feature_min
    feature_range[feature_range == 0] = 1.0
    
    X_normalized = (raw_features - feature_min) / feature_range
    
    return nodes_list, X_normalized, raw_features
```

#### K-Means Clustering

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_kmeans_clustering(X_normalized: np.ndarray, k_range=(2, 6)) -> tuple:
    """
    Runs K-Means clustering and selects the best k using Silhouette Score.
    
    K-Means minimizes within-cluster variance (inertia).
    We test k from 2 to 5 and pick k with highest silhouette score.
    
    Returns:
        best_k: int
        cluster_labels: numpy array of shape (n_nodes,) with cluster IDs
        kmeans_model: fitted KMeans object
    """
    best_k = k_range[0]
    best_score = -1
    best_model = None
    
    for k in range(k_range[0], k_range[1]):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_normalized)
        
        if len(set(labels)) < 2:
            continue  # Silhouette requires at least 2 clusters
        
        score = silhouette_score(X_normalized, labels)
        print(f"  K={k}: Silhouette Score = {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
            best_model = model
    
    cluster_labels = best_model.labels_
    print(f"  [K-Means] Best k = {best_k} (Silhouette = {best_score:.4f})")
    
    return best_k, cluster_labels, best_model
```

#### Interpreting Clusters into Socio-Economic Profiles

After K-Means runs, you must interpret each cluster. This is done by examining the **centroid** of each cluster in the original (unnormalized) feature space.

```python
def interpret_clusters(best_model, raw_features: np.ndarray, best_k: int) -> dict:
    """
    Assigns a human-readable socio-economic profile to each cluster.
    
    Logic based on Phase 1 design (Muntazir's real-world observation):
    - High population + High industrial proximity → "Urban Industrial" → High Risk potential
    - Low population + High industrial proximity → "Industrial Fringe" → Medium Risk potential
    - High population + Low industrial proximity → "Residential Core" → Low Risk potential
    - Low population + Low industrial proximity → "Rural/Peripheral" → Low Risk potential
    
    Returns:
        cluster_profiles: dict mapping cluster_id -> profile_name
    """
    # Get centroids in ORIGINAL (unnormalized) feature space
    # We need to inverse-transform — but since we did Min-Max, we stored feature_min/max
    # For simplicity, use the normalized centroids and threshold at 0.5
    centroids = best_model.cluster_centers_  # shape: (k, 2) in normalized space
    
    cluster_profiles = {}
    
    for cluster_id in range(best_k):
        pop_density_normalized = centroids[cluster_id][0]
        industrial_proximity_normalized = centroids[cluster_id][1]
        
        if industrial_proximity_normalized >= 0.5:
            if pop_density_normalized >= 0.5:
                profile = "Urban Industrial"    # High pop + Near industry
            else:
                profile = "Industrial Fringe"   # Low pop + Near industry
        else:
            if pop_density_normalized >= 0.5:
                profile = "Residential Core"    # High pop + Far from industry
            else:
                profile = "Rural Peripheral"    # Low pop + Far from industry
        
        cluster_profiles[cluster_id] = profile
        print(f"  Cluster {cluster_id}: {profile} "
              f"(pop={pop_density_normalized:.2f}, prox={industrial_proximity_normalized:.2f})")
    
    return cluster_profiles
```

---

### 2.3 Phase B: Synthetic Dataset Generation

This phase produces the labeled training data for the Decision Tree.

#### Ground Truth Risk Assignment Logic

This is your **justifiable** logic from Phase 1:

> "Areas close to industrial zones tend to have higher crime rates due to economic pressure, transient populations, and lower policing levels." — Muntazir Mehdi, Phase 1 Design Document

```python
def assign_ground_truth_risk(nodes_list, X_normalized, cluster_labels, cluster_profiles) -> list:
    """
    Assigns a Ground Truth risk label (High/Medium/Low) to each node
    based on its cluster profile and raw feature values.
    
    The logic is:
    
    PRIMARY RULE (Industrial Proximity):
    - industrial_proximity >= 0.7 AND population_density >= 0.4  → HIGH
    - industrial_proximity >= 0.5 OR population_density >= 0.7   → MEDIUM
    - Otherwise                                                   → LOW
    
    SECONDARY RULE (Location Type Modifier):
    - Industrial nodes themselves are bumped UP one level (Low→Medium, Medium→High)
    - School nodes near industry are bumped UP one level
    - Hospital nodes are kept LOW or MEDIUM (emergency services, better policing)
    
    Returns:
        dataset: list of dicts, each containing:
            {population_density, industrial_proximity, location_type_encoded,
             cluster_id, risk_label}
    """
    dataset = []
    
    location_type_encoding = {
        "Residential": 0,
        "Hospital": 1,
        "School": 2,
        "Industrial": 3,
        "Power Plant": 4,
        "Ambulance Depot": 5
    }
    
    for i, node in enumerate(nodes_list):
        pop = X_normalized[i][0]          # Normalized population density
        ind_prox = X_normalized[i][1]     # Normalized industrial proximity
        cluster_id = cluster_labels[i]
        profile = cluster_profiles[cluster_id]
        loc_type = node.location_type
        
        # --- PRIMARY RISK ASSIGNMENT ---
        if ind_prox >= 0.7 and pop >= 0.4:
            risk = "High"
        elif ind_prox >= 0.5 or pop >= 0.7:
            risk = "Medium"
        else:
            risk = "Low"
        
        # --- SECONDARY MODIFIER (Location Type) ---
        risk_scale = ["Low", "Medium", "High"]
        risk_idx = risk_scale.index(risk)
        
        if loc_type in ("Industrial", "Power Plant"):
            risk_idx = min(risk_idx + 1, 2)   # Bump up
        elif loc_type == "School" and ind_prox >= 0.5:
            risk_idx = min(risk_idx + 1, 2)   # Bump up if near industry
        elif loc_type in ("Hospital", "Ambulance Depot"):
            risk_idx = max(risk_idx - 1, 0)   # Bump down (better policing)
        
        risk = risk_scale[risk_idx]
        
        # --- ADD NOISE FOR REALISM ---
        # 10% chance of randomly flipping to adjacent risk level
        # This prevents the Decision Tree from perfectly overfitting synthetic data
        if random.random() < 0.10:
            flip = random.choice([-1, 1])
            risk_idx = max(0, min(2, risk_idx + flip))
            risk = risk_scale[risk_idx]
        
        dataset.append({
            "population_density": pop,
            "industrial_proximity": ind_prox,
            "location_type_encoded": location_type_encoding.get(loc_type, 0),
            "cluster_id": cluster_id,
            "risk_label": risk
        })
    
    return dataset
```

#### Building the Training DataFrame

```python
import pandas as pd

def build_training_dataframe(dataset: list) -> pd.DataFrame:
    """
    Converts the synthetic dataset list into a pandas DataFrame
    ready for scikit-learn training.
    """
    df = pd.DataFrame(dataset)
    
    # Verify label distribution
    print("\n[Challenge 5] Synthetic Dataset Label Distribution:")
    print(df["risk_label"].value_counts())
    print(f"Total samples: {len(df)}")
    
    return df
```

---

### 2.4 Phase C: Supervised Classification — Decision Tree

#### Why Decision Tree over KNN

| Property | Decision Tree | K-Nearest Neighbors |
|---|---|---|
| **Interpretability** | Produces explicit IF-THEN rules | Black-box distance comparisons |
| **Viva explainability** | Can trace exactly why any node got "High" | Cannot explain individual predictions |
| **Speed at prediction** | O(log n) — follows tree path | O(n×k) — computes k distances each time |
| **Feature importance** | Built-in (Gini importance scores) | Not available |
| **Handling noise** | Pruning controls overfit | Sensitive to noisy neighbors |
| **New node prediction** | Follows same learned rules | Requires all training data in memory |

**Viva Answer:** "We chose Decision Trees because the project evaluator will ask us *why* a specific zone is High Risk. With a Decision Tree, we can literally show the path through the tree: 'If industrial_proximity > 0.5 AND population_density > 0.4 → High Risk.' With KNN, we can only say 'its 5 nearest neighbors were High Risk,' which is a much weaker justification."

#### Model Training

```python
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

class CrimeRiskClassifier:
    """
    Manages the Decision Tree classifier for Challenge 5.
    Handles initial training, prediction, and re-training.
    """
    
    FEATURES = ["population_density", "industrial_proximity", 
                 "location_type_encoded", "cluster_id"]
    TARGET = "risk_label"
    
    # Risk level to multiplier mapping
    RISK_MULTIPLIERS = {
        "High":   1.5,
        "Medium": 1.2,
        "Low":    1.0
    }
    
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.training_df = None
        self.version = 0  # Incremented on each re-train
    
    def train(self, df: pd.DataFrame):
        """
        Trains the Decision Tree on the provided DataFrame.
        
        Hyperparameters:
        - max_depth=5: Prevents overfitting on small synthetic dataset
        - min_samples_split=4: Requires at least 4 samples to split a node
        - criterion='gini': Standard Gini impurity (faster than entropy)
        - class_weight='balanced': Handles imbalanced High/Medium/Low distribution
        """
        self.training_df = df.copy()
        
        X = df[self.FEATURES].values
        y_raw = df[self.TARGET].values
        y = self.label_encoder.fit_transform(y_raw)  # Encode labels to integers
        
        # Train/test split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize model with chosen hyperparameters
        self.model = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=4,
            criterion='gini',
            class_weight='balanced',
            random_state=42
        )
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n[Challenge 5] Decision Tree Training Complete:")
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  Classes: {self.label_encoder.classes_}")
        
        # Print classification report
        print(classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_
        ))
        
        # Print tree structure for viva
        tree_rules = export_text(self.model, feature_names=self.FEATURES)
        print("\n[Decision Tree Rules]:")
        print(tree_rules[:2000])  # Print first 2000 chars of rules
        
        self.version += 1
    
    def predict_all_nodes(self, nodes_list, X_normalized, cluster_labels) -> dict:
        """
        Runs the trained classifier on all nodes.
        
        Returns:
            predictions: dict mapping node -> risk_label ("High"/"Medium"/"Low")
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        location_type_encoding = {
            "Residential": 0, "Hospital": 1, "School": 2,
            "Industrial": 3, "Power Plant": 4, "Ambulance Depot": 5
        }
        
        predictions = {}
        
        for i, node in enumerate(nodes_list):
            features = [
                X_normalized[i][0],   # population_density (normalized)
                X_normalized[i][1],   # industrial_proximity (normalized)
                location_type_encoding.get(node.location_type, 0),
                int(cluster_labels[i])
            ]
            
            encoded_pred = self.model.predict([features])[0]
            risk_label = self.label_encoder.inverse_transform([encoded_pred])[0]
            predictions[node] = risk_label
        
        return predictions
    
    def apply_predictions_to_graph(self, predictions: dict):
        """
        Writes risk_index multipliers onto each node in the shared graph.
        THIS IS YOUR WRITE OPERATION on the shared graph.
        
        After this call, all other modules (Challenge 3 GA, Challenge 4 router)
        will automatically use the updated weights.
        """
        for node, risk_label in predictions.items():
            old_risk = node.risk_index
            node.risk_index = self.RISK_MULTIPLIERS[risk_label]
            
            if old_risk != node.risk_index:
                print(f"  [Risk Update] ({node.row},{node.col}): "
                      f"{old_risk} → {node.risk_index} ({risk_label})")
        
        print(f"[Challenge 5] Risk indices updated on {len(predictions)} nodes.")
        
        # Increment graph version to invalidate GA fitness cache
        global _graph_version
        _graph_version += 1
```

#### Hyperparameter Justification Table

| Hyperparameter | Value | Why |
|---|---|---|
| `max_depth` | 5 | Synthetic dataset has ~400 nodes. Depth 5 allows up to 32 leaf nodes — enough granularity without overfitting |
| `min_samples_split` | 4 | Prevents creating splits based on 1-2 noisy samples from our 10% noise injection |
| `criterion` | `'gini'` | Gini impurity is computationally faster than entropy; equivalent performance on 3-class problem |
| `class_weight` | `'balanced'` | Compensates if High/Medium/Low labels are not equally distributed in synthetic data |
| `random_state` | 42 | Reproducibility — same tree every run (important for viva demos) |

---

### 2.5 Phase D: The Dynamic Learning Loop

This is the most advanced feature of Challenge 5 — and the one that earns marks in the "Ambitious" criterion. It transforms your system from a static predictor into a **living intelligence** that improves as the simulation runs.

#### Concept: "Concept Drift" in Risk Prediction

**Problem:** Your Decision Tree was trained on synthetic data. The synthetic data is based on *assumptions* (industrial proximity → crime). But during the simulation, real events happen. If a zone predicted as "Low Risk" keeps triggering emergencies, the model was wrong about that zone.

**Solution:** The Dynamic Learning Loop detects this divergence and re-trains the model on augmented data that includes real simulation evidence.

#### The `total_emergencies` Counter

You must add this field to every `CityNode`. It tracks how many emergency events have been dispatched to or near this node during the simulation.

```python
# In the CityNode class (coordinate with Ammar who owns graph structure):
class CityNode:
    def __init__(self, row, col, location_type, population_density):
        # ... existing fields ...
        self.total_emergencies = 0    # YOU ADD THIS
        self.predicted_risk = "Low"   # YOU ADD THIS (updated by classifier)
```

#### CrimeStatsManager — Tracking Real-Time Evidence

```python
class CrimeStatsManager:
    """
    Monitors real-time simulation events and detects when the ML model's
    predictions diverge from actual behavior.
    
    This class is owned by Challenge 5 (Muntazir).
    Other modules call notify_emergency() when events occur.
    """
    
    # Thresholds for drift detection
    DRIFT_WINDOW = 5          # Check for drift every 5 simulation steps
    LOW_RISK_EMERGENCY_THRESHOLD = 3   # If "Low Risk" node gets 3+ emergencies → drift
    MEDIUM_RISK_EMERGENCY_THRESHOLD = 6 # If "Medium Risk" gets 6+ → drift
    
    def __init__(self, graph, classifier: CrimeRiskClassifier):
        self.graph = graph
        self.classifier = classifier
        self.simulation_step = 0
        self.drift_detected_nodes = []
    
    def notify_emergency(self, node):
        """
        Called by Challenge 4's router whenever it dispatches to a civilian.
        Increments the emergency counter on the affected node AND its neighbors.
        """
        node.total_emergencies += 1
        
        # Also slightly increment neighbors (spillover effect)
        for neighbor in node.neighbors:
            neighbor.total_emergencies += 0.5
    
    def step(self):
        """
        Called at each simulation step. Checks for drift and re-trains if needed.
        """
        self.simulation_step += 1
        
        if self.simulation_step % self.DRIFT_WINDOW == 0:
            self._check_for_drift()
    
    def _check_for_drift(self):
        """
        Scans all nodes. Identifies nodes where actual emergency frequency
        contradicts the predicted risk label.
        
        Drift condition:
        - A "Low Risk" node has total_emergencies >= LOW_RISK_EMERGENCY_THRESHOLD
        - A "Medium Risk" node has total_emergencies >= MEDIUM_RISK_EMERGENCY_THRESHOLD
        """
        drifted = []
        
        for row in self.graph.grid:
            for node in row:
                pred = node.predicted_risk
                actual = node.total_emergencies
                
                drift = False
                if pred == "Low" and actual >= self.LOW_RISK_EMERGENCY_THRESHOLD:
                    drift = True
                    print(f"  [DRIFT] ({node.row},{node.col}): Predicted LOW but "
                          f"{actual} emergencies → upgrading to MEDIUM/HIGH")
                
                elif pred == "Medium" and actual >= self.MEDIUM_RISK_EMERGENCY_THRESHOLD:
                    drift = True
                    print(f"  [DRIFT] ({node.row},{node.col}): Predicted MEDIUM but "
                          f"{actual} emergencies → upgrading to HIGH")
                
                if drift:
                    drifted.append(node)
        
        if drifted:
            self.drift_detected_nodes = drifted
            self._retrain_with_corrections(drifted)
    
    def _retrain_with_corrections(self, drifted_nodes: list):
        """
        Re-trains the Decision Tree by augmenting the original training data
        with corrected labels for drifted nodes.
        
        Strategy:
        1. Take original training DataFrame
        2. Find rows corresponding to drifted nodes
        3. Correct their labels based on emergency count evidence
        4. Re-train the model on augmented data
        5. Re-apply predictions to the graph
        6. Trigger GA re-evaluation
        """
        print(f"\n[Challenge 5] Re-training triggered for {len(drifted_nodes)} drifted nodes")
        
        # Step 1: Build correction records
        correction_records = []
        
        for node in drifted_nodes:
            # Determine corrected label based on emergency count
            emergencies = node.total_emergencies
            if emergencies >= self.MEDIUM_RISK_EMERGENCY_THRESHOLD:
                corrected_label = "High"
            elif emergencies >= self.LOW_RISK_EMERGENCY_THRESHOLD:
                corrected_label = "Medium"
            else:
                corrected_label = node.predicted_risk  # No change needed
            
            # Create a new training record for this node
            # (features same as before, label corrected)
            record = self._node_to_feature_record(node, corrected_label)
            correction_records.append(record)
        
        # Step 2: Augment training data
        correction_df = pd.DataFrame(correction_records)
        augmented_df = pd.concat([
            self.classifier.training_df,
            correction_df
        ], ignore_index=True)
        
        # Add multiple copies of corrections to give them more weight
        weight_factor = 3
        for _ in range(weight_factor - 1):
            augmented_df = pd.concat([augmented_df, correction_df], ignore_index=True)
        
        print(f"  Original training size: {len(self.classifier.training_df)}")
        print(f"  Augmented training size: {len(augmented_df)}")
        
        # Step 3: Re-train
        self.classifier.train(augmented_df)
        
        # Step 4: Re-predict all nodes
        # (need nodes_list, X_normalized, cluster_labels — store these at pipeline init)
        predictions = self.classifier.predict_all_nodes(
            self.nodes_list, self.X_normalized, self.cluster_labels
        )
        
        # Step 5: Apply to graph
        self.classifier.apply_predictions_to_graph(predictions)
        
        # Step 6: Update predicted_risk field on each node
        for node, label in predictions.items():
            node.predicted_risk = label
        
        # Step 7: Log event
        print(f"  [Step {self.simulation_step}] Risk model re-trained. "
              f"{len(drifted_nodes)} nodes corrected.")
        print(f"  Drifted nodes: {[(n.row, n.col) for n in drifted_nodes]}")
    
    def _node_to_feature_record(self, node, corrected_label: str) -> dict:
        """Converts a node to a training record with corrected label."""
        location_type_encoding = {
            "Residential": 0, "Hospital": 1, "School": 2,
            "Industrial": 3, "Power Plant": 4, "Ambulance Depot": 5
        }
        node_idx = self.nodes_list.index(node)
        
        return {
            "population_density": self.X_normalized[node_idx][0],
            "industrial_proximity": self.X_normalized[node_idx][1],
            "location_type_encoded": location_type_encoding.get(node.location_type, 0),
            "cluster_id": int(self.cluster_labels[node_idx]),
            "risk_label": corrected_label
        }
    
    def set_pipeline_data(self, nodes_list, X_normalized, cluster_labels):
        """Called once after Phase A to store pipeline data for re-training."""
        self.nodes_list = nodes_list
        self.X_normalized = X_normalized
        self.cluster_labels = cluster_labels
```

#### The "Muntazir Learning Model" — Summary Diagram

```
Initial State:
  Node (5,3): location_type=Residential, pop=0.8, ind_prox=0.75
  K-Means → Cluster 0 ("Urban Industrial")
  Synthetic label → "High"
  Decision Tree trains on this → predicts "High"
  risk_index = 1.5
  
Simulation Step 1-4:
  No emergencies at (5,3)
  total_emergencies = 0
  No drift detected
  
Simulation Step 5 (Drift Check):
  Still 0 emergencies → no drift
  
... but suppose at Step 3, another node (7,1) predicted "Low" gets 4 emergencies ...

Drift Check at Step 5:
  (7,1): predicted="Low", total_emergencies=4 >= threshold(3)
  → DRIFT DETECTED
  
Re-training:
  Add record: (7,1) features + corrected_label="Medium" (×3 copies for weight)
  Re-train Decision Tree on augmented data
  New prediction for (7,1) = "Medium"
  (7,1).risk_index updated: 1.0 → 1.2
  
Side Effect:
  _graph_version incremented
  Challenge 3 GA fitness cache invalidated
  Next GA run uses updated edge weights through (7,1)
  Challenge 4 router paths through (7,1) now cost 20% more
```

---

### 2.6 Developer Checklist — Challenge 5

**Phase A — K-Means:**
- [ ] `compute_industrial_proximity()` uses Multi-Source Dijkstra (not Manhattan distance)
- [ ] Both features normalized to [0, 1] using Min-Max scaling
- [ ] Silhouette Score used to select best k (not just hardcoded k=3)
- [ ] `cluster_profiles` dict maps cluster_id → profile name (for viva)

**Phase B — Synthetic Dataset:**
- [ ] Risk assignment logic matches Phase 1 design (industrial proximity primary factor)
- [ ] Location type modifier applied (Industrial bumps up, Hospital bumps down)
- [ ] 10% noise injection prevents perfect fit
- [ ] Label distribution printed and checked for reasonable balance

**Phase C — Decision Tree:**
- [ ] `max_depth=5` to prevent overfitting
- [ ] `class_weight='balanced'` for imbalanced classes
- [ ] Classification report printed (precision/recall/F1 per class)
- [ ] Tree rules exported with `export_text()` for viva
- [ ] `apply_predictions_to_graph()` writes to `node.risk_index` AND `node.predicted_risk`
- [ ] `_graph_version` incremented after every update

**Phase D — Dynamic Loop:**
- [ ] `notify_emergency()` called from Challenge 4's event handler
- [ ] `step()` called at each simulation step from the main loop
- [ ] Drift detection thresholds are configurable constants
- [ ] Re-training uses augmented (not replaced) training data
- [ ] Drifted nodes get 3× weight in augmented dataset
- [ ] Post-retrain: `_graph_version` incremented to invalidate GA cache
- [ ] Event log entry written for every re-train event

---

## Part 3 — System Integration & Shared Graph

---

### 3.1 The Risk Multiplier Logic

Your ML output feeds directly into the edge weight calculations used by both your GA (Challenge 3) and the router (Challenge 4).

#### Exact Multiplier Mapping

| Predicted Risk Level | `node.risk_index` Value | Effect on Edge Weights |
|---|---|---|
| `"Low"` | `1.0` | No cost change. Default state after Challenge 2 builds roads. |
| `"Medium"` | `1.2` | 20% cost increase. Edges entering this node are 20% more expensive. |
| `"High"` | `1.5` | 50% cost increase. Edges entering this node are 50% more expensive. |

#### Application Formula

```
effective_weight(u → v) = graph.get_edge_weight(u, v) × v.risk_index
```

Where `graph.get_edge_weight(u, v)` = `u.neighbors[v]` = the base weight set by Challenge 2 (Kruskal's MST). Standard roads = 1.0, residential roads = 0.8.

**Examples:**
```
Standard road to Low Risk node:    1.0 × 1.0 = 1.0
Standard road to Medium Risk node: 1.0 × 1.2 = 1.2
Standard road to High Risk node:   1.0 × 1.5 = 1.5
Residential road to High Risk node: 0.8 × 1.5 = 1.2
```

#### Default Risk Index

All nodes start with `risk_index = 1.0`. Challenge 5 runs before the simulation begins and updates these values. If Challenge 5 has not run yet (e.g., during testing of Challenge 3 in isolation), the GA still works correctly — all edge weights remain at base values.

---

### 3.2 Synchronization Protocol

```
WHEN Challenge 5 updates risk_index on any node:
    1. node.risk_index = new_value                    (Challenge 5 writes)
    2. node.predicted_risk = new_label                (Challenge 5 writes)
    3. _graph_version += 1                            (Challenge 5 signals)
    4. → GA fitness cache automatically invalid       (Challenge 3 detects)
    5. → AmbulancePlacer.notify_risk_updated() called (Challenge 5 calls)
    6. → If avg risk delta > threshold: GA re-runs    (Challenge 3 responds)
    7. → D* Lite router uses new weights on next path (Challenge 4 reads)
       query (no explicit notification needed —
       router always reads live edge weights)
```

**No Message Queue Needed.** Because all modules share the same `CityNode` objects (Python references), writing to `node.risk_index` is instantly visible to any code that reads that field. There is no "broadcast" mechanism required.

**The Only Explicit Notification:** Challenge 5 must call `ambulance_placer.notify_risk_updated()` after updating the graph. Everything else is implicit through the shared object references.

---

### 3.3 Data Structures: CityNode & CrimeStatsManager

#### Complete CityNode Class Definition

```python
class CityNode:
    """
    Represents a single location in the CityMind grid.
    This is the shared state object used by ALL five challenges.
    
    FIELD OWNERSHIP:
    - row, col, location_type, population_density: Challenge 1 (layout)
    - neighbors: Challenge 2 (road network)
    - is_accessible: Challenge 4 (router) — also read by C3
    - risk_index, predicted_risk, total_emergencies: Challenge 5 (YOU)
    - ambulance_here: Challenge 3 (YOU)
    """
    
    def __init__(self, row: int, col: int, location_type: str, population_density: float):
        # --- SPATIAL IDENTITY (set by Challenge 1, never changes) ---
        self.row = row
        self.col = col
        self.location_type = location_type    # "Residential", "Hospital", etc.
        self.population_density = population_density  # Float: number of occupants
        
        # --- GRAPH CONNECTIVITY (set by Challenge 2) ---
        # neighbors dict: {CityNode_object: base_edge_weight}
        # Example: {node_right: 1.0, node_below: 0.8}
        self.neighbors = {}
        
        # --- DYNAMIC STATE (Challenge 4 writes is_accessible) ---
        self.is_accessible = True  # Set to False when road events isolate this node
        
        # --- CHALLENGE 5 FIELDS (Muntazir writes these) ---
        self.risk_index = 1.0           # Multiplier: 1.0 (Low), 1.2 (Med), 1.5 (High)
        self.predicted_risk = "Low"     # String label: "Low"/"Medium"/"High"
        self.total_emergencies = 0      # Cumulative emergency events at/near this node
        
        # --- CHALLENGE 3 FIELD (Muntazir writes this) ---
        self.ambulance_here = False     # True if current GA solution places ambulance here
    
    def get_neighbor_weight(self, neighbor) -> float:
        """
        Returns the EFFECTIVE (risk-adjusted) edge weight to a neighbor.
        This is the formula that both C3 and C4 must use.
        """
        if neighbor not in self.neighbors:
            return float('inf')
        base_weight = self.neighbors[neighbor]
        return base_weight * neighbor.risk_index
    
    def block_road_to(self, neighbor):
        """
        Removes a neighbor connection (simulates flooding).
        Called by Challenge 4's event system.
        O(1) operation — immediately visible to all modules.
        """
        if neighbor in self.neighbors:
            del self.neighbors[neighbor]
        if self in neighbor.neighbors:
            del neighbor.neighbors[self]
    
    def __repr__(self):
        return (f"CityNode({self.row},{self.col}|{self.location_type[:3]}|"
                f"risk={self.risk_index}|emerg={self.total_emergencies})")
    
    def __hash__(self):
        return hash((self.row, self.col))
    
    def __eq__(self, other):
        return isinstance(other, CityNode) and self.row == other.row and self.col == other.col
```

#### Complete CrimeStatsManager Class Summary

```python
class CrimeStatsManager:
    """
    Owned by: Challenge 5 (Muntazir)
    Purpose:  Monitors simulation events and triggers ML model updates.
    
    Public Interface (called by other challenges):
        notify_emergency(node)  ← Challenge 4 calls this
        step()                  ← Main simulation loop calls this
    
    Internal:
        _check_for_drift()
        _retrain_with_corrections(drifted_nodes)
        _node_to_feature_record(node, label)
    
    See full implementation in §2.5 above.
    """
    pass  # Full implementation defined in §2.5
```

---

### 3.4 Integration Flowchart (Textual)

```
SIMULATION STARTUP:
══════════════════
[C1] Build city layout → place nodes with types + population densities
[C2] Build road network → set neighbor dicts with base edge weights
[C5-A] Run K-Means → assign cluster_id to each node
[C5-B] Generate synthetic dataset → assign ground truth labels
[C5-C] Train Decision Tree → predict risk for all nodes
[C5] Apply risk_index to graph → update multipliers
[C3] Run GA → place 3 ambulances (using risk-weighted distances)
     → ambulance_here = True on 3 nodes
     → best_fitness logged

SIMULATION LOOP (Steps 1–20):
══════════════════════════════
Each step:

  [MAIN] Generate random event (flood or emergency call)
  
  IF flood event:
    [C4] block_road_to(u, v) on affected nodes
    [C4] D* Lite replans route immediately
    [C5] CrimeStatsManager is NOT notified (floods ≠ crimes)
    [C3] AmbulancePlacer.notify_road_flooded() called
         IF flood_count >= 3: GA re-runs
  
  IF emergency call at civilian node:
    [C4] D* Lite routes medical team to civilian
    [C5] CrimeStatsManager.notify_emergency(civilian_node) called
         → total_emergencies on node incremented
  
  [C5] CrimeStatsManager.step() called
       IF step % DRIFT_WINDOW == 0:
         IF drift detected:
           → Re-train Decision Tree
           → Update risk_index on graph
           → _graph_version incremented
           → AmbulancePlacer.notify_risk_updated() called
             IF avg risk delta > 0.15: GA re-runs
  
  [UI] Render updated grid (risk heatmap, ambulance coverage, event log)

END SIMULATION → Print final placement + worst-case distance
```

---

### 3.5 Developer Checklist — Integration

- [ ] `CityNode` has all fields listed in §3.3 (coordinate with Ammar before coding)
- [ ] `_graph_version` global is incremented by Challenge 5 after every risk update
- [ ] `AmbulancePlacer.notify_risk_updated()` is called from `apply_predictions_to_graph()`
- [ ] `CrimeStatsManager.notify_emergency()` is called from Challenge 4's routing function
- [ ] `CrimeStatsManager.step()` is called from the main simulation loop
- [ ] Event log entries written for: GA placement, GA re-trigger, risk model re-train, drift detection

---

## Part 4 — Viva Defense & Live Modification Prep

---

### 4.1 "Why" Justification Bank

Memorize these. The viva questions WILL come from this list.

---

**Q: Why Genetic Algorithm for ambulance placement instead of Brute Force?**

> "The search space for placing 3 ambulances across N eligible positions is C(N,3). For a 20×20 grid with 220 eligible positions, this is 1.75 million combinations. Each requires running Multi-Source Dijkstra — at O((V+E) log V), that's over 1.75 million graph traversals. A GA with 100 chromosomes and 200 generations evaluates at most 20,000 configurations, guided by crossover and mutation to concentrate search in high-quality regions. The result is not guaranteed optimal, but it is near-optimal in a fraction of the time."

---

**Q: Why Minimax (worst-case) instead of average-case fitness?**

> "Minimizing the average distance from citizens to ambulances sounds good mathematically, but it allows the system to create a configuration where 95% of citizens are well-covered but one remote neighborhood has no ambulance within reach at all. In an emergency system, that one uncovered citizen could die. Minimax guarantees that the WORST served citizen is as well-served as possible — it is the equity-focused objective, not just the efficiency-focused one."

---

**Q: Why Tournament Selection over Roulette Wheel?**

> "Roulette Wheel Selection assigns selection probability proportional to fitness. If one chromosome has fitness 0.5 and another has fitness 500, the worse one gets almost zero probability. Over time, all chromosomes converge to the best individual — diversity collapses. Tournament Selection picks the best among k random candidates, which maintains selection pressure without eliminating diversity. It is also computationally simpler and more robust when fitness values vary widely, as they do when risk weights change."

---

**Q: Why K-Means first (unsupervised) before the Decision Tree (supervised)?**

> "The project explicitly requires clustering 'without using pre-labeled data.' We have no historical crime records — we are building a synthetic city. K-Means discovers natural groupings based on population density and industrial proximity, giving us interpretable socio-economic profiles. These profiles then inform our synthetic label generation in Phase B, which produces the training data for the Decision Tree. The unsupervised step is not optional — it is how we generate labeled data in the absence of real crime records."

---

**Q: Why Decision Tree over KNN?**

> "The evaluation explicitly asks us to justify predictions during the viva. A Decision Tree generates explicit IF-THEN rules that I can trace on a whiteboard: 'IF industrial_proximity > 0.5 AND population_density > 0.4 → High Risk.' With KNN, the prediction is 'the 5 nearest points in feature space were High Risk' — this is not an explanation, it is a lookup. Decision Trees also have O(log n) prediction time vs O(n×k) for KNN, and they provide feature importance scores that we can use to explain which variable drives risk most strongly."

---

**Q: Why does the risk index update the edge weights instead of just the nodes?**

> "Because travel cost is fundamentally about traversal. When an ambulance or medical team enters a High Risk zone, they encounter obstacles — crowds, police checkpoints, road conditions — that slow them down. Encoding risk as a node property but applying it as an edge weight multiplier when you enter that node correctly models this: the cost is incurred when you arrive at a risky location. This also means the effect is local — a high-risk node only affects edges pointing into it, not all edges in the graph."

---

**Q: What happens if all nodes get predicted as "High Risk"?**

> "All edge weights would be multiplied by 1.5 uniformly. The GA's fitness function would compute distances that are all 50% higher. But the relative differences between placements remain the same — the optimizer still finds the best configuration for the given graph. The absolute fitness value would be higher, but the best placement would still minimize worst-case response time within the high-risk environment. The system degrades gracefully."

---

**Q: How does Challenge 5's re-training affect Challenge 3?**

> "Challenge 5 writes updated risk_index values to the shared CityNode objects. This increments _graph_version, which invalidates the GA's fitness cache. The AmbulancePlacer module is notified via notify_risk_updated(). If the average risk change exceeds the threshold (0.15), the GA re-runs from scratch — re-evaluating ambulance positions against the new cost landscape. The old ambulance_here flags are cleared first, then set on the new best chromosome."

---

### 4.2 Live Modification Scenarios

These are the types of changes the professor may demand during the viva. Know EXACTLY which line of code to change.

---

**Scenario 1: "Change the number of ambulances from 3 to 5."**

Changes needed:
```python
# 1. In GA_CONFIG:
GA_CONFIG["num_ambulances"] = 5  # Was 3

# 2. In initialize_population():
# Change: chromosome = random.sample(eligible_nodes, 3)
# To:     chromosome = random.sample(eligible_nodes, config["num_ambulances"])

# 3. In is_valid_chromosome():
# Change: if len(set(id(n) for n in chromosome)) != 3:
# To:     if len(set(id(n) for n in chromosome)) != NUM_AMBULANCES:

# 4. No changes needed in fitness function — it already iterates over all
#    elements of the chromosome regardless of length.
```

Viva answer: "The GA is parameterized by `num_ambulances`. I change the constant, regenerate the population, and re-run. The fitness function already handles arbitrary chromosome lengths — it runs Multi-Source Dijkstra from all ambulance positions simultaneously."

---

**Scenario 2: "Change High Risk multiplier from 1.5 to 2.0."**

Changes needed:
```python
# In CrimeRiskClassifier:
RISK_MULTIPLIERS = {
    "High":   2.0,   # Was 1.5
    "Medium": 1.2,
    "Low":    1.0
}
# Then call apply_predictions_to_graph() again to refresh all node.risk_index values
```

Viva answer: "The multiplier is a class constant in CrimeRiskClassifier. Change it, then call apply_predictions_to_graph() to push the new values through. _graph_version increments, fitness cache clears, and both the GA and router automatically use the new weights on their next invocations."

---

**Scenario 3: "What if the grid is 30×30 instead of 20×20?"**

Answer: "The code is grid-size agnostic. The GA reads eligible_nodes from the live graph — if the graph grows, it automatically finds more positions to explore. The only parameter I'd increase is population_size (from 100 to 150) and max_generations (from 200 to 300) to maintain search quality over a larger space. K-Means clustering also scales automatically since it operates on feature vectors, not grid coordinates."

---

**Scenario 4: "Add a 4th risk level: Critical (multiplier 2.0)."**

Changes needed:
```python
# 1. CrimeRiskClassifier.RISK_MULTIPLIERS:
RISK_MULTIPLIERS = {
    "Critical": 2.0,  # NEW
    "High":     1.5,
    "Medium":   1.2,
    "Low":      1.0
}

# 2. assign_ground_truth_risk(): Add Critical condition
if ind_prox >= 0.9 and pop >= 0.8 and loc_type == "Industrial":
    risk = "Critical"

# 3. Drift detection thresholds in CrimeStatsManager: add Critical tier
# 4. Re-generate synthetic dataset and re-train
```

---

**Scenario 5: "Remove K-Means clustering. Can you still run the Decision Tree?"**

Answer: "Yes. K-Means produces cluster_id as a feature. If clustering is removed, I drop cluster_id from the FEATURES list in CrimeRiskClassifier. The Decision Tree trains on [population_density, industrial_proximity, location_type_encoded] — 3 features instead of 4. The model quality decreases slightly because cluster context is useful, but the pipeline still runs. I would note this trade-off in the viva."

---

### 4.3 First-Principles Walkthrough Scripts

Use these to answer "Explain this algorithm from scratch."

---

#### Genetic Algorithm Walkthrough

> "Imagine you're trying to find the best position for 3 fire stations in a city. You can't try every combination — there are millions. So you start with 100 random guesses. You test each one: for each guess, you ask 'what's the farthest a citizen would have to travel to reach ANY of these 3 stations?' That's the fitness score. The worse the score, the longer the worst-case journey.
>
> Now you breed the better guesses together. You take the best 20% and pair them up. From each pair, you create two children by mixing their station locations. Sometimes you randomly move one station to a new location — this is mutation. You keep the best 2 guesses unchanged in every generation — this is elitism.
>
> After 200 rounds of this, the population has converged to a near-optimal solution. The best individual is your ambulance placement. This takes maybe 1 second instead of 30 minutes for brute force."

---

#### Decision Tree Walkthrough

> "A Decision Tree is literally a flowchart. At each step, it asks a yes/no question about the input: 'Is industrial proximity greater than 0.5?' If yes, go left. If no, go right. At the end of the path, it tells you the prediction: High, Medium, or Low.
>
> We build this tree by looking at our training data and finding the question that best separates High Risk zones from Low Risk zones. 'Best' means the question that creates the most pure subsets — if I split here, do I end up with mostly High Risk on one side and mostly Low Risk on the other? This is measured by Gini Impurity.
>
> We keep splitting recursively until either we've used up 5 levels (max_depth=5) or a node is pure enough. Then we stop and label each leaf with the majority class of its training examples."

---

#### K-Means Walkthrough

> "K-Means groups the city's neighborhoods into clusters based on shared characteristics. We use two features: population density and industrial proximity.
>
> Start by placing k random 'centers' in this 2D feature space. Assign each neighborhood to its nearest center. Then move each center to the average position of all its assigned neighborhoods. Repeat until the centers stop moving.
>
> The result is k clusters of neighborhoods that are similar to each other. We label these clusters with human-readable names based on where their centers land in feature space: 'Urban Industrial', 'Residential Core', etc. These labels guide our synthetic crime data generation."

---

## Appendix A: Complete Pseudocode Reference

### A.1 Full Challenge 3 Pseudocode

```
ALGORITHM: GA_Ambulance_Placement(graph)

INPUT: graph (shared CityGraph with all nodes and edges)
OUTPUT: best_placement (list of 3 CityNode), best_fitness (float)

CONSTANTS:
    POP_SIZE = 100
    MAX_GEN = 200
    CROSSOVER_RATE = 0.85
    MUTATION_RATE = 0.15
    ELITE_COUNT = 2
    TOURNAMENT_SIZE = 5
    STAGNATION_LIMIT = 30

STEP 1: INITIALIZATION
    eligible_nodes ← [n for n in graph.all_nodes() 
                       if n.is_accessible AND n.location_type IN {"Ambulance Depot", "Hospital"}]
    
    citizen_nodes ← [n for n in graph.all_nodes() 
                      if n.population_density > 0 AND n.is_accessible]
    
    population ← []
    FOR i = 1 to POP_SIZE:
        chromosome ← random_sample(eligible_nodes, 3)   # No duplicates
        population.append(chromosome)
    
    best_chromosome ← None
    best_fitness ← ∞
    gens_stagnant ← 0

STEP 2: GENERATIONAL LOOP
    FOR generation = 1 to MAX_GEN:
    
        # EVALUATE
        fitness_scores ← []
        FOR each chromosome c IN population:
            dist_map ← MULTI_SOURCE_DIJKSTRA(graph, sources=c, use_risk_weights=True)
            f ← max over all cit IN citizen_nodes of dist_map.get(cit, ∞)
            fitness_scores.append(f)
        
        # TRACK BEST
        gen_best_idx ← argmin(fitness_scores)
        IF fitness_scores[gen_best_idx] < best_fitness:
            best_fitness ← fitness_scores[gen_best_idx]
            best_chromosome ← copy(population[gen_best_idx])
            gens_stagnant ← 0
        ELSE:
            gens_stagnant ← gens_stagnant + 1
        
        # EARLY STOP
        IF gens_stagnant >= STAGNATION_LIMIT:
            BREAK
        
        # BUILD NEXT GENERATION
        sorted_pop ← sort population by fitness ascending
        next_gen ← [copy(sorted_pop[0]), copy(sorted_pop[1])]  # Elitism
        
        adaptive_mut ← IF gens_stagnant > 20 THEN min(MUTATION_RATE×2, 0.4) ELSE MUTATION_RATE
        
        WHILE len(next_gen) < POP_SIZE:
            p1 ← TOURNAMENT_SELECT(population, fitness_scores, TOURNAMENT_SIZE)
            p2 ← TOURNAMENT_SELECT(population, fitness_scores, TOURNAMENT_SIZE)
            
            IF random() < CROSSOVER_RATE:
                c1 ← [p1[0]] + p2[1:]   # Single-point crossover
                c2 ← [p2[0]] + p1[1:]
                c1 ← REPAIR(c1, eligible_nodes)
                c2 ← REPAIR(c2, eligible_nodes)
            ELSE:
                c1, c2 ← copy(p1), copy(p2)
            
            IF random() < adaptive_mut:
                idx ← random_int(0, 2)
                available ← eligible_nodes - set(c1)
                c1[idx] ← random_choice(available)
            
            # Same mutation for c2
            
            next_gen.append(c1)
            IF len(next_gen) < POP_SIZE:
                next_gen.append(c2)
        
        population ← next_gen

STEP 3: APPLY RESULT
    FOR each node IN best_chromosome:
        node.ambulance_here ← True
    
    RETURN best_chromosome, best_fitness
```

### A.2 Full Challenge 5 Pseudocode

```
ALGORITHM: CrimeRisk_Pipeline(graph)

INPUT: graph (shared CityGraph)
OUTPUT: risk_index written to each node in graph

PHASE A: CLUSTERING
    prox_map ← MULTI_SOURCE_DIJKSTRA(graph, industrial_nodes, invert=True)
    
    FOR each node IN graph:
        features[node] ← [node.population_density, prox_map[node]]
    
    X ← normalize(features, method=MinMax)
    
    best_k ← 2
    best_silhouette ← -1
    
    FOR k IN [2, 3, 4, 5]:
        labels ← KMEANS(X, k=k, n_init=10, random_state=42)
        score ← SILHOUETTE_SCORE(X, labels)
        IF score > best_silhouette:
            best_silhouette ← score
            best_k ← k
            best_labels ← labels
    
    cluster_profiles ← INTERPRET_CENTROIDS(best_model.centroids)

PHASE B: SYNTHETIC DATA
    dataset ← []
    
    FOR each node, i IN enumerate(graph.nodes):
        pop ← X[i][0]
        ind ← X[i][1]
        loc ← node.location_type
        
        # Primary rule
        IF ind >= 0.7 AND pop >= 0.4: risk ← "High"
        ELSE IF ind >= 0.5 OR pop >= 0.7: risk ← "Medium"
        ELSE: risk ← "Low"
        
        # Modifier
        IF loc IN {"Industrial", "Power Plant"}: risk ← BUMP_UP(risk)
        ELSE IF loc == "School" AND ind >= 0.5: risk ← BUMP_UP(risk)
        ELSE IF loc IN {"Hospital", "Ambulance Depot"}: risk ← BUMP_DOWN(risk)
        
        # Noise (10%)
        IF random() < 0.10: risk ← ADJACENT_RISK(risk)
        
        dataset.append({features..., "risk_label": risk})
    
    df ← DataFrame(dataset)

PHASE C: CLASSIFICATION
    model ← DecisionTreeClassifier(max_depth=5, min_samples_split=4,
                                    criterion='gini', class_weight='balanced')
    
    X_train, X_test, y_train, y_test ← train_test_split(df, test_size=0.2)
    model.fit(X_train, y_train)
    
    PRINT classification_report(y_test, model.predict(X_test))
    
    FOR each node IN graph:
        pred ← model.predict(features[node])
        node.predicted_risk ← pred
        node.risk_index ← RISK_MULTIPLIERS[pred]
    
    _graph_version += 1

PHASE D: DYNAMIC LOOP (runs per simulation step)
    AT each simulation step s:
        FOR each emergency event AT node c:
            c.total_emergencies += 1
        
        IF s % DRIFT_WINDOW == 0:
            drifted ← []
            FOR each node IN graph:
                IF node.predicted_risk == "Low" AND node.total_emergencies >= 3:
                    drifted.append(node)
                IF node.predicted_risk == "Medium" AND node.total_emergencies >= 6:
                    drifted.append(node)
            
            IF drifted is not empty:
                corrections ← [{features, corrected_label} for node IN drifted]
                augmented_df ← df + corrections×3
                model.fit(augmented_df)
                
                FOR each node IN graph:
                    node.risk_index ← RISK_MULTIPLIERS[model.predict(features[node])]
                
                _graph_version += 1
                ambulance_placer.notify_risk_updated()
```

---

## Appendix B: Python Code Skeleton

The following skeleton shows how all your classes and functions connect. Fill in implementations from the sections above.

```python
# ============================================================
# FILE: challenge3_ambulance_ga.py
# OWNER: Muntazir Mehdi (24I-0847)
# ============================================================

import random
import heapq
from typing import List, Dict, Tuple

# --- CONSTANTS ---
GA_CONFIG = {
    "population_size": 100,
    "max_generations": 200,
    "tournament_size": 5,
    "crossover_rate": 0.85,
    "mutation_rate": 0.15,
    "elite_count": 2,
    "stagnation_limit": 30,
    "num_ambulances": 3,
}

_fitness_cache: Dict = {}
_graph_version: int = 0

# --- HELPER FUNCTIONS ---
def get_eligible_nodes(graph) -> List:
    """See §1.3"""
    pass

def get_citizen_nodes(graph) -> List:
    """See §1.5"""
    pass

def multi_source_dijkstra(graph, sources: List) -> Dict:
    """See §1.5"""
    pass

def calculate_fitness_optimized(chromosome: List, graph) -> float:
    """See §1.5"""
    pass

# --- INITIALIZATION ---
def initialize_population(graph, population_size: int) -> List:
    """See §1.4"""
    pass

def is_valid_chromosome(chromosome: List) -> bool:
    """See §1.3"""
    pass

# --- GA OPERATIONS ---
def tournament_selection(population, fitness_scores, tournament_size) -> List:
    """See §1.6"""
    pass

def crossover(parent1, parent2, eligible_nodes) -> Tuple:
    """See §1.7"""
    pass

def repair_chromosome(chromosome, eligible_nodes) -> List:
    """See §1.7"""
    pass

def mutate(chromosome, eligible_nodes, mutation_rate) -> List:
    """See §1.8"""
    pass

def adaptive_mutation_rate(base_rate, gens_no_improve, stagnation_threshold=20) -> float:
    """See §1.8"""
    pass

# --- MAIN GA ---
def run_ga_ambulance_placement(graph, config=GA_CONFIG) -> Tuple:
    """See §1.12 — full implementation"""
    pass

# --- RE-TRIGGER MANAGER ---
class AmbulancePlacer:
    """See §1.10"""
    pass


# ============================================================
# FILE: challenge5_crime_ml.py
# OWNER: Muntazir Mehdi (24I-0847)
# ============================================================

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import silhouette_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- FEATURE ENGINEERING ---
def compute_industrial_proximity(graph) -> Dict:
    """See §2.2"""
    pass

def extract_and_normalize_features(graph) -> Tuple:
    """See §2.2"""
    pass

# --- CLUSTERING ---
def run_kmeans_clustering(X_normalized, k_range=(2, 6)) -> Tuple:
    """See §2.2"""
    pass

def interpret_clusters(best_model, raw_features, best_k) -> Dict:
    """See §2.2"""
    pass

# --- SYNTHETIC DATA ---
def assign_ground_truth_risk(nodes_list, X_normalized, cluster_labels, cluster_profiles) -> List:
    """See §2.3"""
    pass

def build_training_dataframe(dataset) -> pd.DataFrame:
    """See §2.3"""
    pass

# --- CLASSIFIER ---
class CrimeRiskClassifier:
    """See §2.4 — full implementation"""
    
    FEATURES = ["population_density", "industrial_proximity",
                 "location_type_encoded", "cluster_id"]
    TARGET = "risk_label"
    RISK_MULTIPLIERS = {"High": 1.5, "Medium": 1.2, "Low": 1.0}
    
    def __init__(self): pass
    def train(self, df): pass
    def predict_all_nodes(self, nodes_list, X_normalized, cluster_labels): pass
    def apply_predictions_to_graph(self, predictions): pass

# --- DYNAMIC LEARNING LOOP ---
class CrimeStatsManager:
    """See §2.5 — full implementation"""
    
    DRIFT_WINDOW = 5
    LOW_RISK_EMERGENCY_THRESHOLD = 3
    MEDIUM_RISK_EMERGENCY_THRESHOLD = 6
    
    def __init__(self, graph, classifier): pass
    def notify_emergency(self, node): pass
    def step(self): pass
    def set_pipeline_data(self, nodes_list, X_normalized, cluster_labels): pass
    def _check_for_drift(self): pass
    def _retrain_with_corrections(self, drifted_nodes): pass

# --- MAIN PIPELINE ENTRY POINT ---
def run_crime_risk_pipeline(graph) -> CrimeStatsManager:
    """
    Full pipeline: runs Phases A-C and returns a configured
    CrimeStatsManager ready for Phase D monitoring.
    
    Call this ONCE before the simulation loop starts.
    Then call crime_stats.step() and crime_stats.notify_emergency()
    from inside the simulation loop.
    """
    print("="*50)
    print("CHALLENGE 5: Crime Risk Pipeline Starting")
    print("="*50)
    
    # Phase A: Clustering
    nodes_list, X_normalized, raw_features = extract_and_normalize_features(graph)
    best_k, cluster_labels, kmeans_model = run_kmeans_clustering(X_normalized)
    cluster_profiles = interpret_clusters(kmeans_model, raw_features, best_k)
    
    # Phase B: Synthetic Dataset
    dataset = assign_ground_truth_risk(nodes_list, X_normalized, cluster_labels, cluster_profiles)
    df = build_training_dataframe(dataset)
    
    # Phase C: Train Classifier
    classifier = CrimeRiskClassifier()
    classifier.train(df)
    
    # Apply initial predictions to graph
    predictions = classifier.predict_all_nodes(nodes_list, X_normalized, cluster_labels)
    classifier.apply_predictions_to_graph(predictions)
    for node, label in predictions.items():
        node.predicted_risk = label
    
    # Phase D: Setup monitor
    monitor = CrimeStatsManager(graph, classifier)
    monitor.set_pipeline_data(nodes_list, X_normalized, cluster_labels)
    
    print("\n[Challenge 5] Pipeline complete. Risk monitor active.")
    return monitor
```

---

*Document End*

---

**Version:** 1.0  
**Author:** Developer Manual for Muntazir Mehdi (24I-0847)  
**Project:** CityMind Urban Intelligence System  
**Course:** Artificial Intelligence — BS Computer Science  
**Group:** 24I-0548 (Ammar Iqbal) / 24I-0828 (Muhammad Ali Naqvi) / 24I-0847 (Muntazir Mehdi)  
**Deadline:** 10th May, 11:59 PM
