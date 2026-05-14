**CityMind**

*An Urban Intelligence System*

**PHASE 1: Problem Analysis and Design**

Deadline: 26th April

  -----------------------------------------------------------------------
  **Course**                 Artificial Intelligence
  -------------------------- --------------------------------------------
  **Programme**              BS Computer Science

  **Phase**                  1 of 3 --- Design Document

  **Deadline**               26th April
  -----------------------------------------------------------------------

> **24I-0828 Muhammad Ali Naqvi**
>
> **24I-0847 Muntazir Mehdi**
>
> **24-0548 Ammar Iqbal**

# 1. Problem Understanding

This section describes how our group understands each of the five
challenges presented in the CityMind project statement. All descriptions
are written in our own words following group discussion.

+-----------------------------------------------------------------------+
| **Challenge 1: *City Layout Planning***                               |
+=======================================================================+
| **Our Understanding**                                                 |
|                                                                       |
| This challenge asks us to place different types of city locations --- |
| hospitals, schools, industrial zones, residential areas, power        |
| plants, and ambulance depots --- onto an empty grid while satisfying  |
| a set of urban planning rules. The three core constraints are: (1)    |
| industrial zones cannot be adjacent to schools or hospitals; (2)      |
| every residential area must be within three road hops of at least one |
| hospital; and (3) power plants must be placed within two road hops of |
| an industrial zone.                                                   |
|                                                                       |
| What makes this problem interesting is the fallback requirement: if   |
| no fully valid layout is possible given the grid size, the system     |
| must still find a configuration that violates the fewest constraints  |
| and must identify exactly which rule caused the conflict.             |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| **Challenge 2: *Road Network Optimization***                          |
+=======================================================================+
| **Our Understanding**                                                 |
|                                                                       |
| Once the city layout exists, we must decide which roads to build. The |
| primary goal is to connect all city locations at the lowest possible  |
| total construction cost. There is, however, a non-negotiable safety   |
| requirement: there must always be at least two completely independent |
| paths between the Primary Hospital and the Ambulance Depot. If any    |
| single road is lost --- due to flooding or an accident --- an         |
| alternative route must still exist.                                   |
|                                                                       |
| In our initial group discussion we briefly considered a Genetic       |
| Algorithm, since we had recently covered it in class, but we quickly  |
| recognized that this is fundamentally a graph optimization problem    |
| rather than a search-over-configurations problem. Our final approach  |
| is described in Section 2.                                            |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| **Challenge 3: *Ambulance Placement***                                |
+=======================================================================+
| **Our Understanding**                                                 |
|                                                                       |
| The city has three ambulances that must be placed at depot locations  |
| in the grid. The objective is to minimize the worst-case response     |
| time --- meaning we want the citizen who is furthest from any         |
| ambulance to be as close as possible. This is a minimax placement     |
| problem, not a simple shortest-path problem.                          |
|                                                                       |
| The search space grows extremely fast: if the grid has N eligible     |
| positions and we place 3 ambulances, the number of unique             |
| combinations is C(N,3), which becomes enormous for realistic city     |
| sizes. Exhaustive search is not feasible.                             |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| **Challenge 4: *Emergency Routing Under Changing Conditions***        |
+=======================================================================+
| **Our Understanding**                                                 |
|                                                                       |
| A medical team must travel across the city to reach a sequence of     |
| trapped civilians. The environment is dynamic: roads can flood or     |
| become blocked while the team is already in transit. The moment a     |
| road becomes impassable, the route must be recalculated immediately   |
| --- and the recalculated route must still be the shortest available   |
| path, not just any valid path. The team must also visit all civilian  |
| locations, not just the nearest one.                                  |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| **Challenge 5: *Crime Risk Prediction and Integration***              |
+=======================================================================+
| **Our Understanding**                                                 |
|                                                                       |
| This challenge has three connected parts. First, we must group        |
| neighborhoods into clusters based on population density and           |
| industrial proximity --- without any pre-labeled training data.       |
| Second, we must generate a synthetic crime dataset using logic we     |
| design and justify ourselves, then train a classification model that  |
| assigns each neighborhood a risk level of High, Medium, or Low.       |
| Third, the predicted risk levels must feed back into the shared city  |
| graph as a cost multiplier, making high-risk areas more costly to     |
| route through.                                                        |
|                                                                       |
| Our group debated how to assign crime risk. One suggestion was that   |
| lower-population-density areas would be higher risk since they are    |
| less monitored. However, Muntazir made the observation --- supported  |
| by examples from Pakistani cities --- that areas close to industrial  |
| zones tend to have higher crime rates due to economic pressure,       |
| transient populations, and lower policing levels. We adopted this     |
| logic for our synthetic dataset generation.                           |
+-----------------------------------------------------------------------+

# 2. Algorithm Choices and Justifications

## 2.1 Challenge 1 --- Constraint Satisfaction Problem (CSP) with Backtracking

  -----------------------------------------------------------------------
  **Chosen         CSP with Backtracking + Minimum Remaining Values (MRV)
  Technique**      heuristic
  ---------------- ------------------------------------------------------
  **Why this       City layout planning is a textbook CSP: we have
  fits**           variables (locations to place), domains (valid grid
                   cells), and hard constraints (proximity rules).
                   Backtracking systematically assigns values and undoes
                   assignments the moment a constraint is violated,
                   exploring only consistent partial solutions. This is
                   far more efficient than generating random layouts and
                   checking them afterward.

  **Fallback       If no complete valid assignment exists, we switch to a
  handling**       min-conflict strategy: we place all location types
                   while tracking how many constraints each placement
                   violates, then select the assignment with the least
                   total violations and report the specific constraint
                   that caused the conflict.

  **Alternative    We briefly considered a Genetic Algorithm. GAs can
  considered**     produce near-valid layouts but offer no guarantee of
                   finding a valid solution if one exists, and offer no
                   structured way to identify exactly which constraint is
                   violated. CSP gives us both --- validity guarantee and
                   conflict identification --- which the problem
                   explicitly requires.
  -----------------------------------------------------------------------

## 2.2 Challenge 2 --- Kruskal\'s MST + Uniform Cost Search (UCS)

  -----------------------------------------------------------------------
  **Chosen         Kruskal\'s Minimum Spanning Tree algorithm, augmented
  Technique**      with a UCS-based secondary path to enforce the
                   redundancy constraint
  ---------------- ------------------------------------------------------
  **Why this       Kruskal\'s algorithm finds the globally optimal set of
  fits**           edges that connects all nodes at minimum total cost
                   --- which is exactly the road-building objective.
                   After building the MST, we check whether the Primary
                   Hospital and Ambulance Depot are connected by two
                   independent paths. Since the MST provides only one
                   path between any node pair, we then remove the direct
                   edge between these two nodes temporarily and run UCS
                   to find the cheapest alternative route. This ensures
                   redundancy while adding only the minimum extra cost
                   necessary.

  **Step-by-step   Sort all possible roads by cost. Add roads in order
  logic**          (Kruskal) until all locations are connected. Check
                   Hospital-to-Depot path count. If only one path exists,
                   run UCS on the graph-minus-direct-edge to find the
                   cheapest backup path and add those edges.

  **Alternative    A pure Genetic Algorithm could search for road
  considered**     configurations but would not guarantee minimum cost or
                   the specific two-path redundancy condition. Kruskal\'s
                   gives a mathematically provable minimum spanning tree,
                   and UCS guarantees the cheapest alternative path ---
                   both properties are verifiable and explainable.
  -----------------------------------------------------------------------

## 2.3 Challenge 3 --- Genetic Algorithm (GA)

  -----------------------------------------------------------------------
  **Chosen         Genetic Algorithm with population-based search,
  Technique**      fitness function targeting worst-case response time
  ---------------- ------------------------------------------------------
  **Why this       The placement of 3 ambulances across a large grid has
  fits**           a combinatorial search space that grows as C(N,3)
                   where N is the number of eligible positions. For a
                   20x20 grid this is already over 1,000 possible
                   combinations; realistic city sizes push this into the
                   tens of thousands. GA explores this space efficiently
                   through selection, crossover, and mutation without
                   needing to evaluate every combination. Importantly,
                   every chromosome (set of 3 placements) is
                   automatically a valid state since all positions are
                   legal --- this means mutation never produces invalid
                   solutions and no repair step is needed. The fitness
                   function computes the maximum shortest-path distance
                   from any citizen to the nearest ambulance and
                   minimizes it across generations.

  **Alternative    Simulated Annealing was considered. SA is also
  considered**     suitable for this search space but converges more
                   slowly and is harder to tune for a minimax objective.
                   GA\'s population-based approach naturally maintains
                   diversity and avoids local optima more reliably for
                   placement problems.
  -----------------------------------------------------------------------

## 2.4 Challenge 4 --- Dynamic A\* (D\* Lite)

  -----------------------------------------------------------------------
  **Chosen         D\* Lite (Dynamic A\*), a replanning variant of A\*
  Technique**      designed for changing environments
  ---------------- ------------------------------------------------------
  **Why this       Standard A\* finds the optimal path in a static graph,
  fits**           but Challenge 4 has a dynamic environment where roads
                   can become blocked mid-journey. Rerunning full A\*
                   from scratch every time a road changes is correct but
                   wasteful. D\* Lite is specifically designed for this:
                   it maintains a backward search tree from the goal and
                   when an edge weight changes it updates only the
                   affected portion of the search tree rather than
                   recalculating everything. This guarantees that the
                   path returned is always the current shortest available
                   path --- satisfying the explicit requirement of the
                   problem --- while being efficient enough to run in
                   real time. The heuristic used is Euclidean or
                   Manhattan distance to the current target civilian,
                   which is admissible, ensuring optimality.

  **Alternative    Plain A\* with full replan on every change would also
  considered**     guarantee optimality but is computationally expensive
                   for large grids with frequent changes. BFS guarantees
                   shortest path in unweighted graphs but our graph has
                   variable edge weights (risk multipliers from Challenge
                   5), so BFS would not find the optimal weighted path.
  -----------------------------------------------------------------------

## 2.5 Challenge 5 --- K-Means Clustering + Decision Tree Classifier

  -------------------------------------------------------------------------
  **Step 1 ---       K-Means Clustering (unsupervised). Neighborhoods are
  Clustering**       grouped based on population density and industrial
                     proximity without any labelled data. K-Means is
                     suitable here because both features are continuous
                     numeric values and we want naturally formed clusters
                     rather than predefined categories. We will experiment
                     with k=3 or k=4 clusters.
  ------------------ ------------------------------------------------------
  **Step 2 ---       We generate a synthetic crime dataset using the
  Dataset            following logic: neighborhoods with high industrial
  generation**       proximity AND high population density are rated High
                     risk (transient population, economic pressure);
                     neighborhoods with high industrial proximity but low
                     residential density are rated Medium risk;
                     low-density, low-proximity neighborhoods are rated Low
                     risk. This is consistent with Muntazir\'s real-world
                     observation about Pakistani cities.

  **Step 3 ---       Decision Tree Classifier (supervised). The classifier
  Classification**   is trained on our generated dataset to predict High /
                     Medium / Low risk for each neighborhood. We chose a
                     Decision Tree over K-Nearest Neighbors because
                     Decision Trees produce interpretable rules that we can
                     trace and explain during the viva. KNN is a reasonable
                     alternative but functions as a black-box distance
                     comparator, making it harder to justify specific
                     predictions.

  **Integration**    The predicted risk level for each node is converted to
                     a cost multiplier: High = 1.5x, Medium = 1.2x, Low =
                     1.0x. These multipliers update the shared city graph
                     edge weights and are immediately visible to the
                     ambulance placement module (Challenge 3) and the
                     emergency router (Challenge 4).

  **Why two          Clustering is unsupervised because we do not have
  different learning pre-existing crime labels --- we are discovering
  types**            structure. Classification is supervised because we use
                     our generated (labelled) dataset to learn a mapping
                     from features to risk level. These are genuinely
                     different learning paradigms applied to different
                     sub-problems.
  -------------------------------------------------------------------------

# 3. Shared City Graph Structure

The shared city graph is the single source of truth for the entire
CityMind system. Every module reads from and writes to this one object
--- no module keeps its own internal copy. Any change (a road flood, an
updated risk weight, a new ambulance assignment) is immediately visible
to all other components.

We use a hybrid Object-Oriented design: a 2D array of CityNode objects
where each node stores its own adjacency dictionary of neighbors. This
gives us the spatial advantages of a grid and the routing efficiency of
an adjacency list at the same time.

## 3.1 The CityNode Class

Each position in the grid is represented by a CityNode object. The node
holds two categories of data --- static properties that do not change
after layout initialization, and dynamic properties that are updated
during the simulation:

  ----------------------------------------------------------------------------
  **Coordinates**        row, col --- integer grid position. Used by A\* / D\*
                         Lite as the spatial heuristic input (Manhattan
                         distance to target).
  ---------------------- -----------------------------------------------------
  **Static:              One of: Residential, Hospital, School, Industrial,
  location_type**        Power Plant, Ambulance Depot. Set once by Challenge
                         1.

  **Static:              Numeric value indicating how many people live or work
  population_density**   at this location. Used as input to the K-Means
                         clustering in Challenge 5.

  **Dynamic:             Float value updated by the ML classifier in Challenge
  risk_index**           5. Converted to a cost multiplier: High = 1.5, Medium
                         = 1.2, Low = 1.0.

  **Dynamic:             Boolean flag. Set to False when a road event makes
  is_accessible**        this location unreachable. All pathfinding queries
                         check this flag first.

  **Dynamic: neighbors** A Python dict mapping each adjacent CityNode to its
                         current edge weight. This is the adjacency structure
                         --- stored directly on the node, not in a separate
                         edge list.
  ----------------------------------------------------------------------------

## 3.2 The CityGraph Manager

All CityNode objects are wrapped in a CityGraph class that owns the 2D
array and exposes helper methods used by every module. Storing the grid
as a 2D array (rather than a flat list of nodes) allows Challenge 1 and
Challenge 3 to perform spatial proximity queries directly --- for
example, checking whether a residential node is within 3 hops of a
hospital by doing a BFS starting at grid\[r\]\[c\], without needing to
search through the entire node list.

## 3.3 Edge Weight Logic

  -----------------------------------------------------------------------
  **Standard       Weight = 1.0. Default for all connections.
  road**           
  ---------------- ------------------------------------------------------
  **Residential    Weight = 0.8. Set during Challenge 2 road
  zone road**      construction.

  **Effective      base_weight × destination.risk_index. Risk multipliers
  travel cost**    from Challenge 5 are already stored on the node, so
                   this multiplication happens at traversal time inside
                   the router --- no separate data structure needed.

  **Blocked road   node.block_road_to(neighbor) removes the key from the
  (flood)**        neighbors dict in O(1). The change is immediate ---
                   D\* Lite in Challenge 4 detects the missing edge on
                   its next traversal attempt and triggers replanning.
  -----------------------------------------------------------------------

## 3.4 Module Interaction with the Graph

The table below summarises what each module does with the shared graph
object:

  --------------------------------------------------------------------------
  **Module**    **Reads from       **Writes to        **Trigger**
                Graph**            Graph**            
  ------------- ------------------ ------------------ ----------------------
  **C1:         Empty grid         Node types,        Runs once at startup
  Layout**      structure          coordinates,       
                                   initial            
                                   accessibility      

  **C2: Roads** Node positions     Edge weights, road Runs after C1
                from C1            connections        completes

  **C3:         Node positions,    Ambulance depot    Runs at simulation
  Ambulance**   edge weights, risk assignments on     start; re-evaluates as
                multipliers        nodes              risk weights change

  **C4:         Edge weights,      Nothing ---        Triggered on each
  Router**      accessibility      read-only          emergency event and on
                flags, risk                           any road-blocked event
                multipliers                           

  **C5: Crime   Population         Risk multiplier on Runs once before
  ML**          density, location  each node          simulation; can re-run
                type, industrial                      if layout changes
                proximity per node                    
  --------------------------------------------------------------------------

# 4. User Interface Wireframe

The CityMind interface is divided into four zones: a top control bar,
the main city grid canvas, a right-side panel with layer toggles and
statistics, and a bottom live event log. The layout is described below
and a visual sketch follows.

  -----------------------------------------------------------------------
  **Top Bar**      Application title, simulation control buttons
                   (Initialize, Run Step, Run All, Reset), and a status
                   indicator showing the current simulation step number.
  ---------------- ------------------------------------------------------
  **City Grid      The main visualization area. Displays the 20x20 city
  Canvas           grid with color-coded nodes by location type. Supports
  (center)**       three overlay toggles: Road Network (edges drawn
                   between connected nodes), Ambulance Coverage (colored
                   coverage zones around each ambulance depot), and Crime
                   Risk Heatmap (node fill intensity based on predicted
                   risk level).

  **Right Panel**  Layer toggle checkboxes for the three overlays. A
                   summary statistics box showing total road cost,
                   worst-case ambulance response distance, and current
                   blocked road count.

  **Bottom Event   A scrollable chronological list of system decisions:
  Log**            e.g. \'Step 7: Road (3,4)-(3,5) blocked. Router
                   recalculated path for Team A via (3,6)-(4,6).\'
                   Entries are color-coded by event type (routing = blue,
                   risk update = orange, road blocked = red).

  **Legend**       A color key for node types and risk levels, displayed
                   in the bottom-right corner of the canvas.
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------
  ![](./image1.png){width="6.427083333333333in"
  height="9.000694444444445in"}
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

# 5. Work Breakdown

Work is divided evenly across three group members. Each member is
responsible for two primary deliverables in Phase 2 plus their
corresponding section in this Phase 1 document. Integration of the
shared city graph is a joint responsibility with a designated lead.

+--------------+-----------------------------+------------------------+
| **Member**   | **Responsibilities**        | **Phase 1              |
|              |                             | Contribution**         |
+==============+=============================+========================+
| **Member 1\  | -   Challenge 1: CSP city   | -   Wrote problem      |
| (Ali         |     layout planner with     |     understanding for  |
| Naqvi)**     |     backtracking and        |     C1 and C2          |
|              |     min-conflict fallback   |                        |
|              |                             | -   Researched and     |
|              | -   Challenge 2: Road       |     justified CSP vs   |
|              |     network builder using   |     GA for C1          |
|              |     Kruskal\'s MST + UCS    |                        |
|              |     redundancy path         | -   Researched and     |
|              |                             |     justified          |
|              | -   Phase 1: Authored       |     Kruskal + UCS vs   |
|              |     Sections 1.1, 1.2, 2.1, |     pure GA for C2     |
|              |     2.2                     |                        |
+--------------+-----------------------------+------------------------+
| **Member 2   | -   Challenge 3: Genetic    | -   Wrote problem      |
| (Muntazir)** |     Algorithm ambulance     |     understanding for  |
|              |     placement with minimax  |     C3 and C5          |
|              |     fitness                 |                        |
|              |                             | -   Proposed           |
|              | -   Challenge 5: K-Means    |     industrial-zone    |
|              |     clustering + Decision   |     crime risk logic   |
|              |     Tree classifier + graph |     for synthetic      |
|              |     integration             |     dataset            |
|              |                             |                        |
|              | -   Phase 1: Authored       | -   Researched GA for  |
|              |     Sections 1.3, 1.5, 2.3, |     C3 and Decision    |
|              |     2.5                     |     Tree vs KNN for C5 |
+--------------+-----------------------------+------------------------+
| **Member 3** | -   Challenge 4: D\* Lite   | -   Wrote problem      |
|              |     emergency routing with  |     understanding for  |
| (Ammar       |     real-time replanning    |     C4                 |
| Iqbal)       |                             |                        |
|              | -   Shared city graph       | -   Designed city      |
|              |     structure and           |     graph structure    |
|              |     initialization module   |     and module         |
|              |                             |     interaction table  |
|              | -   Visual interface (city  |                        |
|              |     grid, overlay toggles,  | -   Drafted the UI     |
|              |     event log)              |     wireframe          |
|              |                             |     description        |
|              | -   Phase 1: Authored       |                        |
|              |     Sections 1.4, 2.4, 3, 4 | -   Researched D\*     |
|              |                             |     Lite vs plain A\*  |
|              |                             |     for C4             |
+--------------+-----------------------------+------------------------+

+-----------------------------------------------------------------------+
| **Note on AI Tool Usage**                                             |
|                                                                       |
| We used AI tools during Phase 1 for brainstorming and to compare      |
| algorithmic alternatives. All decisions --- especially the choice of  |
| D\* Lite over plain A\*, the hybrid Kruskal + UCS approach, and the   |
| crime risk logic --- were debated within the group before adoption.   |
| Every algorithm in this document can be explained by the group member |
| responsible for it from first principles.                             |
+=======================================================================+
+-----------------------------------------------------------------------+

**MEET DISCUSSION :**

![](./image2.jpeg){width="6.75in" height="3.2083333333333335in"}
