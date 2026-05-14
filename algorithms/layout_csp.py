# =============================================================================
# algorithms/layout_csp.py — Challenge 1: CSP City Layout Planner
# =============================================================================
# CSP Formulation
# ---------------
#   Variables  : every CityNode in the 20x20 grid  (400 variables)
#   Domain     : {Empty, Residential, Hospital, School,
#                 Industrial, Power Plant, Ambulance Depot}
#
# Constraints
# -----------
#   C1 (Binary)    Industrial cannot be adjacent to Hospital or School
#   C2 (Proximity) Every Residential within 3 grid hops of a Hospital
#   C3 (Proximity) Every Power Plant within 2 grid hops of an Industrial zone
#   C4 (Quota)     Exact counts per type must match QUOTAS
#
# Algorithm — Two-Phase Approach
# --------------------------------
#   Phase 0  : Seed hospitals (3x3 grid, 9 total) + industrial (30 total).
#              Precompute hospital_covered and industrial_covered sets.
#              Prune Residential from uncovered-zone domains immediately.
#
#   Phase 1A : Backtrack over the COVERED zone (203 nodes) first.
#              Assigns Residential, School, Power Plant, Ambulance, Empty.
#              After this, C2 and C3 are guaranteed satisfied.
#
#   Phase 1B : Backtrack over the UNCOVERED zone (158 nodes).
#              Only Empty and School (if not adjacent to Industrial) are legal.
#              Very fast — almost no real choices.
#
#   Phase 2  : Min-Conflicts repair only if either phase hits budget.
#
# Why two phases?
# ---------------
# Interleaving covered/uncovered nodes causes MRV to jump between zones,
# creating conflicts that are hard to resolve (Residential lands outside
# hospital coverage because backtracking didn't know to avoid that).
# Separating the zones makes the constraint graph almost trivially satisfiable
# in each phase, and the backtracking terminates in milliseconds.
# =============================================================================

import random
from collections import defaultdict
import math

from config import (
    LOC_EMPTY, LOC_RESIDENTIAL, LOC_HOSPITAL, LOC_SCHOOL,
    LOC_INDUSTRIAL, LOC_POWER_PLANT, LOC_AMBULANCE,
    ALL_LOCATION_TYPES, GRID_ROWS, GRID_COLS,
)
from models.city_graph import CityGraph
from models.city_node import CityNode

# ---------------------------------------------------------------------------
# Quotas  (9 hospitals instead of 4 — necessary for 3-hop coverage of 120
# residential cells on a 20x20 grid.  9 hospitals in a 3x3 grid cover 219
# cells at 3 hops, which comfortably fits 120 residential.)
# ---------------------------------------------------------------------------
QUOTAS = {
    LOC_RESIDENTIAL: 120,
    LOC_HOSPITAL:      9,
    LOC_SCHOOL:       20,
    LOC_INDUSTRIAL:   30,
    LOC_POWER_PLANT:   4,
    LOC_AMBULANCE:     3,
    LOC_EMPTY:        39,  # 225 total cells
}


assert sum(QUOTAS.values()) == GRID_ROWS * GRID_COLS  # 120+9+20+30+4+3+214 = 400

MAX_BT_NODES         = 50_000
HOSPITAL_HOP_LIMIT   = 3   # hard constraint — enforced by domain restriction
INDUSTRIAL_HOP_LIMIT = 2

_INDUSTRIAL_SENSITIVE = {LOC_HOSPITAL, LOC_SCHOOL}


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def run_layout(graph: CityGraph) -> dict:
    """
    Assign location types to all 400 grid nodes satisfying all constraints.
    Returns dict with:
        - success: True if zero-violation layout found, False if min-conflicts used
        - violations: dict with c1, c2, c3, total violation counts
    """
    solver = CSPSolver(graph)
    success = solver.solve()
    solver.apply_to_graph()
    violations = solver.get_violations()
    return {
        "success": success,
        "violations": violations
    }


# =============================================================================
# CONSTRAINT CHECKERS
# =============================================================================

def _adjacent_values(node, assignment, graph):
    return [assignment[nb] for nb in graph.grid_neighbors(node) if nb in assignment]


def check_industrial_separation(node, value, assignment, graph) -> bool:
    """C1: Industrial cannot be adjacent to Hospital or School."""
    adj = _adjacent_values(node, assignment, graph)
    if value == LOC_INDUSTRIAL:
        return not any(t in _INDUSTRIAL_SENSITIVE for t in adj)
    if value in _INDUSTRIAL_SENSITIVE:
        return LOC_INDUSTRIAL not in adj
    return True


def check_quota_not_exceeded(value, used_counts) -> bool:
    """C4: Don't exceed quota for any type."""
    return used_counts[value] < QUOTAS[value]


def is_consistent(node, value, assignment, graph, used_counts) -> bool:
    """
    Constraint checker used during backtracking.
    C2 and C3 (proximity) are enforced by domain restriction at setup time,
    not checked here — so this is fast (just quota + binary adjacency).
    """
    adj = _adjacent_values(node, assignment, graph)
    if value == LOC_AMBULANCE and LOC_AMBULANCE in adj:
        return False

    #there is no rule defined but the placement of the abulance depot in the challenge 1    
    return (
        check_quota_not_exceeded(value, used_counts)
        and check_industrial_separation(node, value, assignment, graph)
    )


# =============================================================================
# CONFLICT COUNTER  (for min-conflicts phase)
# =============================================================================

def count_conflicts(node, value, assignment, graph,
                    hospital_covered=None, industrial_covered=None) -> int:
    """Number of constraint violations if node = value.
    
    If hospital_covered/industrial_covered is None, performs LIVE BFS check
    against the current assignment (correct but slower).
    If a set is provided, uses cached coverage (faster but may be stale).
    """
    c = 0
    adj = _adjacent_values(node, assignment, graph)
    if value == LOC_INDUSTRIAL:
        c += sum(1 for t in adj if t in _INDUSTRIAL_SENSITIVE)
    elif value in _INDUSTRIAL_SENSITIVE:
        c += adj.count(LOC_INDUSTRIAL)
    
    # C2: Residential must be within HOSPITAL_HOP_LIMIT of a hospital
    if value == LOC_RESIDENTIAL:
        if hospital_covered is not None:
            # Use cache (may be stale!)
            if node not in hospital_covered:
                c += 1
        else:
            # Live BFS check against current assignment (correct)
            covered = any(
                assignment.get(nb) == LOC_HOSPITAL
                for nb in graph.bfs_hops(node, HOSPITAL_HOP_LIMIT)
            )
            if not covered:
                c += 1
    
    # C3: Power plant must be within INDUSTRIAL_HOP_LIMIT of an industrial zone
    if value == LOC_POWER_PLANT:
        if industrial_covered is not None:
            if node not in industrial_covered:
                c += 1
        else:
            covered = any(
                assignment.get(nb) == LOC_INDUSTRIAL
                for nb in graph.bfs_hops(node, INDUSTRIAL_HOP_LIMIT)
            )
            if not covered:
                c += 1
    return c


def total_conflict_count(assignment, graph,
                         hospital_covered=None, industrial_covered=None) -> int:
    # 1. Count spatial violations (adjacency and proximity) — each node once
    spatial_conflicts = sum(
        count_conflicts(n, v, assignment, graph, hospital_covered, industrial_covered)
        for n, v in assignment.items()
    )

    # 2. Count quota shortfalls (missing buildings) — per building type, not per node
    used_counts = defaultdict(int)
    for v in assignment.values():
        used_counts[v] += 1

    quota_conflicts = sum(
        max(0, target - used_counts[loc_type])
        for loc_type, target in QUOTAS.items()
    )

    return spatial_conflicts + quota_conflicts


# =============================================================================
# PHASE 0 — SEEDING AND COVERAGE PRECOMPUTATION
# =============================================================================


def _seed_anchors(graph: CityGraph) -> dict:
    """
    Dynamically place hospitals and industrial zones driven purely by QUOTAS.
    Works for any hospital count — no hardcoded [3, 10, 17] positions.
    """
    assignment = {}
    rows, cols = graph.rows, graph.cols
    n_hospitals = QUOTAS[LOC_HOSPITAL]

    # ── Hospitals: compute an evenly-spaced sub-grid ──────────────────────
    # Find the nearest rectangular arrangement for n_hospitals.
    # e.g. 9→3×3, 16→4×4, 12→3×4, 6→2×3, 4→2×2
    nr = max(1, round(math.sqrt(n_hospitals)))
    nc = math.ceil(n_hospitals / nr)

    # Space hospitals evenly by putting them at cell-centre fractions
    row_positions = [int((i + 0.5) * rows / nr) for i in range(nr)]
    col_positions = [int((j + 0.5) * cols / nc) for j in range(nc)]

    placed_hospitals = 0
    for qr in row_positions:
        for qc in col_positions:
            if placed_hospitals >= n_hospitals:
                break
            node = graph.node(qr, qc)
            assignment[node] = LOC_HOSPITAL
            placed_hospitals += 1

    # ── Industrial: scale belt density to quota ───────────────────────────
    # Instead of fixed rows 8 and 14, pick belt rows proportionally.
    n_industrial = QUOTAS[LOC_INDUSTRIAL]
    belt_rows = [int(rows * 0.25), int(rows * 0.5), int(rows * 0.75)]  # 3 belts for even vertical coverage

    candidates = []
    for i, belt_row in enumerate(belt_rows):
        step = 2
        offset = i % 2          # alternate start col so belts interleave
        for col in range(offset, cols - 1, step):
            candidates.append((belt_row, col))
    random.shuffle(candidates)

    placed = 0
    for row, col in candidates:
        if placed >= n_industrial:
            break
        node = graph.node(row, col)
        if node in assignment:
            continue
        if any(assignment.get(nb) == LOC_HOSPITAL
               for nb in graph.grid_neighbors(node)):
            continue
        assignment[node] = LOC_INDUSTRIAL
        placed += 1

    # Fallback: fill remaining quota from any free non-hospital-adjacent node
    for node in graph.all_nodes():
        if placed >= n_industrial:
            break
        if node in assignment:
            continue
        if any(assignment.get(nb) == LOC_HOSPITAL
               for nb in graph.grid_neighbors(node)):
            continue
        assignment[node] = LOC_INDUSTRIAL
        placed += 1

    return assignment


def _precompute_coverage(anchors, graph):
    """BFS from each seeded hospital/industrial — called once, results cached."""
    hcov, icov = set(), set()
    for node, val in anchors.items():
        if val == LOC_HOSPITAL:
            hcov |= graph.bfs_hops(node, HOSPITAL_HOP_LIMIT)
        elif val == LOC_INDUSTRIAL:
            icov |= graph.bfs_hops(node, INDUSTRIAL_HOP_LIMIT)
    return hcov, icov


# =============================================================================
# CSP SOLVER
# =============================================================================

class CSPSolver:
    """
    Two-phase backtracking CSP solver.

    Phase A: covered zone (hospital_covered ∩ non-anchored nodes)
             Domains: all types valid by quota and adjacency.
             Residential ONLY appears in covered-zone domains.
    Phase B: uncovered zone (remaining non-anchored nodes)
             Domains: Empty + School (if adjacency allows) + Ambulance.
             These nodes can never be Residential or Power Plant.

    Separating zones eliminates the main source of backtracking failures:
    Residential landing outside hospital coverage.
    """

    def __init__(self, graph: CityGraph):
        self.graph             = graph
        self._nodes            = list(graph.all_nodes())
        self._domains          = {n: list(ALL_LOCATION_TYPES) for n in self._nodes}
        self._assignment       = {}
        self._used             = defaultdict(int)
        self._bt_count         = 0
        self._success          = False
        self._hospital_covered   = set()
        self._industrial_covered = set()

        # Two ordered lists filled after seeding
        self._covered_nodes   = []   # Phase A
        self._uncovered_nodes = []   # Phase B

    # ---- public --------------------------------------------------------

    def solve(self) -> bool:
        # Phase 0: seed
        anchors = _seed_anchors(self.graph)
        self._anchors = set(anchors.keys())  # locked — min-conflicts must not move these
        self._hospital_covered, self._industrial_covered = \
            _precompute_coverage(anchors, self.graph)
        # =================================================================
        # APPLY ANCHORS + PARTITION NODES
        # (must happen before validity checks so min-conflicts has data)
        # =================================================================
        for node, val in anchors.items():
            self._assignment[node] = val
            self._used[val] += 1
            self._domains[node] = [val]

        print(f"[C1] Seeded {len(anchors)} anchors "
              f"({self._used[LOC_HOSPITAL]} hospitals, "
              f"{self._used[LOC_INDUSTRIAL]} industrial).")
        print(f"[C1] Hospital coverage: {len(self._hospital_covered)}/400 cells | "
              f"Industrial coverage: {len(self._industrial_covered)}/400 cells")

        # Partition remaining nodes into covered / uncovered zones
        for node in self._nodes:
            if node in self._assignment:
                continue
            if node in self._hospital_covered:
                self._covered_nodes.append(node)
            else:
                self._uncovered_nodes.append(node)

        # Prune domains by zone
        for node in self._covered_nodes:
            if node not in self._industrial_covered:
                self._domains[node] = [v for v in self._domains[node]
                                        if v != LOC_POWER_PLANT]
        for node in self._uncovered_nodes:
            self._domains[node] = [v for v in self._domains[node]
                                    if v not in {LOC_RESIDENTIAL, LOC_POWER_PLANT}]

        print(f"[C1] Zone split: {len(self._covered_nodes)} covered nodes, "
              f"{len(self._uncovered_nodes)} uncovered nodes.")

        # =================================================================
        # =================================================================
        # MATHEMATICAL VALIDITY CHECKS — run ALL before acting
        # =================================================================
        needed_res      = QUOTAS[LOC_RESIDENTIAL]
        actual_coverage = len(self._hospital_covered)
        c2_violated     = needed_res > actual_coverage

        needed_power = QUOTAS[LOC_POWER_PLANT]
        ind_coverage = len(self._industrial_covered)
        c3_violated  = needed_power > ind_coverage

        if c2_violated or c3_violated:
            print("\n========================================================")
            print("❌ MATHEMATICAL VALIDITY CHECK: FAILED")
            print("========================================================")
            if c2_violated and c3_violated:
                print("Conflicting Rules: C2 (Proximity) AND C3 (Proximity)")
            elif c2_violated:
                print("Specific Rule Causing Conflict: Constraint C2 (Proximity)")
            else:
                print("Specific Rule Causing Conflict: Constraint C3 (Proximity)")

            if c2_violated:
                print(f"\n  [C2] 'Every Residential must be within {HOSPITAL_HOP_LIMIT} hops of a Hospital'.")
                print(f"       Quota demands {needed_res} residential, but "
                      f"{QUOTAS[LOC_HOSPITAL]} hospital(s) cover only "
                      f"{actual_coverage} cells (shortfall: {needed_res - actual_coverage}).")
            if c3_violated:
                print(f"\n  [C3] 'Every Power Plant must be within {INDUSTRIAL_HOP_LIMIT} hops of an Industrial zone'.")
                print(f"       Quota demands {needed_power} power plants, but "
                      f"{QUOTAS[LOC_INDUSTRIAL]} industrial zone(s) cover only "
                      f"{ind_coverage} cells (shortfall: {needed_power - ind_coverage}).")

            print("\nAction: Bypassing standard CSP search. Proposing Minimum Conflict Solution...\n")
            self._success = False
            self._min_conflicts_repair()
            return False
        # ================================================================= 

        # Phase 1A: backtrack over covered zone
        print("[C1] Phase A — backtracking over covered zone (MRV+LCV+FC)...")
        covered_unassigned = set(self._covered_nodes)
        success_a = self._backtrack(covered_unassigned)
        

        # Phase A "succeeded" structurally but may not have placed enough residential
        if success_a:
            placed_res = self._used[LOC_RESIDENTIAL]
            needed_res = QUOTAS[LOC_RESIDENTIAL]
            if placed_res < needed_res:
                print(f"[C1] Phase A placed only {placed_res}/{needed_res} residential "
                      f"— hospital coverage too small for this quota. "
                      f"Need at least {needed_res} covered cells, have {len(self._covered_nodes)}.")
                success_a = False

        # Phase 1B: backtrack over uncovered zone
        if success_a:
            print(f"[C1] Phase A done ({self._bt_count} backtracks). "
                  "Phase B — uncovered zone...")
            bt_before = self._bt_count
            uncovered_unassigned = set(self._uncovered_nodes)
            success_b = self._backtrack(uncovered_unassigned)
            print(f"[C1] Phase B done ({self._bt_count - bt_before} backtracks).")
            self._success = success_b
        else:
            print(f"[C1] Phase A budget hit. Running Min-Conflicts...")
            self._success = False

        if not self._success:
            self._min_conflicts_repair()

        return self._success

    def apply_to_graph(self) -> None:
        for node, loc_type in self._assignment.items():
            self.graph.set_location_type(node, loc_type)
            self.graph.set_population_density(node, _density_for(loc_type))

        # Full constraint verification — all 3 rules
        res_nodes = [n for n, v in self._assignment.items() if v == LOC_RESIDENTIAL]
        pp_nodes  = [n for n, v in self._assignment.items() if v == LOC_POWER_PLANT]

        c2_violations = sum(
            1 for node in res_nodes
            if not any(self._assignment.get(nb) == LOC_HOSPITAL
                       for nb in self.graph.bfs_hops(node, HOSPITAL_HOP_LIMIT))
        )
        c3_violations = sum(
            1 for node in pp_nodes
            if not any(self._assignment.get(nb) == LOC_INDUSTRIAL
                       for nb in self.graph.bfs_hops(node, INDUSTRIAL_HOP_LIMIT))
        )
        c1_violations = 0
        for n, v in self._assignment.items():
            adj = [self._assignment.get(nb)
                   for nb in self.graph.grid_neighbors(n) if nb in self._assignment]
            if v == LOC_INDUSTRIAL:
                c1_violations += sum(1 for t in adj if t in _INDUSTRIAL_SENSITIVE)
            elif v in _INDUSTRIAL_SENSITIVE:
                c1_violations += adj.count(LOC_INDUSTRIAL)
        c1_violations //= 2  # each pair counted from both sides

        print("\n" + "=" * 60)
        print("  CONSTRAINT VERIFICATION (applied layout)")
        print("=" * 60)
        print(f"  C1 Industrial adjacency  : "
              f"{'✔  0 violations' if c1_violations == 0 else f'✘  {c1_violations} violation(s) — industrial touches hospital/school'}")
        print(f"  C2 Residential coverage  : "
              f"{'✔  0 violations' if c2_violations == 0 else f'✘  {c2_violations} violation(s) — residential outside 3-hop hospital range'}")
        print(f"  C3 Power plant coverage  : "
              f"{'✔  0 violations' if c3_violations == 0 else f'✘  {c3_violations} violation(s) — power plant outside 2-hop industrial range'}")
        total = c1_violations + c2_violations + c3_violations
        print(f"  {'─' * 38}")
        print(f"  Total violations         : {total} "
              f"({'fully satisfied' if total == 0 else 'minimum possible given current quotas'})")
        print("=" * 60 + "\n")

        # Store violations for retrieval
        self._violations = {
            "c1": c1_violations,
            "c2": c2_violations,
            "c3": c3_violations,
            "total": total
        }

    def get_violations(self) -> dict:
        """Return constraint violation counts after solve."""
        return getattr(self, '_violations', {"c1": 0, "c2": 0, "c3": 0, "total": 0})

    # ---- backtracking --------------------------------------------------

    def _backtrack(self, unassigned: set) -> bool:
        """
        Generic backtracking over a given set of nodes.
        MRV picks from `unassigned`; LCV orders values; FC prunes neighbours
        that are also in `unassigned`.
        """
        if not unassigned:
            return True
        if self._bt_count > MAX_BT_NODES:
            return False

        node   = self._select_mrv(unassigned)
        values = self._order_lcv(node, unassigned)

        for value in values:
            if not is_consistent(node, value, self._assignment, self.graph, self._used):
                continue

            self._assignment[node] = value
            self._used[value] += 1
            unassigned.discard(node)

            pruned = self._forward_check(node, value, unassigned)
            if pruned is not None:
                if self._backtrack(unassigned):
                    return True

            del self._assignment[node]
            self._used[value] -= 1
            unassigned.add(node)
            if pruned is not None:
                self._restore_domains(pruned)

        self._bt_count += 1
        return False

    # ---- MRV -----------------------------------------------------------

    def _select_mrv(self, unassigned: set) -> CityNode:
        """
        MRV: pick the node from `unassigned` with the fewest remaining
        domain values. Ties broken by degree (more assigned neighbours = first).
        """
        best_node, best_d, best_deg = None, float('inf'), -1
        for n in unassigned:
            d = len(self._domains[n])
            if d < best_d:
                best_d, best_node = d, n
                best_deg = sum(1 for nb in self.graph.grid_neighbors(n)
                               if nb in self._assignment)
            elif d == best_d:
                deg = sum(1 for nb in self.graph.grid_neighbors(n)
                          if nb in self._assignment)
                if deg > best_deg:
                    best_deg, best_node = deg, n
        return best_node

    # ---- LCV -----------------------------------------------------------

    def _order_lcv(self, node: CityNode, unassigned: set) -> list:
        """
        LCV: sort domain values so we try the least-constraining one first.

        Three-component score (all ascending — lower = tried first):
          1. adj_score   — adjacency conflict count (Industrial separation)
          2. fill_ratio  — how full the quota already is (0.0=empty→try first)
          3. zone_bonus  — negative discount for correct zone matches:
                           Residential in hospital zone, PowerPlant in industrial zone.
                           Positive penalty for Empty in covered zones.

        This fixes the bug where raw `remaining` made Empty (39 left) sort
        before Residential (120 left) in covered zones — the opposite of intent.
        fill_ratio normalises across quotas so all types compete fairly.
        """
        in_hospital_zone   = node in self._hospital_covered
        in_industrial_zone = node in self._industrial_covered

        def score(value: str) -> tuple:
            # Component 1: adjacency conflict count (unchanged)
            if value == LOC_INDUSTRIAL:
                adj_score = sum(
                    1 for nb in self.graph.grid_neighbors(node)
                    if nb in unassigned
                    and any(v in _INDUSTRIAL_SENSITIVE for v in self._domains[nb])
                )
            elif value in _INDUSTRIAL_SENSITIVE:
                adj_score = sum(
                    1 for nb in self.graph.grid_neighbors(node)
                    if nb in unassigned
                    and LOC_INDUSTRIAL in self._domains[nb]
                )
            else:
                adj_score = 0

            # Component 2: fill ratio — 0.0 (unfilled) tried first, 1.0 (full quota) last
            fill_ratio = self._used[value] / max(QUOTAS[value], 1)

            # Component 3: zone-aware bonus (negative = preferred, positive = discouraged)
            zone_bonus = 0.0
            if value == LOC_RESIDENTIAL and in_hospital_zone:
                zone_bonus = -0.8   # strongly prefer Residential in covered zone
            elif value == LOC_POWER_PLANT and in_industrial_zone:
                zone_bonus = -0.6   # prefer PowerPlant in industrial zone
            elif value == LOC_EMPTY and in_hospital_zone:
                zone_bonus = +0.3   # mildly discourage Empty in covered zone

            return (adj_score, fill_ratio + zone_bonus)

        return sorted(self._domains[node], key=score)

    # ---- forward checking ----------------------------------------------

    def _forward_check(self, node: CityNode, value: str, unassigned: set):
        """
        Prune now-illegal values from unassigned neighbours' domains.
        Returns {node -> [pruned values]} or None on domain wipeout.
        """
        pruned = defaultdict(list)
        temp   = {**self._assignment, node: value}

        for nb in self.graph.grid_neighbors(node):
            if nb not in unassigned:
                continue
            for nb_val in list(self._domains[nb]):
                if not is_consistent(nb, nb_val, temp, self.graph, self._used):
                    self._domains[nb].remove(nb_val)
                    pruned[nb].append(nb_val)

            if len(self._domains[nb]) == 0:
                self._restore_domains(pruned)
                return None

        return pruned

    def _restore_domains(self, pruned: dict) -> None:
        for nb, vals in pruned.items():
            self._domains[nb].extend(vals)

    # ---- min-conflicts -------------------------------------------------

    def _min_conflicts_repair(self, max_iterations: int = 5000) -> None:
        """
        Min-Conflicts fallback used when the CSP is mathematically over-
        constrained (e.g. 200 residential with only 2 hospitals).

        Two-stage process
        -----------------
        Stage 1 — Relaxed greedy completion:
            Fill all unassigned nodes WITHOUT hard zone restrictions.
            Residential may land outside hospital coverage; these become
            "soft violations" that stage 2 repairs.  Quota must be met first
            so the repair has something to work with.

        Stage 2 — Min-Conflicts iteration:
            Repeatedly pick a conflicted node and reassign it to the value
            that minimises its local conflict count.  Zone restrictions are
            again SOFT (violations are scored, not forbidden) so the solver
            can trade covered ↔ uncovered residential placements freely.

        After repair, print a full diagnosis explaining which rule is violated
        and how many violations remain — this is the "minimum conflict solution"
        report required by the project spec.
        """
        # ------------------------------------------------------------------ #
        # Diagnosis BEFORE repair — identify the conflicting rule             #
        # ------------------------------------------------------------------ #
        needed_res   = QUOTAS[LOC_RESIDENTIAL]
        covered_cap  = len(self._hospital_covered)
        c2_violated  = needed_res > covered_cap
        c2_shortfall = max(0, needed_res - covered_cap)

        needed_pp    = QUOTAS[LOC_POWER_PLANT]
        ind_cap      = len(self._industrial_covered)
        c3_violated  = needed_pp > ind_cap
        c3_shortfall = max(0, needed_pp - ind_cap)

        print("\n" + "=" * 60)
        print("  MIN-CONFLICT SOLUTION REPORT")
        print("=" * 60)
        if c2_violated:
            print(f"  Conflicting Rule : C2 — Proximity (Residential ↔ Hospital)")
            print(f"  Cause            : {QUOTAS[LOC_HOSPITAL]} hospital(s) cover only "
                  f"{covered_cap} cells within {HOSPITAL_HOP_LIMIT} hops.")
            print(f"  Quota demanded   : {needed_res} residential zones")
            print(f"  Coverable cells  : {covered_cap}")
            print(f"  Irreducible viol.: {c2_shortfall} residential outside hospital coverage")
        if c3_violated:
            if c2_violated:
                print()
            print(f"  Conflicting Rule : C3 — Proximity (Power Plant ↔ Industrial)")
            print(f"  Cause            : {QUOTAS[LOC_INDUSTRIAL]} industrial zone(s) cover only "
                  f"{ind_cap} cells within {INDUSTRIAL_HOP_LIMIT} hops.")
            print(f"  Quota demanded   : {needed_pp} power plants")
            print(f"  Coverable cells  : {ind_cap}")
            print(f"  Irreducible viol.: {c3_shortfall} power plants outside industrial coverage")
        print(f"\n  Proposed resolution:")
        if c2_violated:
            print(f"    • Place {needed_res} residential — prefer covered cells, "
                  f"accept {c2_shortfall} violations.")
        if c3_violated:
            print(f"    • Place {needed_pp} power plants — prefer industrial-covered cells, "
                  f"accept {c3_shortfall} violations.")
        print(f"    • All satisfied constraints remain fully enforced.")
        print("=" * 60 + "\n")

        # ------------------------------------------------------------------ #
        # Stage 1 — Relaxed greedy completion (fills quota, allows violations)#
        # ------------------------------------------------------------------ #
        print("[C1]   Stage 1: Relaxed greedy completion (quota-first, soft zones)...")
        self._greedy_complete(relaxed=True)

        # ------------------------------------------------------------------ #
        # Stage 2 — Min-Conflicts iteration                                   #
        # ------------------------------------------------------------------ #
        best      = dict(self._assignment)
        best_conf = total_conflict_count(
            self._assignment, self.graph,
            self._hospital_covered, self._industrial_covered)
        print(f"[C1]   Stage 1 done. Conflicts after greedy: {best_conf}")
        print(f"[C1]   Stage 2: Min-Conflicts repair ({max_iterations} iterations)...")

        no_improve_streak = 0
        MAX_NO_IMPROVE    = 200   # restart random selection after stagnation

        for iteration in range(max_iterations):
            if best_conf == 0:
                break

            # Collect all nodes with at least 1 conflict — never touch anchor nodes
            conflicted = [
                n for n, v in self._assignment.items()
                if n not in self._anchors
                and count_conflicts(n, v, self._assignment, self.graph,
                                    self._hospital_covered, self._industrial_covered) > 0
            ]
            if not conflicted:
                break

            # Pick a random conflicted node
            node    = random.choice(conflicted)
            cur_val = self._assignment[node]

            # Build candidate values: any type that either keeps or frees quota
            # Relaxed: no hard zone restriction — violations are counted, not banned
            candidates = [
                val for val in ALL_LOCATION_TYPES
                if val == cur_val or self._used[val] < QUOTAS[val]
            ]

            best_val   = cur_val
            best_score = count_conflicts(
                node, cur_val, self._assignment, self.graph,
                self._hospital_covered, self._industrial_covered)

            for val in candidates:
                if val == cur_val:
                    continue
                temp  = {**self._assignment, node: val}
                score = count_conflicts(node, val, temp, self.graph,
                                        self._hospital_covered, self._industrial_covered)
                if score < best_score:
                    best_score, best_val = score, val

            # Apply the best reassignment found
            self._assignment[node] = best_val
            self._used[cur_val]   -= 1
            self._used[best_val]  += 1

            total = total_conflict_count(
                self._assignment, self.graph,
                self._hospital_covered, self._industrial_covered)

            if total < best_conf:
                best_conf = total
                best      = dict(self._assignment)
                no_improve_streak = 0
            else:
                no_improve_streak += 1
                if no_improve_streak >= MAX_NO_IMPROVE:
                    # Stagnated — restore best known and try a random restart
                    self._assignment = dict(best)
                    for k, v in self._assignment.items():
                        self._used[k] = 0
                    used_tmp = defaultdict(int)
                    for v in self._assignment.values():
                        used_tmp[v] += 1
                    for k, v in used_tmp.items():
                        self._used[k] = v
                    no_improve_streak = 0

        self._assignment = best

        # ------------------------------------------------------------------ #
        # Final summary report                                                 #
        # ------------------------------------------------------------------ #
        used_final = defaultdict(int)
        for v in self._assignment.values():
            used_final[v] += 1

        res_nodes   = [n for n, v in self._assignment.items() if v == LOC_RESIDENTIAL]
        viol_res    = [n for n in res_nodes if n not in self._hospital_covered]
        covered_res = len(res_nodes) - len(viol_res)

        pp_nodes    = [n for n, v in self._assignment.items() if v == LOC_POWER_PLANT]
        viol_pp     = [n for n in pp_nodes if n not in self._industrial_covered]
        covered_pp  = len(pp_nodes) - len(viol_pp)

        print("\n" + "=" * 60)
        print("  MIN-CONFLICT SOLUTION — FINAL LAYOUT SUMMARY")
        print("=" * 60)
        print(f"  Total conflicts remaining : {best_conf}")
        print()
        if c2_violated:
            print(f"  [C2] Residential placed   : {used_final[LOC_RESIDENTIAL]}/{needed_res}")
            print(f"    ✔ Within hospital coverage : {covered_res}")
            print(f"    ✘ Outside coverage         : {len(viol_res)}  ← irreducible")
            hrs_needed = math.ceil(needed_res / max(covered_cap / max(QUOTAS[LOC_HOSPITAL], 1), 1))
            print(f"    → Add {max(0, hrs_needed - QUOTAS[LOC_HOSPITAL])} more hospital(s) "
                  f"to fully satisfy C2")
        if c3_violated:
            print()
            print(f"  [C3] Power Plants placed  : {used_final[LOC_POWER_PLANT]}/{needed_pp}")
            print(f"    ✔ Within industrial coverage : {covered_pp}")
            print(f"    ✘ Outside coverage            : {len(viol_pp)}  ← irreducible")
            ind_needed = math.ceil(needed_pp / max(ind_cap / max(QUOTAS[LOC_INDUSTRIAL], 1), 1))
            print(f"    → Add {max(0, ind_needed - QUOTAS[LOC_INDUSTRIAL])} more industrial zone(s) "
                  f"to fully satisfy C3")
        # C1 violation count (each pair counted twice — once per node — so halve)
        c1_violations = 0
        for n, v in self._assignment.items():
            adj = [self._assignment.get(nb)
                   for nb in self.graph.grid_neighbors(n)
                   if nb in self._assignment]
            if v == LOC_INDUSTRIAL:
                c1_violations += sum(1 for t in adj if t in _INDUSTRIAL_SENSITIVE)
            elif v in _INDUSTRIAL_SENSITIVE:
                c1_violations += adj.count(LOC_INDUSTRIAL)
        c1_violations //= 2

        print()
        print(f"  [C1] Industrial adjacency violations : {c1_violations}")
        if c1_violations > 0:
            print(f"       Industrial zones touching Hospital or School.")
            print(f"       Occurs when grid is too dense to separate them fully.")
        else:
            print(f"       All Industrial zones properly separated. ✔")
        print()
        print(f"  Hospitals placed     : {used_final[LOC_HOSPITAL]}")
        print(f"  Industrial placed    : {used_final[LOC_INDUSTRIAL]}")
        print(f"  Schools placed       : {used_final[LOC_SCHOOL]}")
        print(f"  Ambulance placed     : {used_final[LOC_AMBULANCE]}")
        print("=" * 60 + "\n")

    def _greedy_complete(self, relaxed: bool = False) -> None:
        """
        Greedy assignment of remaining nodes.

        relaxed=False (normal): hard zone restrictions — residential only in
                                hospital_covered, power plant only in
                                industrial_covered.
        relaxed=True  (min-conflicts fallback): two-pass strategy.
            Pass 1 — covered nodes get RESIDENTIAL first (up to quota).
                     This guarantees min(covered_count, quota) residential
                     land inside hospital coverage — the minimum possible
                     violations.
            Pass 2 — all remaining unassigned nodes filled greedily with
                     whatever type minimises conflicts, no hard zone ban.
        """
        all_unassigned = [n for n in self._covered_nodes + self._uncovered_nodes
                          if n not in self._assignment]

        if relaxed:
            # -------------------------------------------------------------- #
            # PASS 1: saturate covered nodes with residential (quota-limited) #
            # -------------------------------------------------------------- #
            covered_unassigned = [n for n in all_unassigned
                                   if n in self._hospital_covered]
            # Shuffle so placement is varied across runs
            random.shuffle(covered_unassigned)

            for node in covered_unassigned:
                if self._used[LOC_RESIDENTIAL] >= QUOTAS[LOC_RESIDENTIAL]:
                    break   # quota met — stop forcing residential
                # Only assign residential if it doesn't push C1 (industrial adj)
                # Prefer non-conflicting positions but accept them if needed
                adj_industrial = any(
                    self._assignment.get(nb) == LOC_INDUSTRIAL
                    for nb in self.graph.grid_neighbors(node)
                )
                if not adj_industrial:
                    self._assignment[node] = LOC_RESIDENTIAL
                    self._used[LOC_RESIDENTIAL] += 1

            # Second sweep: accept industrial-adjacent covered cells too
            for node in covered_unassigned:
                if node in self._assignment:
                    continue
                if self._used[LOC_RESIDENTIAL] >= QUOTAS[LOC_RESIDENTIAL]:
                    break
                self._assignment[node] = LOC_RESIDENTIAL
                self._used[LOC_RESIDENTIAL] += 1

            # -------------------------------------------------------------- #
            # PASS 1b: industrial-covered nodes -> power plants first          #
            # -------------------------------------------------------------- #
            ind_covered_unassigned = [n for n in all_unassigned
                                       if n not in self._assignment
                                       and n in self._industrial_covered]
            import random as _r; _r.shuffle(ind_covered_unassigned)
            for node in ind_covered_unassigned:
                if self._used[LOC_POWER_PLANT] >= QUOTAS[LOC_POWER_PLANT]:
                    break
                self._assignment[node] = LOC_POWER_PLANT
                self._used[LOC_POWER_PLANT] += 1

            # -------------------------------------------------------------- #
            # PASS 2: fill everything else (covered leftovers + all uncovered)#
            # -------------------------------------------------------------- #
            remaining = [n for n in all_unassigned if n not in self._assignment]
            random.shuffle(remaining)

            for node in remaining:
                best_val, best_score = None, float('inf')
                for val in ALL_LOCATION_TYPES:
                    if self._used[val] >= QUOTAS[val]:
                        continue
                    # C1 soft enforcement: penalise industrial-sensitive placements
                    # adjacent to industrial (and vice versa) — we score rather
                    # than forbid so the solver still makes progress when the grid
                    # is too dense to satisfy C1 everywhere.
                    score = count_conflicts(node, val, self._assignment, self.graph,
                                             self._hospital_covered, self._industrial_covered)
                    # Extra C1 penalty to strongly discourage violations
                    adj = [self._assignment.get(nb) for nb in self.graph.grid_neighbors(node)
                           if nb in self._assignment]
                    if val == LOC_INDUSTRIAL:
                        score += 5 * sum(1 for t in adj if t in _INDUSTRIAL_SENSITIVE)
                    elif val in _INDUSTRIAL_SENSITIVE:
                        score += 5 * adj.count(LOC_INDUSTRIAL)
                    if score < best_score:
                        best_score, best_val = score, val

                if best_val is None:
                    for val in [LOC_RESIDENTIAL, LOC_EMPTY, LOC_SCHOOL,
                                 LOC_AMBULANCE, LOC_POWER_PLANT,
                                 LOC_HOSPITAL, LOC_INDUSTRIAL]:
                        if self._used[val] < QUOTAS[val]:
                            best_val = val
                            break
                if best_val is None:
                    best_val = LOC_EMPTY

                self._assignment[node] = best_val
                self._used[best_val] += 1

        else:
            # -------------------------------------------------------------- #
            # NORMAL mode — hard zone restrictions, single pass               #
            # -------------------------------------------------------------- #
            random.shuffle(all_unassigned)
            for node in all_unassigned:
                best_val, best_score = None, float('inf')
                for val in ALL_LOCATION_TYPES:
                    if self._used[val] >= QUOTAS[val]:
                        continue
                    if val == LOC_RESIDENTIAL and node not in self._hospital_covered:
                        continue
                    if val == LOC_POWER_PLANT and node not in self._industrial_covered:
                        continue
                    score = count_conflicts(node, val, self._assignment, self.graph,
                                             self._hospital_covered, self._industrial_covered)
                    if score < best_score:
                        best_score, best_val = score, val

                if best_val is None:
                    for val in [LOC_EMPTY, LOC_SCHOOL, LOC_AMBULANCE,
                                 LOC_RESIDENTIAL, LOC_POWER_PLANT,
                                 LOC_HOSPITAL, LOC_INDUSTRIAL]:
                        if self._used[val] >= QUOTAS[val]:
                            continue
                        if val == LOC_RESIDENTIAL and node not in self._hospital_covered:
                            continue
                        if val == LOC_POWER_PLANT and node not in self._industrial_covered:
                            continue
                        best_val = val
                        break
                if best_val is None:
                    best_val = LOC_EMPTY

                self._assignment[node] = best_val
                self._used[best_val] += 1


# =============================================================================
# POPULATION DENSITY
# =============================================================================

def _density_for(loc_type: str) -> float:
    """Population density per cell type — used by Challenge 5 K-Means."""
    ranges = {
        LOC_RESIDENTIAL: (800,  1200),
        LOC_HOSPITAL:    (200,   400),
        LOC_SCHOOL:      (300,   600),
        LOC_INDUSTRIAL:  (100,   300),
        LOC_POWER_PLANT: ( 50,   150),
        LOC_AMBULANCE:   ( 30,    80),
        LOC_EMPTY:       (  0,    20),
    }
    lo, hi = ranges.get(loc_type, (0, 0))
    return random.uniform(lo, hi)