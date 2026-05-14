# =============================================================================
# simulation/loop.py — 20-Step Simulation Engine (Phase 14)
# =============================================================================
# Ties all 5 challenges together into one coherent simulation.
#
# Startup sequence (Section 4 of project statement):
#   0a. Challenge 1 — CSP layout (sets node types + population densities)
#   0b. Challenge 2 — Kruskal MST + UCS redundancy (builds roads)
#   0c. Challenge 5 — Crime ML pipeline (sets risk weights)
#   0d. Challenge 3 — GA ambulance placement (initial placement)
#
# Simulation loop (steps 1-20):
#   Each step randomly generates: flood OR emergency OR quiet
#   Every event is logged, graphed, and propagated to all modules.
#
# Integration hooks (per execution flow document Phase 14):
#   crime_monitor.notify_emergency(civilian)  ← after every routed emergency
#   crime_monitor.step()                      ← end of every simulation step
#   ambulance_placer.notify_road_flooded(u,v) ← after every flood event
# =============================================================================

import random
import sys
import os

from config import GRID_ROWS, GRID_COLS, SIM_STEPS
from models.city_graph import CityGraph
from models.city_node import CityNode
from simulation.logger import EventLogger


def run_simulation(
    grid_rows: int = GRID_ROWS,
    grid_cols: int = GRID_COLS,
    sim_steps: int = SIM_STEPS,
    seed: int = None,
) -> tuple:
    """
    Full CityMind simulation — startup + 20 steps.

    Parameters
    ----------
    grid_rows, grid_cols : grid dimensions (default from config.py)
    sim_steps            : number of simulation steps (default 20)
    seed                 : random seed for reproducibility (None = random)

    Returns
    -------
    (graph, logger, ambulance_placer, crime_monitor)
        All live objects — useful for the UI dashboard to read final state.
    """
    if seed is not None:
        random.seed(seed)

    logger = EventLogger()

    print("╔" + "═" * 52 + "╗")
    print("║   CityMind Urban Intelligence System" + " " * 15 + "║")
    print(f"║   Grid: {grid_rows}×{grid_cols}   |   Steps: {sim_steps}" + " " * 26 + "║")
    print("╚" + "═" * 52 + "╝\n")

    # ── STEP 0a: Challenge 1 — Layout ────────────────────────────────────────
    graph = CityGraph(grid_rows, grid_cols)
    try:
        from algorithms.layout_csp import run_layout
        valid = run_layout(graph)
        status = "valid layout" if valid else "min-conflict fallback"
        logger.log(0, "LAYOUT", f"City layout initialized ({grid_rows}×{grid_cols}). {status}.")
    except Exception as e:
        logger.log(0, "ERROR", f"Challenge 1 failed: {e}")
        print(f"[ERROR] Challenge 1: {e}")
        raise

    # ── STEP 0b: Challenge 2 — Roads ─────────────────────────────────────────
    try:
        from algorithms.road_network import run_road_network
        road_result = run_road_network(graph)
        cost = road_result["total_cost"]
        redundancy_ok = road_result["redundancy_ok"]
        r_status = "redundant ✔" if redundancy_ok else "single path ✘"
        logger.log(0, "ROADS",
                   f"Road network built. Total cost: {cost:.1f}. H↔Depot: {r_status}.")
    except Exception as e:
        logger.log(0, "ERROR", f"Challenge 2 failed: {e}")
        print(f"[ERROR] Challenge 2: {e}")
        raise

    # ── STEP 0c: Challenge 5 — Crime ML ──────────────────────────────────────
    from challenge3.placer import AmbulancePlacer
    # FIX 3: Pass logger to AmbulancePlacer
    ambulance_placer = AmbulancePlacer(graph, logger=logger)

    try:
        from challenge5.pipeline import run_crime_risk_pipeline
        # FIX 3: Pass logger to pipeline
        crime_monitor = run_crime_risk_pipeline(graph, ambulance_placer, logger=logger)
        logger.log(0, "RISK",
                   f"Crime risk pipeline complete. Graph v={graph.version}.")
    except Exception as e:
        logger.log(0, "ERROR", f"Challenge 5 failed: {e}")
        print(f"[ERROR] Challenge 5: {e}")
        raise

    # ── STEP 0d: Challenge 3 — Initial Ambulance Placement ───────────────────
    try:
        placement, fitness = ambulance_placer.initial_placement()
        positions = [(n.row, n.col) for n in placement]
        logger.log(0, "AMBULANCE",
                   f"GA placement: {positions}. Worst-case dist: {fitness:.4f}.")
    except Exception as e:
        logger.log(0, "ERROR", f"Challenge 3 GA failed: {e}")
        print(f"[ERROR] Challenge 3: {e}")
        raise

    # ── STEP 0e: Challenge 4 — D* Lite Router Setup ──────────────────────────
    civilians = _generate_civilian_list(graph, count=8)
    router = _build_router(graph, civilians, ambulance_placer, logger)

    # ── SIMULATION LOOP ──────────────────────────────────────────────────────
    for step in range(1, sim_steps + 1):
        print(f"\n{'─'*40}")
        print(f"  SIMULATION STEP {step}/{sim_steps}")
        print(f"{'─'*40}")

        # FIX 5: flood=25%, emergency=50% (2× flood), quiet=25% — matches spec
        event_roll = random.random()
        if event_roll < 0.25:
            event_type = "flood"
        elif event_roll < 0.75:
            event_type = "emergency"
        else:
            event_type = "quiet"

        # ── FLOOD EVENT ──────────────────────────────────────────────────────
        if event_type == "flood":
            pair = _pick_random_road(graph)
            if pair:
                u, v = pair
                try:
                    graph.block_road(u, v)
                    logger.log(step, "FLOOD",
                               f"Road ({u.row},{u.col})↔({v.row},{v.col}) flooded.")
                    print(f"  [FLOOD] Road ({u.row},{u.col})↔({v.row},{v.col}) blocked.")

                    # Notify ambulance placer — may trigger GA re-run
                    ambulance_placer.notify_road_flooded(u, v)

                    # D* Lite replans automatically via graph observer
                    # (DStarLite registers with graph.register_observer() at init)
                except Exception as e:
                    logger.log(step, "ERROR", f"Flood failed: {e}")
            else:
                logger.log(step, "QUIET", "No roads available to flood.")
                print("  [QUIET] No roads available to flood.")

        # ── EMERGENCY EVENT ───────────────────────────────────────────────────
        elif event_type == "emergency":
            result = _route_to_civilian(router, graph, logger, step)
            if result:
                civilian, path, cost = result
                # KEY INTEGRATION: notify Challenge 5 dynamic learning loop
                crime_monitor.notify_emergency(civilian)
                logger.log(step, "ROUTE",
                           f"Emergency at ({civilian.row},{civilian.col}). "
                           f"Path length: {len(path)} hops. Cost: {cost:.4f}.")
                print(f"  [EMERGENCY] Routed to ({civilian.row},{civilian.col}). "
                      f"Cost={cost:.4f}")
            else:
                logger.log(step, "QUIET", "No reachable civilians remaining.")
                print("  [QUIET] No reachable civilians.")

        # ── QUIET STEP ────────────────────────────────────────────────────────
        else:
            logger.log(step, "QUIET", "No events this step.")
            print("  [QUIET] No events.")

        # ── END OF STEP: advance Challenge 5 monitor ─────────────────────────
        # This triggers drift detection every DRIFT_WINDOW steps
        crime_monitor.step()

        # Log current system state
        current_positions = [(n.row, n.col) for n in ambulance_placer.get_current_placement()]
        current_fitness   = ambulance_placer.get_current_fitness()
        logger.log(step, "STATE",
                   f"Ambulances: {current_positions}. "
                   f"Worst-case dist: {current_fitness:.4f}. "
                   f"Graph v={graph.version}.")

    # ── SIMULATION COMPLETE ───────────────────────────────────────────────────
    print("\n╔" + "═" * 52 + "╗")
    print("║          SIMULATION COMPLETE" + " " * 23 + "║")
    print("╚" + "═" * 52 + "╝")
    logger.print_summary()

    # Final placement summary
    final_placement = ambulance_placer.get_current_placement()
    final_fitness   = ambulance_placer.get_current_fitness()
    print(f"\n  Final ambulance positions  : {[(n.row,n.col) for n in final_placement]}")
    print(f"  Final worst-case distance  : {final_fitness:.4f}")
    print(f"  Total road cost            : {graph.total_road_cost():.2f}")
    print(f"  Graph risk version         : {graph.version}\n")

    return graph, logger, ambulance_placer, crime_monitor


# =============================================================================
# HELPERS
# =============================================================================

def _generate_civilian_list(graph: CityGraph, count: int = 8) -> list:
    """Randomly selects Residential nodes to serve as trapped civilians."""
    from config import LOC_RESIDENTIAL
    residential = [
        n for n in graph.all_nodes()
        if n.location_type == LOC_RESIDENTIAL and n.is_accessible
    ]
    if not residential:
        print("[WARN] No residential nodes found — using any accessible node as civilians.")
        residential = [n for n in graph.all_nodes() if n.is_accessible]
    return random.sample(residential, min(count, len(residential)))


def _build_router(graph: CityGraph, civilians: list,
                  ambulance_placer, logger: EventLogger):
    """
    Attempts to build a D* Lite router (Challenge 4 — Ammar's module).
    Falls back to a simple Dijkstra router if dstar_lite is not available.
    """
    try:
        from algorithms.dstar_lite import run_dstar_lite, DStarLite

        class _DStarRouter:
            """Thin wrapper around DStarLite for sequential civilian routing."""
            def __init__(self):
                self.civilians = list(civilians)
                self.current_idx = 0
                self.planner = None
                # Start position: first ambulance's node
                placement = ambulance_placer.get_current_placement()
                self.start = placement[0] if placement else None

            def route_to_next_civilian(self):
                if self.current_idx >= len(self.civilians):
                    return None
                goal = self.civilians[self.current_idx]
                if not goal.is_accessible or self.start is None:
                    self.current_idx += 1
                    return None
                try:
                    planner = run_dstar_lite(graph, self.start, goal, subscribe=False)
                    path = planner.get_path()
                    if not path:
                        self.current_idx += 1
                        return None
                    cost = sum(
                        path[i].effective_cost_to(path[i+1])
                        for i in range(len(path)-1)
                    )
                    self.start = goal
                    self.current_idx += 1
                    return goal, path, cost
                except Exception as e:
                    print(f"  [WARN] D* Lite routing failed: {e}")
                    self.current_idx += 1
                    return None

            def notify_edge_blocked(self, u, v):
                pass  # D* Lite handles this via graph observer

        logger.log(0, "ROUTE", "D* Lite router initialized.")
        return _DStarRouter()

    except Exception as e:
        # Fallback: simple Dijkstra router
        print(f"[INFO] D* Lite not available ({e}). Using Dijkstra fallback router.")
        logger.log(0, "ROUTE", f"Using Dijkstra fallback router ({e}).")
        return _DijkstraRouter(graph, civilians, ambulance_placer)


class _DijkstraRouter:
    """
    Simple Dijkstra-based sequential router used when D* Lite is unavailable.
    Not incremental — re-runs full Dijkstra from scratch on each call.
    """

    def __init__(self, graph: CityGraph, civilians: list, ambulance_placer):
        self.graph            = graph
        self.civilians        = list(civilians)
        self.current_idx      = 0
        self.ambulance_placer = ambulance_placer

    def route_to_next_civilian(self):
        if self.current_idx >= len(self.civilians):
            return None
        goal = self.civilians[self.current_idx]
        if not goal.is_accessible:
            self.current_idx += 1
            return None

        placement = self.ambulance_placer.get_current_placement()
        if not placement:
            return None
        start = placement[0]

        dist, prev = self.graph.dijkstra(start, goal=goal, use_risk=True)
        if goal not in dist or dist[goal] == float('inf'):
            self.current_idx += 1
            return None

        path = self.graph.reconstruct_path(prev, goal)
        cost = dist[goal]
        self.current_idx += 1
        return goal, path, cost

    def notify_edge_blocked(self, u, v):
        pass  # Dijkstra router re-plans from scratch each call anyway


def _route_to_civilian(router, graph: CityGraph,
                        logger: EventLogger, step: int):
    """Calls the router and handles exceptions gracefully."""
    try:
        return router.route_to_next_civilian()
    except Exception as e:
        logger.log(step, "ERROR", f"Routing error: {e}")
        print(f"  [ERROR] Routing: {e}")
        return None


def _pick_random_road(graph: CityGraph):
    """
    Picks a random existing road edge (undirected) to flood.
    Returns (node_u, node_v) or None if no roads exist.
    """
    candidates = []
    seen = set()
    for node in graph.all_nodes():
        for neighbor in node.get_neighbors():
            key = (min(id(node), id(neighbor)), max(id(node), id(neighbor)))
            if key not in seen:
                seen.add(key)
                candidates.append((node, neighbor))
    return random.choice(candidates) if candidates else None