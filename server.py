from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os
import random

from config import GRID_ROWS, GRID_COLS, LOC_RESIDENTIAL
from models.city_graph import CityGraph

from algorithms.layout_csp import run_layout
from algorithms.road_network import run_road_network
from algorithms.dstar_lite import run_dstar_lite
from challenge3.placer import AmbulancePlacer
from challenge5.pipeline import run_crime_risk_pipeline
from challenge5.police import get_police_positions, place_police_officers


app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(BASE_DIR, "ui")

graph = None
monitor = None
placer = None
simulation_step = 0
current_route = []
_blocked_edges: set[tuple[str, str]] = set()  # (from_id, to_id) of blocked roads
_current_planner = None  # Persistent D* Lite planner for Challenge 4
_current_target = None   # Current target node for C4
_mst_edge_set = None     # Persisted after C2 so all serialize calls include MST flags
_extra_edge_set = None   # Persisted after C2 so all serialize calls include redundant flags
_demo_flood_u = None     # Edge chosen in Step 1 to flood in Step 2 (guaranteed alternate path)
_demo_flood_v = None


def _new_graph() -> CityGraph:
    return CityGraph(GRID_ROWS, GRID_COLS)


def _reset_runtime_state(build_baseline: bool = True) -> None:
    global graph, monitor, placer, simulation_step, current_route, _blocked_edges, _current_planner, _current_target, _mst_edge_set, _extra_edge_set, _demo_flood_u, _demo_flood_v
    graph = _new_graph()
    monitor = None
    placer = None
    simulation_step = 0
    current_route = []
    _blocked_edges.clear()
    _current_planner = None
    _current_target = None
    _mst_edge_set = None
    _extra_edge_set = None
    _demo_flood_u = None
    _demo_flood_v = None

    if build_baseline:
        # Only initialize the graph structure — C1 and C2 run on-demand via UI
        node_count = len(list(graph.all_nodes()))
        print(f"[SERVER] Graph initialized: {node_count} nodes (run C1 from UI to build layout)")


def _node_id(node) -> str:
    return f"N_{str(node.row).zfill(2)}_{str(node.col).zfill(2)}"


def _risk_value(node) -> float:
    multiplier = float(getattr(node, "risk_multiplier", 1.0))
    if multiplier >= 1.5:
        return 1.0
    if multiplier >= 1.2:
        return 0.55
    return 0.15


def _serialize_node(node) -> dict:
    return {
        "id": _node_id(node),
        "row": node.row,
        "col": node.col,
        "type": node.location_type,
        "pop": int(getattr(node, "population_density", 0)),
        "risk": getattr(node, "risk_level", "Low"),
        "riskVal": _risk_value(node),
        "accessible": getattr(node, "is_accessible", True),
        "clusterId": int(getattr(node, "cluster_id", 0)),
        "ambulanceId": getattr(node, "ambulance_id", None),
        "ambulanceHere": getattr(node, "ambulance_id", None) is not None,
        "predictedRisk": getattr(node, "predicted_risk", "Low"),
        "totalEmergencies": float(getattr(node, "total_emergencies", 0.0)),
        "justUpdated": bool(getattr(node, "just_updated", False)),
        "costMult": float(getattr(node, "risk_multiplier", 1.0)),
    }


def _serialize_graph(mst_edge_set=None, extra_edge_set=None, hospital=None, depots=None) -> dict:
    # Fall back to persisted edge sets if not explicitly provided
    if mst_edge_set is None:
        mst_edge_set = _mst_edge_set
    if extra_edge_set is None:
        extra_edge_set = _extra_edge_set
    nodes = []
    edges = []
    edge_map = {}
    seen = set()
    edge_states: dict[str, str] = {}  # edge_id -> state
    
    # Convert edge sets to sets of tuples for proper lookup (they may be lists from JSON)
    if mst_edge_set is not None:
        mst_edge_set = set(tuple(tuple(inner) for inner in x) for x in mst_edge_set)
    if extra_edge_set is not None:
        extra_edge_set = set(tuple(tuple(inner) for inner in x) for x in extra_edge_set)
    
    # Mark flooded/blocked edges
    for from_id, to_id in _blocked_edges:
        edge_id = f"E_{from_id}_{to_id}"
        edge_id_rev = f"E_{to_id}_{from_id}"
        edge_states[edge_id] = "flooded"
        edge_states[edge_id_rev] = "flooded"
    
    for node in graph.all_nodes():
        nodes.append(_serialize_node(node))
        for neighbor, weight in node.get_neighbors().items():
            # Use coordinate-based key for stable edge identification
            coord_key = tuple(sorted([(node.row, node.col), (neighbor.row, neighbor.col)]))
            if coord_key in seen:
                continue
            seen.add(coord_key)
            # Determine if this edge is MST or redundant using coordinate tuple
            is_mst = mst_edge_set is not None and coord_key in mst_edge_set
            is_redundant = extra_edge_set is not None and coord_key in extra_edge_set
            edge_id = f"E_{_node_id(node)}_{_node_id(neighbor)}"
            # Note: removed verbose per-edge logging for redundant edges
            # Check if blocked, otherwise standard
            state = edge_states.get(edge_id, "standard")
            edge = {
                "id": edge_id,
                "fromId": _node_id(node),
                "toId": _node_id(neighbor),
                "cost": float(weight),
                "eff": float(node.effective_cost_to(neighbor)),
                "state": state,
                "isMST": is_mst,
                "isRedundant": is_redundant,
            }
            edges.append(edge)
            edge_map[edge["id"]] = edge

    # Add flooded/blocked edges that were removed from graph but need UI display
    for from_id, to_id in _blocked_edges:
        edge_id = f"E_{from_id}_{to_id}"
        # Parse node coords from IDs (format: N_RR_CC)
        _, r1, c1 = from_id.split("_")
        _, r2, c2 = to_id.split("_")
        coord_key = tuple(sorted([(int(r1), int(c1)), (int(r2), int(c2))]))
        if coord_key in seen:
            continue  # Already serialized (shouldn't happen but safety check)
        seen.add(coord_key)
        is_mst = mst_edge_set is not None and coord_key in mst_edge_set
        is_redundant = extra_edge_set is not None and coord_key in extra_edge_set
        edge = {
            "id": edge_id,
            "fromId": from_id,
            "toId": to_id,
            "cost": 1.0,
            "eff": 1.0,
            "state": "flooded",
            "isMST": is_mst,
            "isRedundant": is_redundant,
        }
        edges.append(edge)
        edge_map[edge["id"]] = edge

    ambulances = []
    placed = [n for n in graph.all_nodes() if getattr(n, "ambulance_id", None) is not None]
    for idx in range(3):
        node = next((n for n in placed if getattr(n, "ambulance_id", None) == idx), None)
        ambulances.append(
            {
                "id": idx,
                "nodeId": _node_id(node) if node else None,
                "status": "STANDBY" if node else "UNPLACED",
                "coverage": 0,
            }
        )

    result = {
        "rows": graph.rows,
        "cols": graph.cols,
        "version": graph.version,
        "nodes": nodes,
        "nodeMap": {node["id"]: node for node in nodes},
        "edges": edges,
        "edgeMap": edge_map,
        "ambulances": ambulances,
        "police": get_police_positions(graph),
        "summary": graph.summary(),
        "riskSummary": {
            "High": sum(1 for n in nodes if n["risk"] == "High"),
            "Medium": sum(1 for n in nodes if n["risk"] == "Medium"),
            "Low": sum(1 for n in nodes if n["risk"] == "Low"),
        },
        "simulationStep": simulation_step,
        "route": current_route,
    }
    if hospital:
        result["hospital"] = hospital
    if depots:
        result["depots"] = depots
    return result


def _clear_update_flags() -> None:
    if graph is not None:
        graph.clear_just_updated_flags()


def _pick_random_road():
    candidates = []
    seen = set()
    for node in graph.all_nodes():
        for neighbor in node.get_neighbors():
            key = tuple(sorted(((node.row, node.col), (neighbor.row, neighbor.col))))
            if key in seen:
                continue
            seen.add(key)
            candidates.append((node, neighbor))
    return random.choice(candidates) if candidates else None


def _pick_random_target():
    targets = [n for n in graph.all_nodes() if n.location_type == LOC_RESIDENTIAL and n.is_accessible]
    if not targets:
        targets = [n for n in graph.all_nodes() if n.is_accessible]
    return random.choice(targets) if targets else None


def _pick_target_with_two_paths(placement: list) -> tuple:
    """
    Find a target node that has at least 2 edge-disjoint paths from any
    ambulance, AND return the mid-route edge that sits only on the PRIMARY
    path (so flooding it forces D* Lite to provably reroute).

    Returns (target_node, flood_u, flood_v) or (None, None, None) if not found.

    Strategy:
      For each candidate target, build a D* Lite plan from the nearest ambulance.
      Then temporarily remove the mid-path edge and check if a second path still
      exists (BFS). If yes, that target qualifies and that edge is our flood edge.
    """
    import math

    def _bfs_reachable(start, goal, blocked_u, blocked_v):
        """BFS ignoring the blocked edge — returns True if goal reachable."""
        from collections import deque
        visited = {start}
        q = deque([start])
        while q:
            node = q.popleft()
            if node is goal:
                return True
            for nb in node.get_neighbors():
                if nb in visited:
                    continue
                # Skip the blocked edge in both directions
                if (node is blocked_u and nb is blocked_v) or \
                   (node is blocked_v and nb is blocked_u):
                    continue
                if not nb.is_accessible:
                    continue
                visited.add(nb)
                q.append(nb)
        return False

    # Residential nodes are the natural targets
    candidates = [n for n in graph.all_nodes()
                  if n.location_type == LOC_RESIDENTIAL and n.is_accessible]
    random.shuffle(candidates)

    for target in candidates:
        # Find nearest ambulance path via D* Lite
        best_path = []
        best_cost = math.inf
        for amb in placement:
            planner = run_dstar_lite(graph, amb, target, subscribe=False)
            path = planner.get_path()
            if len(path) < 3:
                continue
            cost = sum(path[i].effective_cost_to(path[i+1]) for i in range(len(path)-1))
            if cost < best_cost:
                best_cost = cost
                best_path = path

        if len(best_path) < 3:
            continue  # path too short to flood mid-edge meaningfully

        # Try edges from the middle of the path outward — flood the one that
        # still leaves an alternate route available
        mid = len(best_path) // 2
        for offset in range(len(best_path) - 1):
            idx = (mid + offset) % (len(best_path) - 1)
            u = best_path[idx]
            v = best_path[idx + 1]
            # Check: if we remove this edge, can we still reach target from start?
            if _bfs_reachable(best_path[0], target, u, v):
                print(f"[D* SETUP] Found target ({target.row},{target.col}) with "
                      f"alternate path. Flood edge: ({u.row},{u.col})↔({v.row},{v.col})")
                return target, u, v

    print("[D* SETUP] No target with guaranteed alternate path found — falling back to random")
    return None, None, None


def _replace_planner(new_planner):
    """Unregister old planner from graph observer, then set new planner."""
    global _current_planner
    if _current_planner is not None and graph is not None:
        _current_planner.unregister_from_graph(graph)
        print(f"[D* DEBUG] Unregistered old planner from graph observer")
    _current_planner = new_planner


def _nearest_ambulance(placement: list, target, subscribe: bool = True) -> tuple:
    """Return (nearest_node, planner) by comparing path costs from each ambulance.
    Runs one D* Lite per ambulance (typically 3) and picks the cheapest."""
    import math
    best_amb, best_planner, best_cost = None, None, math.inf
    for amb in placement:
        planner = run_dstar_lite(graph, amb, target, subscribe=False)
        path = planner.get_path()
        cost = sum(path[i].effective_cost_to(path[i+1]) for i in range(len(path)-1)) if len(path) > 1 else math.inf
        print(f"[DISPATCH] Ambulance ({amb.row},{amb.col}) -> target ({target.row},{target.col}): cost={cost:.2f}")
        if cost < best_cost:
            best_amb, best_planner, best_cost = amb, planner, cost
    # Re-create the winner with subscribe if needed (so it auto-repairs on floods)
    if subscribe and best_amb is not None:
        best_planner = run_dstar_lite(graph, best_amb, target, subscribe=True)
    print(f"[DISPATCH] Winner: ambulance ({best_amb.row},{best_amb.col}) with cost {best_cost:.2f}")
    return best_amb, best_planner


def _ensure_placer():
    global placer
    if placer is None:
        placer = AmbulancePlacer(graph)

        # Register callback so when C3 re-runs, we update the D* planner
        def on_placement_changed(new_placement, new_fitness):
            global _current_target, current_route
            print(f"[D* INTEGRATION] C3 placement changed, updating D* planner...")
            _replace_planner(None)  # Unregister old planner, force recreation on next flood/event
            _current_target = None
            current_route = []
            print(f"[D* INTEGRATION] Planner cleared. Will recreate on next flood/emergency.")

        placer.on_placement_changed = on_placement_changed
    return placer


def _run_challenge(challenge_num: int):
    global monitor, current_route
    if challenge_num == 1:
        result = run_layout(graph)
        payload = {
            "ok": True,
            "challenge": 1,
            "success": result["success"],
            "violations": result.get("violations", {"c1": 0, "c2": 0, "c3": 0, "total": 0}),
            "graph": _serialize_graph()
        }
        _clear_update_flags()
        return payload
    if challenge_num == 2:
        global _mst_edge_set, _extra_edge_set
        result = run_road_network(graph)
        _mst_edge_set = result.get("mst_edge_set")
        _extra_edge_set = result.get("extra_edge_set")
        payload = {
            "ok": True,
            "challenge": 2,
            "totalCost": result["total_cost"],
            "redundancyOk": result["redundancy_ok"],
            "mstEdges": result["mst_edges"],
            "extraEdges": result["extra_edges"],
            "graph": _serialize_graph(
                mst_edge_set=result.get("mst_edge_set"),
                extra_edge_set=result.get("extra_edge_set"),
                hospital=result.get("hospital"),
                depots=result.get("depots")
            ),
        }
        _clear_update_flags()
        return payload
    if challenge_num == 3:
        pl = _ensure_placer()
        placement, fitness = pl.initial_placement()
        current_route = []
        # JSON doesn't support Infinity - convert to None or a large number
        if fitness == float('inf') or fitness == float('-inf'):
            fitness_json = None  # or use a large number like 9999.99
        else:
            fitness_json = fitness
        payload = {
            "ok": True,
            "challenge": 3,
            "fitness": fitness_json,
            "positions": [
                {"id": i, "nodeId": _node_id(node)} for i, node in enumerate(placement)
            ],
            "graph": _serialize_graph(),
        }
        _clear_update_flags()
        return payload
    if challenge_num == 4:
        global _current_planner, _current_target
        pl = _ensure_placer()
        placement = pl.get_current_placement()
        if not placement:
            return {"ok": False, "challenge": 4, "message": "Run C3 first.", "graph": _serialize_graph()}
        target = _pick_random_target()
        if target is None:
            return {"ok": False, "challenge": 4, "message": "No target available.", "graph": _serialize_graph()}
        # Pick nearest ambulance using D* Lite g-values, then build persistent planner
        _nearest, new_planner = _nearest_ambulance(placement, target)
        _replace_planner(new_planner)
        _current_target = target
        path = _current_planner.get_path()
        current_route = [_node_id(node) for node in path]
        if monitor is not None:
            monitor.notify_emergency(target)
        payload = {
            "ok": True,
            "challenge": 4,
            "target": _node_id(target),
            "path": current_route,
            "cost": sum(path[i].effective_cost_to(path[i + 1]) for i in range(len(path) - 1)) if len(path) > 1 else 0.0,
            "graph": _serialize_graph(),
        }
        _clear_update_flags()
        return payload
    if challenge_num == 5:
        pl = _ensure_placer()
        monitor = run_crime_risk_pipeline(graph, pl)
        # Collect initial risk predictions from C5 to show in UI
        # Don't clear flags yet — Step 1 will collect them as risk events
        risk_updates = []
        for node in graph.all_nodes():
            if getattr(node, "just_updated", False):
                risk_updates.append({
                    "type": "risk",
                    "nodeId": _node_id(node),
                    "risk": node.predicted_risk,
                })
        police_positions = get_police_positions(graph)
        print(f"[C5] Initial predictions applied to {len(risk_updates)} nodes")
        print(f"[C5] Police officers deployed: {len(police_positions)}")
        payload = {
            "ok": True,
            "challenge": 5,
            "riskUpdates": risk_updates,
            "policeCount": len(police_positions),
            "graph": _serialize_graph()
        }
        # Note: Flags NOT cleared here — first sim step will emit these as events
        return payload
    return {"ok": False, "message": f"Unknown challenge {challenge_num}"}


def _do_flood(u, v) -> list:
    """Block a road between u and v, notify placer, and auto-reroute if the
    active route passes through the now-impassable edge. Returns a list of
    event dicts to append to the step payload."""
    global current_route, _current_planner, _current_target
    events = []

    print(f"[D* DEBUG] Flooding edge: {_node_id(u)} -> {_node_id(v)}")
    graph.block_road(u, v)
    from_id = _node_id(u)
    to_id   = _node_id(v)
    _blocked_edges.add((from_id, to_id))
    edge_id = f"E_{from_id}_{to_id}"
    print(f"[FLOOD EVENT] edge_id={edge_id}, from=({u.row},{u.col}), to=({v.row},{v.col})")
    events.append({"type": "road", "edgeId": edge_id, "state": "flooded"})

    if placer is not None:
        placer.notify_road_flooded(u, v)

    # ── Auto-reroute if the active route crosses the flooded edge ──────────
    print(f"[D* DEBUG] current_route: {current_route}")
    print(f"[D* DEBUG] _current_planner: {_current_planner}")
    print(f"[D* DEBUG] _current_target: {_node_id(_current_target) if _current_target else None}")

    if current_route and len(current_route) >= 2:
        route_edges = set(zip(current_route, current_route[1:]))
        route_edges |= {(b, a) for a, b in route_edges}  # undirected
        flooded_pair = (from_id, to_id)
        rev_pair     = (to_id, from_id)
        print(f"[D* DEBUG] route_edges: {route_edges}")
        print(f"[D* DEBUG] flooded_pair: {flooded_pair}, rev_pair: {rev_pair}")

        if flooded_pair in route_edges or rev_pair in route_edges:
            print("[D* DEBUG] Flooded edge IS on current route! Triggering reroute...")
            target = _current_target
            pl2 = placer
            # If no active target, fall back to a random one
            if target is None and pl2 and pl2.get_current_placement():
                target = _pick_random_target()

            if target is not None and pl2 and pl2.get_current_placement():
                try:
                    old_path = current_route[:]

                    # ── Unregister old planner BEFORE creating the new one ────
                    # The graph is already updated (block_road ran above).
                    # Unregistering prevents the old planner's observer from
                    # partially running during the new planner's initialize().
                    _replace_planner(None)

                    # ── Use _nearest_ambulance on the POST-FLOOD graph ────────
                    # After a flood, a different ambulance may now have the
                    # shortest alternate path. _nearest_ambulance runs D* Lite
                    # from every ambulance on the current (post-flood) graph and
                    # picks the one with the lowest cost path to the target.
                    best_amb, new_planner = _nearest_ambulance(
                        pl2.get_current_placement(), target, subscribe=True
                    )
                    _replace_planner(new_planner)
                    _current_target = target

                    path = new_planner.get_path()
                    current_route = [_node_id(n) for n in path]
                    print(f"[D* DEBUG] Old path ({len(old_path)} hops): {old_path[:4]}...")
                    print(f"[D* DEBUG] New path ({len(path)} hops) via amb "
                          f"({best_amb.row},{best_amb.col}): {current_route[:4]}...")

                    if not path:
                        events.append({
                            "type":    "reroute_blocked",
                            "trigger": "flood",
                            "path":    [],
                            "cost":    0.0,
                            "target":  _node_id(target),
                            "message": "No alternate route — all paths flooded",
                        })
                        print("[D* DEBUG] No alternate route — target unreachable")
                    else:
                        cost = sum(
                            path[i].effective_cost_to(path[i + 1])
                            for i in range(len(path) - 1)
                        ) if len(path) > 1 else 0.0
                        events.append({
                            "type":    "reroute",
                            "trigger": "flood",
                            "path":    current_route,
                            "cost":    cost,
                            "target":  _node_id(target),
                        })
                        print(f"[D* DEBUG] Reroute SUCCESS — {len(path)} hops, cost {cost:.2f}")
                except Exception as exc:
                    print(f"[D* DEBUG] Reroute FAILED: {exc}")
                    import traceback
                    traceback.print_exc()
                    events.append({
                        "type":   "reroute_failed",
                        "trigger": "flood",
                        "reason":  str(exc),
                    })
            else:
                print("[D* DEBUG] No target or placer available — flood recorded, no reroute")
        else:
            print("[D* DEBUG] Flooded edge NOT on current route - no reroute needed")
    else:
        print(f"[D* DEBUG] No current route to check (current_route={current_route})")
    return events


def _advance_simulation_step():
    """
    20-step simulation demonstrating all 5 challenges:
      C1 — city layout (shared graph state)
      C2 — road network (MST + redundant edges for rerouting)
      C3 — ambulance placement (re-evaluated on risk shifts)
      C4 — D* Lite dynamic routing with flood rerouting
      C5 — crime risk predictions update weights & trigger C3 re-runs
    """
    global simulation_step, current_route, _current_planner, _current_target
    simulation_step += 1
    step = simulation_step
    events = []
    event_type = "quiet"
    print(f"\n{'─'*50}")
    print(f"[SIM] ══ Step {step}/20 ══")

    # ── C5: Risk monitoring runs every step ──────────────────────────
    # Collect initial risk predictions from C5 (set when C5 ran, flags not cleared)
    # Also collect any new drift detection updates
    risk_events = []
    for node in graph.all_nodes():
        if getattr(node, "just_updated", False):
            risk_events.append({
                "type": "risk",
                "nodeId": _node_id(node),
                "risk": node.predicted_risk,
            })
            node.just_updated = False
    if risk_events:
        print(f"[C5] {len(risk_events)} risk predictions applied (initial or drift)")
        events.extend(risk_events)
    # Run drift detection (may trigger retrain and new risk updates for next step)
    if monitor is not None:
        monitor.step()

    # ── Check prerequisites ───────────────────────────────────────────
    pl = placer
    has_placement = pl is not None and len(pl.get_current_placement()) > 0
    if not has_placement:
        print(f"[SIM] ⚠ C3 not run yet — no ambulance placement. Run Challenge 3 first.")
        events.append({"type": "quiet", "message": "C3 placement required"})
        payload = {
            "ok": True, "step": simulation_step, "eventType": "quiet",
            "events": events, "graph": _serialize_graph(),
        }
        return payload

    # ── Scripted demo steps ──────────────────────────────────────────
    # Step 1: Emergency dispatch → C3+C4 (nearest ambulance + D* route)
    # Step 2: Flood ON active route → C4 reroute demonstration
    # Step 3+: Random events with proper probabilities

    if step == 1:
        event_type = "emergency"
        # Pick a target that provably has an alternate path so Step 2 can
        # flood one edge and D* Lite will visibly reroute via the other.
        target, _s2_flood_u, _s2_flood_v = _pick_target_with_two_paths(
            pl.get_current_placement()
        )
        # Store the chosen flood edge so Step 2 uses exactly this edge
        global _demo_flood_u, _demo_flood_v
        _demo_flood_u = _s2_flood_u
        _demo_flood_v = _s2_flood_v
        if target is None:
            target = _pick_random_target()  # graceful fallback
        if target is not None:
            events.append({"type": "emergency", "nodeId": _node_id(target)})
            if monitor is not None:
                monitor.notify_emergency(target)
            best_amb, new_planner = _nearest_ambulance(
                pl.get_current_placement(), target, subscribe=True
            )
            _replace_planner(new_planner)
            _current_target = target
            path = _current_planner.get_path()
            current_route = [_node_id(n) for n in path]
            cost = sum(path[i].effective_cost_to(path[i+1]) for i in range(len(path)-1)) if len(path) > 1 else 0.0
            events.append({"type": "route", "path": current_route, "cost": cost})
            print(f"[SIM] 🚨 Emergency at ({target.row},{target.col})")
            print(f"[SIM]   C3: Dispatched ambulance ({best_amb.row},{best_amb.col})")
            print(f"[SIM]   C4: D* Lite computed route → {len(path)} hops, cost {cost:.1f}")
            if _demo_flood_u:
                print(f"[SIM]   C4: Step 2 will flood ({_demo_flood_u.row},{_demo_flood_u.col})"
                      f"↔({_demo_flood_v.row},{_demo_flood_v.col}) — alternate path confirmed ✔")
        else:
            events.append({"type": "quiet", "message": "No valid target"})

    elif step == 2:
        event_type = "flood"
        # Use the flood edge chosen in Step 1 (guaranteed alternate path exists)
        u = getattr(graph, '__s2u', None)  # fallback attr (won't exist)
        v = getattr(graph, '__s2v', None)
        # Use module-level vars set in Step 1
        u = globals().get('_demo_flood_u')
        v = globals().get('_demo_flood_v')

        if u is not None and v is not None and u.has_road(v):
            print(f"[SIM] 🌊 Flooding pre-chosen edge ON active route: "
                  f"({u.row},{u.col})↔({v.row},{v.col})")
            print(f"[SIM]   C4: Alternate path confirmed — D* Lite will reroute...")
            flood_events = _do_flood(u, v)
            events.extend(flood_events)
            print(f"[SIM]   C4: Reroute events: {[e['type'] for e in flood_events]}")
        elif current_route and len(current_route) >= 3:
            # Fallback: flood mid-edge of current route
            mid = len(current_route) // 2
            from_id = current_route[mid]
            to_id   = current_route[mid + 1]
            _, r1, c1 = from_id.split("_")
            _, r2, c2 = to_id.split("_")
            u = graph.node(int(r1), int(c1))
            v = graph.node(int(r2), int(c2))
            print(f"[SIM] 🌊 Flooding mid-route edge (fallback): {from_id} → {to_id}")
            flood_events = _do_flood(u, v)
            events.extend(flood_events)
        else:
            print(f"[SIM] ⚠ No active route from Step 1 — falling back to random flood")
            pair = _pick_random_road()
            if pair:
                u, v = pair
                print(f"[SIM] 🌊 Random flood ({u.row},{u.col})↔({v.row},{v.col})")
                events += _do_flood(u, v)

    else:
        # Random events with clear challenge attribution
        event_type = random.choices(["flood", "emergency", "quiet"], weights=[35, 50, 15])[0]

        if event_type == "flood":
            # Prefer the pre-verified flood edge (from _pick_target_with_two_paths)
            # if it's still in the graph and on the active route.
            # This guarantees D* Lite demonstrates a real reroute every time.
            u, v = None, None
            demo_u = globals().get('_demo_flood_u')
            demo_v = globals().get('_demo_flood_v')
            if (demo_u is not None and demo_v is not None
                    and demo_u.has_road(demo_v)
                    and current_route and len(current_route) >= 2):
                # Check the demo edge is still on the current route
                route_edges = set(zip(current_route, current_route[1:]))
                route_edges |= {(b, a) for a, b in route_edges}
                du_id, dv_id = _node_id(demo_u), _node_id(demo_v)
                if (du_id, dv_id) in route_edges or (dv_id, du_id) in route_edges:
                    u, v = demo_u, demo_v
                    print(f"[SIM] 🌊 Flooding pre-verified edge on active route: "
                          f"({u.row},{u.col})↔({v.row},{v.col}) — alternate path confirmed ✔")
            if u is None or v is None:
                pair = _pick_random_road()
                if pair:
                    u, v = pair
                    print(f"[SIM] 🌊 Random flood on C2 road: ({u.row},{u.col})↔({v.row},{v.col})")
                else:
                    event_type = "quiet"
                    events.append({"type": "quiet", "message": "No road to flood"})
            if u is not None and v is not None:
                print(f"[SIM]   C4: D* Lite will reroute if this blocks active path...")
                events += _do_flood(u, v)

        elif event_type == "emergency":
            # Always try to find a target with an alternate path so that if a
            # flood hits this route the D* Lite reroute is provably demonstrable.
            target, flood_u, flood_v = _pick_target_with_two_paths(pl.get_current_placement())
            if target is None:
                target = _pick_random_target()   # graceful fallback
                flood_u = flood_v = None
            if target is not None:
                # Store the pre-verified flood edge so any subsequent flood on
                # this route has a guaranteed alternate path.
                _demo_flood_u = flood_u
                _demo_flood_v = flood_v
                events.append({"type": "emergency", "nodeId": _node_id(target)})
                print(f"[SIM] 🚨 Emergency at ({target.row},{target.col})")
                if flood_u:
                    print(f"[SIM]   Alternate path confirmed — flood edge "
                          f"({flood_u.row},{flood_u.col})↔({flood_v.row},{flood_v.col}) stored")
                if monitor is not None:
                    monitor.notify_emergency(target)
                best_amb, new_planner = _nearest_ambulance(
                    pl.get_current_placement(), target, subscribe=True
                )
                _replace_planner(new_planner)
                _current_target = target
                path = new_planner.get_path()
                current_route = [_node_id(n) for n in path]
                cost = sum(path[i].effective_cost_to(path[i+1]) for i in range(len(path)-1)) if len(path) > 1 else 0.0
                events.append({"type": "route", "path": current_route, "cost": cost})
                print(f"[SIM]   C3: Nearest ambulance ({best_amb.row},{best_amb.col})")
                print(f"[SIM]   C4: D* route → {len(path)} hops, cost {cost:.1f}")
            else:
                events.append({"type": "quiet", "message": "No valid target"})

        else:
            print(f"[SIM] — Quiet step (all 5 challenges monitoring)")
            events.append({"type": "quiet", "message": "No event"})

    print(f"[SIM] Step {step} complete: {len(events)} events: {[e['type'] for e in events]}")

    payload = {
        "ok": True,
        "step": simulation_step,
        "eventType": event_type,
        "events": events,
        "graph": _serialize_graph(),
    }
    _clear_update_flags()
    return payload


@app.route("/")
def index():
    return send_from_directory(UI_DIR, "index.html")


@app.route("/dashboard.html")
def dashboard():
    return send_from_directory(UI_DIR, "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/health", methods=["GET", "OPTIONS"])
def api_health():
    graph_ok = graph is not None and hasattr(graph, 'version')
    node_count = len(list(graph.all_nodes())) if graph_ok else 0
    edge_count = len(_blocked_edges) if graph_ok else 0
    return jsonify({
        "status": "ok",
        "graph_ready": graph_ok,
        "nodes": node_count,
        "blocked_edges": edge_count,
        "step": simulation_step
    })


@app.route("/api/graph", methods=["GET"])
def api_graph():
    payload = _serialize_graph()
    _clear_update_flags()
    return jsonify(payload)


@app.route("/api/challenge/<int:challenge_num>/run", methods=["POST"])
def api_run_challenge(challenge_num: int):
    try:
        return jsonify(_run_challenge(challenge_num))
    except Exception as exc:
        return jsonify({"ok": False, "challenge": challenge_num, "error": str(exc)}), 500


@app.route("/api/simulation/step", methods=["POST"])
def api_step():
    try:
        return jsonify(_advance_simulation_step())
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/flood", methods=["POST"])
def api_flood():
    """Flood a specific edge (from_id, to_id) or a random one if none supplied.
    Always re-routes immediately if the active path is affected."""
    from flask import request as flask_request
    try:
        body      = flask_request.get_json(silent=True) or {}
        from_id   = body.get("fromId")
        to_id     = body.get("toId")

        # Try to resolve the specific edge from the payload
        u, v = None, None
        if from_id and to_id:
            # Parse N_RR_CC format
            def parse_id(nid):
                _, r, c = nid.split("_")
                return graph.node(int(r), int(c))
            try:
                u = parse_id(from_id)
                v = parse_id(to_id)
                print(f"[D* DEBUG] Parsed flood target: ({u.row},{u.col}) -> ({v.row},{v.col})")
            except Exception as e:
                print(f"[D* DEBUG] Failed to parse flood IDs ({from_id}, {to_id}): {e}")
                u, v = None, None

        if u is None or v is None:
            pair = _pick_random_road()
            if pair is None:
                return jsonify({"ok": False, "message": "No roads to flood"}), 400
            u, v = pair

        events = _do_flood(u, v)
        payload = {
            "ok":     True,
            "events": events,
            "graph":  _serialize_graph(),
        }
        _clear_update_flags()
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/demo_route", methods=["POST"])
def api_demo_route():
    """
    Pick a target with a guaranteed alternate path and compute the D* Lite route.
    Returns { ok, path, floodFrom, floodTo, cost } where floodFrom/floodTo is
    a mid-route edge whose removal still leaves an alternate route to the target.
    Called exclusively by the UI's demoReroute() button.
    """
    global _current_planner, _current_target, current_route, _demo_flood_u, _demo_flood_v
    try:
        pl = placer
        if pl is None or not pl.get_current_placement():
            return jsonify({"ok": False, "message": "Run C3 first to place ambulances."}), 400

        target, flood_u, flood_v = _pick_target_with_two_paths(pl.get_current_placement())
        if target is None:
            # Fallback: any target, any mid-route edge
            target = _pick_random_target()
            flood_u = flood_v = None

        if target is None:
            return jsonify({"ok": False, "message": "No reachable target found."}), 400

        best_amb, new_planner = _nearest_ambulance(pl.get_current_placement(), target, subscribe=True)
        _replace_planner(new_planner)
        _current_target = target
        _demo_flood_u   = flood_u
        _demo_flood_v   = flood_v

        path = _current_planner.get_path()
        current_route = [_node_id(n) for n in path]
        cost = sum(path[i].effective_cost_to(path[i+1]) for i in range(len(path)-1)) if len(path) > 1 else 0.0

        # If server found a specific flood edge, use it; else pick mid-route
        if flood_u is not None and flood_v is not None:
            flood_from = _node_id(flood_u)
            flood_to   = _node_id(flood_v)
        elif len(path) >= 3:
            mid = len(path) // 2
            flood_from = _node_id(path[mid - 1])
            flood_to   = _node_id(path[mid])
        else:
            flood_from = flood_to = None

        print(f"[DEMO_ROUTE] target=({target.row},{target.col}), "
              f"path={len(path)} hops, flood={flood_from}↔{flood_to}")

        payload = {
            "ok":        True,
            "path":      current_route,
            "cost":      cost,
            "target":    _node_id(target),
            "floodFrom": flood_from,
            "floodTo":   flood_to,
            "graph":     _serialize_graph(),
        }
        _clear_update_flags()
        return jsonify(payload)
    except Exception as exc:
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/simulation/reset", methods=["POST"])
def api_reset():
    try:
        _reset_runtime_state(build_baseline=True)
        payload = {"ok": True, "graph": _serialize_graph()}
        _clear_update_flags()
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


_reset_runtime_state(build_baseline=True)


if __name__ == '__main__':
    app.run(debug=True, port=5000)