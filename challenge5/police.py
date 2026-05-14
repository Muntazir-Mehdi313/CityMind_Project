# =============================================================================
# challenge5/police.py — Police Officer Placement (Challenge 5 Extension)
# =============================================================================
# OWNER: Muntazir Mehdi (24I-0847)
#
# WHAT THIS DOES:
#   After the Decision Tree classifies all nodes with risk labels, this module
#   identifies the optimal positions for police officers — nodes where deploying
#   a police presence will have the highest crime-reduction impact.
#
# PLACEMENT STRATEGY: Weighted Coverage Maximization
#   Score each node by:
#       score = risk_weight × population_density × (1 + emergency_spillover)
#
#   risk_weight:
#       "High"   → 3.0
#       "Medium" → 1.5
#       "Low"    → 0.0  (Low-risk nodes are never selected)
#
#   Then greedily select `num_officers` nodes that collectively maximize
#   coverage, with a minimum-distance spread constraint so officers are
#   not clustered at the same corner of the city.
#
# INTEGRATION:
#   Called from pipeline.py at the end of Phase C (after predictions applied).
#   Writes node.police_here = True on selected nodes.
#   Returns a list of dicts ready to be serialised by server.py and read by
#   the UI's police officer overlay.
#
# CALLED BY:
#   challenge5/pipeline.py::run_crime_risk_pipeline()
#   algorithms/crime_ml.py::run_crime_pipeline()   (standalone)
#   server.py /api/challenge/5/run  (via pipeline)
#
# VIVA POINTS:
#   Q: Why greedy and not another GA?
#   A: The placement is a maximum weighted coverage problem — NP-hard in
#      general, but greedy gives a (1-1/e) ≈ 63% approximation guarantee
#      and runs in O(n²) which is fast on a 15×15 = 225-node grid.
#   Q: Why the spread constraint?
#   A: Without it, all officers cluster in one industrial zone. The Manhattan
#      distance floor ensures geographic diversity — each officer covers a
#      distinct quadrant of the city.
# =============================================================================

from typing import List, Dict, Optional
from models.city_node import CityNode
from models.city_graph import CityGraph
from config import RISK_HIGH, RISK_MEDIUM, RISK_LOW


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RISK_WEIGHT = {
    RISK_HIGH:   3.0,
    RISK_MEDIUM: 1.5,
    RISK_LOW:    0.0,
}

# Minimum Manhattan-distance spread between any two officers.
# On a 15×15 grid, 3 means officers can't be placed in adjacent cells.
MIN_OFFICER_SPREAD = 3

# Number of officers to place (can be overridden at call site)
DEFAULT_NUM_OFFICERS = 5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def place_police_officers(
        graph: CityGraph,
        num_officers: int = DEFAULT_NUM_OFFICERS,
        logger=None,
        sim_step: int = 0,
) -> List[Dict]:
    """
    Compute optimal police officer positions based on current risk labels,
    population density, and accumulated emergency counts.

    Parameters
    ----------
    graph        : CityGraph — shared graph with up-to-date risk labels
    num_officers : int — how many officers to place (default 5)
    logger       : EventLogger | None — if provided, logs POLICE events
    sim_step     : int — simulation step number for log entries

    Returns
    -------
    List[Dict] with one entry per officer:
        {
          "officer_id"  : int,          # 0-indexed
          "row"         : int,
          "col"         : int,
          "node_id"     : str,          # "N_RR_CC" format for UI
          "risk"        : str,          # "High" | "Medium"
          "score"       : float,        # composite placement score
          "population"  : float,        # raw population density
        }

    Side Effects
    ------------
    - Clears node.police_here on ALL nodes first
    - Sets  node.police_here = True on selected nodes
    """
    # ── Step 0: clear previous placements ────────────────────────────────────
    for node in graph.all_nodes():
        node.police_here = False

    # ── Step 1: score every node ──────────────────────────────────────────────
    scored: List[tuple] = []   # (score, node)

    for node in graph.all_nodes():
        risk_label = getattr(node, "predicted_risk", None) or node.risk_level
        w = RISK_WEIGHT.get(risk_label, 0.0)

        if w == 0.0:
            continue   # skip Low-risk nodes entirely

        # Emergency spillover bonus: nodes that already received emergencies
        # are more likely to receive future ones
        emg = getattr(node, "total_emergencies", 0.0)
        emg_bonus = 1.0 + min(emg * 0.1, 1.0)   # capped at 2.0× bonus

        pop = node.population_density
        score = w * pop * emg_bonus

        scored.append((score, node))

    # Sort descending by score
    scored.sort(key=lambda t: t[0], reverse=True)

    if not scored:
        print("[C5-Police] Warning: no High/Medium risk nodes found — no officers placed.")
        return []

    # ── Step 2: greedy placement with spread constraint ────────────────────────
    placed: List[CityNode] = []

    for score, candidate in scored:
        if len(placed) >= num_officers:
            break

        # Check minimum distance from all already-placed officers
        too_close = False
        for existing in placed:
            dist = abs(candidate.row - existing.row) + abs(candidate.col - existing.col)
            if dist < MIN_OFFICER_SPREAD:
                too_close = True
                break

        if not too_close:
            placed.append(candidate)
            candidate.police_here = True

    # If spread constraint was too tight, fill remaining slots without it
    if len(placed) < num_officers:
        for score, candidate in scored:
            if len(placed) >= num_officers:
                break
            if not getattr(candidate, "police_here", False):
                placed.append(candidate)
                candidate.police_here = True

    # ── Step 3: build result list ─────────────────────────────────────────────
    result = []
    for i, node in enumerate(placed):
        risk_label = getattr(node, "predicted_risk", None) or node.risk_level
        node_id = f"N_{str(node.row).zfill(2)}_{str(node.col).zfill(2)}"

        # Re-compute score for output
        w = RISK_WEIGHT.get(risk_label, 0.0)
        emg = getattr(node, "total_emergencies", 0.0)
        emg_bonus = 1.0 + min(emg * 0.1, 1.0)
        score = w * node.population_density * emg_bonus

        entry = {
            "officer_id": i,
            "row":        node.row,
            "col":        node.col,
            "node_id":    node_id,
            "risk":       risk_label,
            "score":      round(score, 4),
            "population": round(node.population_density, 2),
        }
        result.append(entry)

    # ── Step 4: print summary ─────────────────────────────────────────────────
    print(f"\n[C5-Police] {len(result)} officers placed:")
    for r in result:
        print(f"  Officer {r['officer_id']}: ({r['row']},{r['col']}) "
              f"risk={r['risk']} score={r['score']:.2f} pop={r['population']:.1f}")

    # ── Step 5: log to EventLogger if provided ────────────────────────────────
    if logger:
        positions = [(r["row"], r["col"]) for r in result]
        logger.log(
            sim_step, "POLICE",
            f"{len(result)} officers deployed at {positions}."
        )

    return result


def get_police_positions(graph: CityGraph) -> List[Dict]:
    """
    Returns current police positions from node attributes.
    Used by server.py /api/graph to serialise state for the UI.
    """
    result = []
    i = 0
    for node in graph.all_nodes():
        if getattr(node, "police_here", False):
            risk_label = getattr(node, "predicted_risk", None) or node.risk_level
            result.append({
                "officer_id": i,
                "row":        node.row,
                "col":        node.col,
                "node_id":    f"N_{str(node.row).zfill(2)}_{str(node.col).zfill(2)}",
                "risk":       risk_label,
            })
            i += 1
    return result


def reposition_officers_after_drift(
        graph: CityGraph,
        num_officers: int = DEFAULT_NUM_OFFICERS,
        logger=None,
        sim_step: int = 0,
) -> List[Dict]:
    """
    Called by CrimeStatsManager after a re-train cycle so officers
    follow the updated risk landscape.

    This is a thin wrapper around place_police_officers() with a
    distinct name for clarity in the simulation log.
    """
    print(f"\n[C5-Police] Risk landscape updated — repositioning officers.")
    return place_police_officers(graph, num_officers, logger, sim_step)