#!/usr/bin/env python3
# =============================================================================
# run_hybrid.py — Launch CityMind with Pure-Pygame UI
# =============================================================================
# Usage: python run_hybrid.py
#
# All five challenges wired to a single-window pure-pygame dashboard.
# NOTE: 'n' variable collision in challenge 4/5 fixed (was shadowing param).
# =============================================================================

import sys
import threading
import random

from models.city_graph import CityGraph
from config import GRID_ROWS, GRID_COLS, SIM_STEPS

from algorithms.layout_csp     import run_layout
from algorithms.road_network    import run_road_network
from challenge3.placer          import AmbulancePlacer
from algorithms.dstar_lite      import run_dstar_lite
from challenge5.pipeline        import run_crime_risk_pipeline

from ui.hybrid_dashboard import HybridDashboard


class CityMindApp:
    """
    Main application: owns the shared CityGraph, wires all challenge
    runners to the dashboard, and starts the event loop.
    """

    def __init__(self):
        self.graph   = None
        self.placer  = None
        self.monitor = None
        self.dash    = None

    # ── Bootstrap ─────────────────────────────────────────────────────────────

    def initialize(self):
        print("[INIT] CityMind starting…")
        self.graph = CityGraph(GRID_ROWS, GRID_COLS)
        self.dash  = HybridDashboard(self.graph)

        self.dash.log("CityMind ready")
        self.dash.log(f"Grid: {GRID_ROWS}×{GRID_COLS}  |  Sim steps: {SIM_STEPS}")
        self.dash.log("")
        self.dash.log("Run challenges in order: C1 → C2 → C3 → C4 / C5")

        self._wire_buttons()

    def _wire_buttons(self):
        for i in range(1, 6):
            self.dash.challenge_buttons[i].config(command=self._make_handler(i))

    def _make_handler(self, n):
        def handler():
            self.run_challenge(n)
        return handler

    # ── Challenge runners ─────────────────────────────────────────────────────

    def run_challenge(self, n):
        self.dash.log(f"▶ Running Challenge {n}…")
        try:
            if n == 1:
                self._c1()
            elif n == 2:
                self._c2()
            elif n == 3:
                self._c3()
            elif n == 4:
                self._c4()
            elif n == 5:
                self._c5()
        except Exception as exc:
            self.dash.log(f"✗ C{n} error: {exc}")
            import traceback
            traceback.print_exc()

    def _c1(self):
        success = run_layout(self.graph)
        self.dash.log(f"✓ C1: Layout {'complete' if success else 'fallback mode'}")
        self.dash.set_graph(self.graph)

    def _c2(self):
        result    = run_road_network(self.graph)
        total     = result["total_cost"]
        redundant = result["redundancy_ok"]
        mst_cnt   = result["mst_edges"]
        extra_cnt = result["extra_edges"]
        road_cnt  = mst_cnt + extra_cnt

        self.dash.log(f"✓ C2: {mst_cnt} MST + {extra_cnt} extra edges  cost={total:.1f}")
        self.dash.log(f"   Redundancy: {'✓' if redundant else '✗'}")
        self.dash.stat_labels["roads"].config(text=str(road_cnt))
        self.dash.overlays["mst"] = True
        self.dash.overlay_vars["mst"].set(True)

    def _c3(self):
        if not self.placer:
            self.placer = AmbulancePlacer(self.graph)
        placement, fitness = self.placer.initial_placement()
        self.dash.log(f"✓ C3: GA complete  fitness={fitness:.2f}")
        self.dash.stat_labels["ambulances"].config(text="3/3")
        for idx, node in enumerate(placement):
            self.dash.log(f"   AMB-{idx + 1}: ({node.row},{node.col})")

    def _c4(self):
        if not self.placer or not self.placer.get_current_placement():
            self.dash.log("✗ C4: Run C3 first to place ambulances")
            return

        start = self.placer.get_current_placement()[0]

        # FIX: use 'nd' to avoid shadowing outer variable 'n'
        targets = [nd for nd in self.graph.all_nodes()
                   if nd.location_type == "Residential" and nd.is_accessible]
        if not targets:
            self.dash.log("✗ C4: No accessible residential targets")
            return

        target  = random.choice(targets)
        planner = run_dstar_lite(self.graph, start, target, subscribe=False)
        path    = planner.get_path()

        cost = (sum(path[i].effective_cost_to(path[i + 1])
                    for i in range(len(path) - 1))
                if len(path) > 1 else 0)

        self.dash.log(f"✓ C4: Path found  {len(path) - 1} hops  cost={cost:.1f}")

    def _c5(self):
        if not self.placer:
            self.placer = AmbulancePlacer(self.graph)

        self.monitor = run_crime_risk_pipeline(self.graph, self.placer)

        # FIX: use 'nd' to avoid shadowing outer variable 'n'
        high = sum(1 for nd in self.graph.all_nodes()
                   if getattr(nd, "predicted_risk", "Low") == "High")
        med  = sum(1 for nd in self.graph.all_nodes()
                   if getattr(nd, "predicted_risk", "Low") == "Medium")

        self.dash.log("✓ C5: ML pipeline complete")
        self.dash.log(f"   Risk → {high} High, {med} Medium")
        self.dash.stat_labels["risk_high"].config(text=str(high))
        self.dash.overlays["heatmap"] = True
        self.dash.overlay_vars["heatmap"].set(True)

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self):
        self.initialize()
        self.dash.run()   # blocks until window is closed


def main():
    app = CityMindApp()
    app.run()


if __name__ == "__main__":
    main()