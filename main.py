# =============================================================================
# main.py — CityMind Master Orchestrator
# =============================================================================
# =============================================================================
# main.py — CityMind Master Orchestrator
# =============================================================================
# This is the MAIN ENTRY POINT for the full CityMind system.
#
# According to the project specification (Section 4):
#   "During evaluation, your system will be run through a simulation 
#    scenario that lasts 20 steps."
#
# This script calls simulation/loop.py::run_simulation() which:
#   1. Initializes all 5 challenges in sequence
#   2. Runs a 20-step simulation with random events (floods, emergencies)
#   3. Returns final state (graph, logger, ambulance_placer, crime_monitor)
#
# Usage:
#   python main.py          # Run with random seed
#   python main.py 42       # Run with seed=42 (reproducible)
# =============================================================================

import sys
from simulation.loop import run_simulation
from config import GRID_ROWS, GRID_COLS, SIM_STEPS


def main(seed: int = None):
    """
    Main orchestrator — runs the full CityMind system.
    
    Parameters
    ----------
    seed : int, optional
        Random seed for reproducible runs (useful for testing/viva).
        If None, each run is different.
    
    Returns
    -------
    (graph, logger, ambulance_placer, crime_monitor)
        Final state of all system components after 20-step simulation.
    """
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║     CityMind: Urban Intelligence System" + " " * 18 + "║")
    print("║     20-Step Simulation Engine" + " " * 28 + "║")
    print("╚" + "═" * 58 + "╝\n")
    
    # Run the full 20-step simulation (includes all challenge initialization)
    graph, logger, ambulance_placer, crime_monitor = run_simulation(
        grid_rows=GRID_ROWS,
        grid_cols=GRID_COLS,
        sim_steps=SIM_STEPS,
        seed=seed
    )
    
    # Print final summary
    print("\n" + "─" * 60)
    print("SIMULATION RESULTS")
    print("─" * 60)
    final_positions = [(n.row, n.col) for n in ambulance_placer.get_current_placement()]
    print(f"  Final ambulance positions: {final_positions}")
    print(f"  Final worst-case distance: {ambulance_placer.get_current_fitness():.4f}")
    print(f"  Total events logged:       {len(logger.entries)}")
    print(f"  Graph risk version:        {graph.version}")
    from challenge5.police import get_police_positions
    police = get_police_positions(graph)
    print(f"  Police officers deployed:  {len(police)}")
    for p in police:
        print(f"    Officer {p['officer_id']}: ({p['row']},{p['col']}) risk={p['risk']}")
    print("─" * 60 + "\n")
    
    return graph, logger, ambulance_placer, crime_monitor


if __name__ == "__main__":
    # Optional: pass seed for reproducible testing
    # e.g., python main.py 42  →  uses seed=42
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
    graph, logger, ambulance_placer, crime_monitor = main(seed=seed)
    
    print("✔ Simulation complete. To view results in the UI dashboard:")
    print("  1. Run: python server.py")
    print("  2. Open: http://localhost:5000/dashboard.html\n")