# challenge5/learning_loop.py — Dynamic Learning Loop Entry Helpers (Phase D)

# The simulation loop calls:
#   notify_emergency(monitor, node)  — when a civilian is reached
#   advance_step(monitor)            — at the end of every simulation step

from challenge5.monitor import CrimeStatsManager
from models.city_node import CityNode


def notify_emergency(monitor: CrimeStatsManager, node: CityNode) -> None:
    """
    Call whenever the D* Lite router reaches a civilian node.
    Increments emergency counter + 0.3 spillover to road neighbors.
    """
    monitor.notify_emergency(node)


def advance_step(monitor: CrimeStatsManager) -> None:
    """
    Call at the END of every simulation step.
    Every DRIFT_WINDOW steps, checks for concept drift and re-trains if needed.
    """
    monitor.step()


def get_risk_summary(monitor: CrimeStatsManager) -> dict:
    """
    Returns current risk distribution across all nodes.
    Used by event logger and UI stats panel.
    """
    counts = {"High": 0, "Medium": 0, "Low": 0}
    for node in monitor.graph.all_nodes():
        label = getattr(node, "predicted_risk", "Low")
        if label in counts:
            counts[label] += 1
    counts["total"] = sum(counts.values())
    counts["drift_step"] = monitor.simulation_step
    return counts


def get_emergency_hotspots(monitor: CrimeStatsManager, top_n: int = 5) -> list:
    """
    Returns top_n nodes with most accumulated emergencies.
    Useful for UI heatmap overlay.
    """
    nodes_with_emg = [
        (node, getattr(node, "total_emergencies", 0.0))
        for node in monitor.graph.all_nodes()
        if getattr(node, "total_emergencies", 0.0) > 0
    ]
    nodes_with_emg.sort(key=lambda x: x[1], reverse=True)
    return nodes_with_emg[:top_n]