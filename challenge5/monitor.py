# challenge5/monitor.py — Dynamic Learning Loop / CrimeStatsManager (Phase 12)
# Updated: police officer repositioning added after every retrain cycle.

import pandas as pd
from typing import List

from models.city_node import CityNode
from models.city_graph import CityGraph
from challenge5.classifier import CrimeRiskClassifier
from challenge3.placer import AmbulancePlacer
from challenge5.dataset import LOCATION_TYPE_ENCODING, RISK_SCALE
from challenge5.police import reposition_officers_after_drift   # NEW
from simulation.logger import EventLogger


class CrimeStatsManager:

    DRIFT_WINDOW                    = 5
    LOW_RISK_EMERGENCY_THRESHOLD    = 3
    MEDIUM_RISK_EMERGENCY_THRESHOLD = 6

    def __init__(
            self,
            graph: CityGraph,
            classifier: CrimeRiskClassifier,
            ambulance_placer: AmbulancePlacer,
            logger=None,
            num_police_officers: int = 5,    # NEW
    ):
        self.graph               = graph
        self.classifier          = classifier
        self.ambulance_placer    = ambulance_placer
        self.logger              = logger
        self.num_police_officers = num_police_officers   # NEW
        self.simulation_step     = 0

        self.nodes_list    = None
        self.X_normalized  = None
        self.cluster_labels = None

        # Initialise total_emergencies on all nodes as a dynamic attribute
        for node in graph.all_nodes():
            if not hasattr(node, "total_emergencies"):
                node.total_emergencies = 0.0

    def set_pipeline_data(self, nodes_list, X_normalized, cluster_labels):
        """Called once after Phase A-C to store data for re-training."""
        self.nodes_list     = nodes_list
        self.X_normalized   = X_normalized
        self.cluster_labels = cluster_labels
        # O(1) lookup instead of O(n) list.index()
        self._node_index = {node: i for i, node in enumerate(nodes_list)}

    # -------------------------------------------------------------------------
    # Public simulation hooks
    # -------------------------------------------------------------------------

    def notify_emergency(self, node: CityNode) -> None:
        """
        Called by Challenge 4 whenever the medical team dispatches
        to a civilian location.
        """
        if not hasattr(node, "total_emergencies"):
            node.total_emergencies = 0.0
        node.total_emergencies += 1.0

        for neighbor in node.get_neighbors():
            if not hasattr(neighbor, "total_emergencies"):
                neighbor.total_emergencies = 0.0
            neighbor.total_emergencies += 0.3

    def step(self) -> None:
        """Called by the main simulation loop every step."""
        self.simulation_step += 1
        if self.simulation_step % self.DRIFT_WINDOW == 0:
            self._check_for_drift()

    # -------------------------------------------------------------------------
    # Drift detection
    # -------------------------------------------------------------------------

    def _check_for_drift(self) -> None:
        drifted = []

        for node in self.graph.all_nodes():
            pred   = getattr(node, "predicted_risk", "Low")
            actual = getattr(node, "total_emergencies", 0.0)

            is_drifted = (
                (pred == "Low"    and actual >= self.LOW_RISK_EMERGENCY_THRESHOLD) or
                (pred == "Medium" and actual >= self.MEDIUM_RISK_EMERGENCY_THRESHOLD)
            )

            if is_drifted:
                drifted.append(node)
                print(f"  [DRIFT] Node ({node.row},{node.col}): "
                      f"predicted '{pred}' but {actual:.1f} emergencies observed.")

        if drifted:
            print(f"\n[CrimeStatsManager] Step {self.simulation_step}: "
                  f"{len(drifted)} drifted nodes → re-training.")
            if self.logger:
                self.logger.log(
                    self.simulation_step, "DRIFT",
                    f"{len(drifted)} nodes drifted from predicted risk. Re-training triggered."
                )
            self._retrain_with_corrections(drifted)

    # -------------------------------------------------------------------------
    # Re-training
    # -------------------------------------------------------------------------

    def _retrain_with_corrections(self, drifted_nodes: List[CityNode]) -> None:
        if self.nodes_list is None:
            raise RuntimeError(
                "set_pipeline_data() must be called before drift can be handled."
            )

        corrections = []
        for node in drifted_nodes:
            emergencies = getattr(node, "total_emergencies", 0.0)
            corrected_label = (
                "High" if emergencies >= self.MEDIUM_RISK_EMERGENCY_THRESHOLD
                else "Medium"
            )

            node_idx = self._node_index[node]
            corrections.append({
                "population_density":    float(self.X_normalized[node_idx][0]),
                "industrial_proximity":  float(self.X_normalized[node_idx][1]),
                "location_type_encoded": float(
                    LOCATION_TYPE_ENCODING.get(node.location_type, 0)
                ),
                "cluster_id":            float(self.cluster_labels[node_idx]),
                "risk_label":            corrected_label,
            })

        correction_df = pd.DataFrame(corrections)
        augmented_df  = pd.concat(
            [self.classifier.training_df] + [correction_df] * 3,
            ignore_index=True
        )

        print(f"  Training data: {len(self.classifier.training_df)} → "
              f"{len(augmented_df)} rows")

        self.classifier.train(augmented_df)

        predictions = self.classifier.predict_all_nodes(
            self.nodes_list, self.X_normalized, self.cluster_labels
        )
        self.classifier.apply_predictions_to_graph(predictions, self.graph)

        for node, label in predictions.items():
            node.predicted_risk = label

        # Notify ambulance placer — may trigger GA re-run
        self.ambulance_placer.notify_risk_updated()

        # ── NEW: Reposition police officers after risk landscape changes ──────
        reposition_officers_after_drift(
            self.graph,
            num_officers=self.num_police_officers,
            logger=self.logger,
            sim_step=self.simulation_step,
        )

        if self.logger:
            self.logger.log(
                self.simulation_step, "RETRAIN",
                f"Decision Tree retrained v{self.classifier.version}. "
                f"{len(drifted_nodes)} correction(s) ×3 weight. "
                f"Officers repositioned."
            )

        print(f"  [Step {self.simulation_step}] Re-train + officer reposition complete.")
