# challenge5/classifier.py — Decision Tree Classifier (Phase 11)
# other classes used here :
#   Uses models.city_node.CityNode  &
#   models.city_graph.CityGraph


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict

from models.city_node import CityNode
from models.city_graph import CityGraph
from challenge5.dataset import LOCATION_TYPE_ENCODING


class CrimeRiskClassifier:
    """
    Decision Tree classifier for crime risk prediction.

    Trains on synthetic data, predicts for all nodes, and applies risk
    multipliers to the shared city graph via graph.set_risk() so that
    the observer pattern fires automatically and D* Lite replans.

    Why Decision Tree?
    - Fully explainable: every prediction traces to an IF-THEN rule.
    - Can print the full tree with export_text() for the viva.
    - Fast training on ~400-node synthetic datasets.
    """

    FEATURES = [
        "population_density",
        "industrial_proximity",
        "location_type_encoded",
        "cluster_id",
    ]
    TARGET = "risk_label"

    # Hyperparameters
    # class_weight=None (uniform) — using "balanced" inflated High/Medium
    # predictions because the synthetic data was already skewed toward those
    # classes, causing the tree to over-predict red nodes on the map.
    HYPERPARAMS = dict(
        max_depth=5,
        min_samples_split=4,
        criterion="gini",
        class_weight=None,
        random_state=42,
    )

    def __init__(self):
        self.model: DecisionTreeClassifier = None
        self.label_encoder  = LabelEncoder()
        self.training_df: pd.DataFrame = None
        self.version: int = 0

    # Training

    def train(self, df: pd.DataFrame) -> None:
        """
        Trains the Decision Tree on the provided DataFrame.
        Prints accuracy, classification report, and tree rules.
        """
        self.training_df = df.copy()

        X     = df[self.FEATURES].values
        y_raw = df[self.TARGET].values
        y     = self.label_encoder.fit_transform(y_raw)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = DecisionTreeClassifier(**self.HYPERPARAMS)
        self.model.fit(X_train, y_train)

        y_pred   = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n[C5-Classifier] Decision Tree trained (v{self.version + 1}):")
        print(f"  Training samples : {len(X_train)} | Test samples: {len(X_test)}")
        print(f"  Test Accuracy    : {accuracy:.4f}")
        print(classification_report(
            y_test, y_pred, target_names=self.label_encoder.classes_
        ))

        rules = export_text(self.model, feature_names=self.FEATURES)
        print("[Decision Tree Rules (first 2000 chars)]:")
        print(rules[:2000])

        self.version += 1

    # Prediction

    def predict_all_nodes(
            self,
            nodes_list: List[CityNode],
            X_normalized: np.ndarray,
            cluster_labels: np.ndarray
    ) -> Dict[CityNode, str]:
        """
        Runs the trained classifier on all city nodes.
        Returns dict: node → risk_label ("High" / "Medium" / "Low")
        """
        if self.model is None:
            raise RuntimeError("Call train() before predict_all_nodes().")

        predictions = {}
        for i, node in enumerate(nodes_list):
            features = [
                float(X_normalized[i][0]),
                float(X_normalized[i][1]),
                float(LOCATION_TYPE_ENCODING.get(node.location_type, 0)),
                float(getattr(node, "cluster_id", 0)),   # dynamic attr
            ]
            encoded = self.model.predict([features])[0]
            label   = self.label_encoder.inverse_transform([encoded])[0]
            predictions[node] = label

        return predictions

    # Apply to graph

    def apply_predictions_to_graph(
            self,
            predictions: Dict[CityNode, str],
            graph: CityGraph
    ) -> None:
        """
        Writes risk levels to the shared graph using graph.set_risk().

        Using graph.set_risk() is critical — it fires the "risk_updated"
        observer event so D* Lite replans automatically. Do NOT write
        node.risk_multiplier directly.

        Also writes predicted_risk as a dynamic attribute on each node
        (partner's CityNode has no predicted_risk field by default).
        """
        changed = 0
        for node, label in predictions.items():
            old_risk = node.risk_level   # partner's field: risk_level (str)
            if old_risk != label:
                changed += 1

            # This fires the observer → D* Lite replans automatically
            graph.set_risk(node, label)

            node.predicted_risk = label

        print(f"[C5-Classifier] Risk applied: {changed} nodes changed.")