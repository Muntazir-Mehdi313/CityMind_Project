# =============================================================================
# algorithms/crime_ml.py — Challenge 5 Public Entry Point
# =============================================================================
# Thin wrapper that server.py and main.py call.
# Real implementation lives in challenge5/pipeline.py and its sub-modules.
#
# server.py calls:   run_crime_pipeline(graph)
# main.py calls:     run_crime_pipeline(graph)  (standalone, no placer)
#
# Returns: dict mapping CityNode -> risk_label string ("High"/"Medium"/"Low")
# =============================================================================

from models.city_graph import CityGraph
from models.city_node import CityNode
from challenge5.features import extract_and_normalize_features
from challenge5.clustering import run_kmeans_clustering, interpret_clusters, assign_cluster_ids_to_nodes
from challenge5.dataset import assign_ground_truth_risk, build_training_dataframe
from challenge5.classifier import CrimeRiskClassifier
from typing import Dict


def run_crime_pipeline(graph: CityGraph) -> Dict[CityNode, str]:
    """
    Public entry point for Challenge 5 — Crime Risk Prediction Pipeline.

    Called by:
        - main.py standalone (without AmbulancePlacer, before C3 runs)
        - server.py /api/challenge/5/run endpoint

    For the full integrated pipeline (with AmbulancePlacer re-trigger support),
    use challenge5/pipeline.py::run_crime_risk_pipeline() instead.

    Parameters
    ----------
    graph : CityGraph
        The shared city graph. Must already have:
          - location types set (Challenge 1 done)
          - population densities set (Challenge 1 done)
          - roads built (Challenge 2 done) — needed for industrial proximity Dijkstra

    Returns
    -------
    Dict[CityNode, str]
        Mapping of every node to its predicted risk label.
        Also writes node.risk_multiplier and node.predicted_risk directly onto
        each node in the shared graph.

    Side Effects
    ------------
    - node.risk_multiplier updated on all nodes (1.0, 1.2, or 1.5)
    - node.predicted_risk updated on all nodes ("Low", "Medium", "High")
    - node.cluster_id set on all nodes (int, K-Means cluster assignment)
    - graph fires "risk_updated" observer events via graph.set_risk() per node
    """
    print("\n" + "=" * 55)
    print("CHALLENGE 5: Crime Risk Prediction Pipeline (standalone)")
    print("=" * 55)

    # Phase A: Feature extraction + clustering
    print("\n[Phase A] Feature Extraction & K-Means Clustering")
    nodes_list, X_normalized, X_raw = extract_and_normalize_features(graph)
    best_k, cluster_labels, kmeans_model = run_kmeans_clustering(X_normalized)
    cluster_profiles = interpret_clusters(kmeans_model, best_k)
    assign_cluster_ids_to_nodes(nodes_list, cluster_labels)

    # Phase B: Synthetic dataset generation
    print("\n[Phase B] Synthetic Dataset Generation")
    dataset = assign_ground_truth_risk(
        nodes_list, X_normalized, cluster_labels, cluster_profiles
    )
    df = build_training_dataframe(dataset)

    # Phase C: Train classifier + apply to graph
    print("\n[Phase C] Decision Tree Training & Graph Update")
    classifier = CrimeRiskClassifier()
    classifier.train(df)

    predictions = classifier.predict_all_nodes(nodes_list, X_normalized, cluster_labels)

    # Apply to graph using graph.set_risk() so observer events fire
    classifier.apply_predictions_to_graph(predictions, graph)

    # Write predicted_risk label onto each node
    for node, label in predictions.items():
        node.predicted_risk = label

    print(f"\n[Challenge 5] Standalone pipeline complete.")
    print(f"  Nodes classified: {len(predictions)}")
    high   = sum(1 for v in predictions.values() if v == "High")
    medium = sum(1 for v in predictions.values() if v == "Medium")
    low    = sum(1 for v in predictions.values() if v == "Low")
    print(f"  High: {high} | Medium: {medium} | Low: {low}")
    print("=" * 55 + "\n")

    return predictions