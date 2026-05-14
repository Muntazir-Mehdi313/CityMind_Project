# challenge5/pipeline.py — Challenge 5 Entry Point (Phase 13)
# Updated: Police officer placement added after Phase C.

from models.city_graph import CityGraph
from challenge3.placer import AmbulancePlacer
from challenge5.features import extract_and_normalize_features
from challenge5.clustering import (
    run_kmeans_clustering,
    interpret_clusters,
    assign_cluster_ids_to_nodes,
)
from challenge5.dataset import assign_ground_truth_risk, build_training_dataframe
from challenge5.classifier import CrimeRiskClassifier
from challenge5.monitor import CrimeStatsManager
from challenge5.police import place_police_officers   # NEW


def run_crime_risk_pipeline(
        graph: CityGraph,
        ambulance_placer: AmbulancePlacer,
        logger=None,
        num_police_officers: int = 5,         # NEW param
) -> CrimeStatsManager:
    """
    Full Challenge 5 pipeline. Call ONCE before the simulation loop.

    Execution Order:
      Phase A: Feature extraction + K-Means clustering
      Phase B: Synthetic dataset generation
      Phase C: Decision Tree training + initial graph risk update
      Phase C2: Police officer placement (NEW)
      Phase D: CrimeStatsManager setup

    Returns:
        CrimeStatsManager ready for Phase D.
        Call .step() and .notify_emergency() from the simulation loop.
    """
    print("\n" + "=" * 55)
    print("CHALLENGE 5: Crime Risk Prediction Pipeline")
    print("=" * 55)

    # ── PHASE A: Feature Extraction + Clustering ──────────────────────────────
    print("\n[Phase A] Feature Extraction & K-Means Clustering")

    nodes_list, X_normalized, X_raw = extract_and_normalize_features(graph)
    best_k, cluster_labels, kmeans_model = run_kmeans_clustering(X_normalized)
    cluster_profiles = interpret_clusters(kmeans_model, best_k)
    assign_cluster_ids_to_nodes(nodes_list, cluster_labels)

    # ── PHASE B: Synthetic Dataset ────────────────────────────────────────────
    print("\n[Phase B] Synthetic Dataset Generation")

    dataset = assign_ground_truth_risk(
        nodes_list, X_normalized, cluster_labels, cluster_profiles
    )
    df = build_training_dataframe(dataset)

    # ── PHASE C: Decision Tree + Graph Update ─────────────────────────────────
    print("\n[Phase C] Decision Tree Training & Graph Update")

    classifier = CrimeRiskClassifier()
    classifier.train(df)

    predictions = classifier.predict_all_nodes(
        nodes_list, X_normalized, cluster_labels
    )
    # apply_predictions_to_graph calls graph.set_risk() per node,
    # which fires observer events → D* Lite replans automatically
    classifier.apply_predictions_to_graph(predictions, graph)

    # FIX 4: Notify placer after initial risk apply — may trigger GA re-run
    ambulance_placer.notify_risk_updated()

    # ── PHASE C2: Police Officer Placement (NEW) ──────────────────────────────
    print("\n[Phase C2] Police Officer Placement")

    police_positions = place_police_officers(
        graph,
        num_officers=num_police_officers,
        logger=logger,
        sim_step=0,
    )

    # ── PHASE D: Dynamic Monitor Setup ───────────────────────────────────────
    print("\n[Phase D] Initialising Dynamic Learning Monitor")

    monitor = CrimeStatsManager(
        graph, classifier, ambulance_placer,
        logger=logger,
        num_police_officers=num_police_officers,   # NEW: monitor repositions after drift
    )
    monitor.set_pipeline_data(nodes_list, X_normalized, cluster_labels)

    print(f"\n[Challenge 5] Pipeline complete.")
    print(f"  Officers deployed : {len(police_positions)}")
    print("=" * 55 + "\n")

    return monitor