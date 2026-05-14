# challenge5/clustering.py — K-Means Clustering (Phase 9)
# classes used here :
#   Uses models.city_node.CityNode
#   No CityGraph API calls here — only numpy arrays and sklearn.
#   CityNode has no cluster_id. We add it dynamically at runtime via
#   assign_cluster_ids_to_nodes(). Python allows dynamic attribute setting.

import numpy as np
from typing import Tuple, Dict, List
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from models.city_node import CityNode

# OPTIMAL K SELECTION

def run_kmeans_clustering(
        X_normalized: np.ndarray,
        k_range: Tuple[int, int] = (2, 6)
) -> Tuple[int, np.ndarray, KMeans]:
    """
    Runs K-Means for each k in [k_range[0], k_range[1]) and selects the
    best k using the Silhouette Score (higher = better cluster separation).

    Args:
        X_normalized : numpy array (n_nodes, 2), values in [0, 1]
        k_range      : (min_k, max_k_exclusive), default (2,6) → k in {2,3,4,5}

    Returns:
        best_k         : optimal number of clusters
        cluster_labels : numpy array (n_nodes,) — cluster ID per node
        best_model     : fitted KMeans object
    """
    best_k     = k_range[0]
    best_score = -1.0
    best_model = None

    print("\n[C5-Clustering] Searching for optimal k (Silhouette Score):")

    for k in range(k_range[0], k_range[1]):
        model  = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = model.fit_predict(X_normalized)

        if len(set(labels)) < 2:
            print(f"  k={k}: SKIPPED — degenerate clustering")
            continue

        score = silhouette_score(X_normalized, labels)
        marker = "  ← best so far" if score > best_score else ""
        print(f"  k={k}: Silhouette Score = {score:.4f}{marker}")

        if score > best_score:
            best_score = score
            best_k     = k
            best_model = model

    if best_model is None:
        print("  [WARNING] All k values degenerate — falling back to k=2.")
        best_model = KMeans(n_clusters=2, random_state=42, n_init=10)
        best_model.fit(X_normalized)
        best_k = 2

    cluster_labels = best_model.labels_
    sizes = ", ".join(
        f"Cluster {i}: {(cluster_labels == i).sum()}" for i in range(best_k)
    )
    print(f"\n  Best k = {best_k}  (Silhouette = {best_score:.4f})")
    print(f"  Cluster sizes: {sizes}")

    return best_k, cluster_labels, best_model

# CLUSTER INTERPRETATION

def interpret_clusters(best_model: KMeans, best_k: int) -> Dict[int, str]:
    """
    Assigns a human-readable socio-economic profile to each cluster based
    on centroid position in the normalized 2D feature space.

    Quadrant mapping:
      ind >= 0.5 AND pop >= 0.5  → "Urban Industrial"   (HIGH crime risk)
      ind >= 0.5 AND pop <  0.5  → "Industrial Fringe"  (MEDIUM crime risk)
      ind <  0.5 AND pop >= 0.5  → "Residential Core"   (LOW crime risk)
      ind <  0.5 AND pop <  0.5  → "Rural Peripheral"   (LOW crime risk)

    Returns:
        cluster_profiles : dict mapping cluster_id (int) → profile name (str)
    """
    centroids        = best_model.cluster_centers_
    cluster_profiles: Dict[int, str] = {}

    print("\n[C5-Clustering] Cluster Profiles:")
    print(f"  {'Cluster':>9}  {'pop_norm':>10}  {'ind_norm':>10}  Profile")
    print(f"  {'-'*9}  {'-'*10}  {'-'*10}  {'-'*22}")

    for cid in range(best_k):
        pop_norm = float(centroids[cid][0])
        ind_norm = float(centroids[cid][1])

        if ind_norm >= 0.5:
            profile = "Urban Industrial" if pop_norm >= 0.5 else "Industrial Fringe"
        else:
            profile = "Residential Core" if pop_norm >= 0.5 else "Rural Peripheral"

        cluster_profiles[cid] = profile
        print(f"  Cluster {cid:>2}:  {pop_norm:>10.4f}  {ind_norm:>10.4f}  {profile}")

    duplicates = [p for p, c in Counter(cluster_profiles.values()).items() if c > 1]
    if duplicates:
        print(f"\n  [NOTE] Duplicate profile(s): {duplicates} — numerically distinct.")

    return cluster_profiles

# WRITE CLUSTER IDs TO NODES

def assign_cluster_ids_to_nodes(
        nodes_list: List[CityNode],
        cluster_labels: np.ndarray
) -> None:
    """
    Dynamically writes cluster_id onto each CityNode as a Python attribute.
    Partner's CityNode has no cluster_id field by default, so we set it here.
    Python allows adding attributes at runtime — this is intentional.

    Args:
        nodes_list     : ordered list of CityNode objects
        cluster_labels : numpy array of cluster IDs (same order as nodes_list)
    """
    for i, node in enumerate(nodes_list):
        node.cluster_id = int(cluster_labels[i])

    print(f"\n[C5-Clustering] cluster_id written to {len(nodes_list)} nodes.")