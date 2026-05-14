# Synthetic Dataset Generator
# classes used here :
#   Uses models.city_node.CityNode 

import random
import numpy as np
import pandas as pd
from typing import List, Dict

from models.city_node import CityNode
from config import (
    LOC_RESIDENTIAL, LOC_HOSPITAL, LOC_SCHOOL,
    LOC_INDUSTRIAL, LOC_POWER_PLANT, LOC_AMBULANCE,
)



LOCATION_TYPE_ENCODING = {
    LOC_RESIDENTIAL: 0,
    LOC_HOSPITAL:    1,
    LOC_SCHOOL:      2,
    LOC_INDUSTRIAL:  3,
    LOC_POWER_PLANT: 4,
    LOC_AMBULANCE:   5,
}

RISK_SCALE = ["Low", "Medium", "High"]

def _bump(risk: str, direction: int) -> str:
    """Move risk up (+1) or down (-1) one level, clamped to [Low, High]."""
    idx = RISK_SCALE.index(risk)
    return RISK_SCALE[max(0, min(2, idx + direction))]


def assign_ground_truth_risk(
        nodes_list: List[CityNode],
        X_normalized: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_profiles: Dict[int, str]
) -> List[dict]:
    """
    RULE (industrial proximity drives risk) — tightened thresholds:
      ind >= 0.75 AND pop >= 0.55              → HIGH
      ind >= 0.65 AND pop >= 0.35              → MEDIUM
      ind >= 0.50 OR  pop >= 0.80              → MEDIUM
      otherwise                                → LOW

    Target distribution: ~15-20% High, ~30-35% Medium, ~45-55% Low

    NOISE (10% random ±1 level shift — prevents overfit)
    """
    dataset      = []
    label_counts = {"Low": 0, "Medium": 0, "High": 0}

    for i, node in enumerate(nodes_list):
        pop      = float(X_normalized[i][0])
        ind      = float(X_normalized[i][1])
        cid      = int(cluster_labels[i])
        loc_type = node.location_type

        # Primary rule — stricter thresholds
        if ind >= 0.75 and pop >= 0.55:
            risk = "High"
        elif ind >= 0.65 and pop >= 0.35:
            risk = "Medium"
        elif ind >= 0.50 or pop >= 0.80:
            risk = "Medium"
        else:
            risk = "Low"

        # Secondary modifier: location type
        if loc_type in (LOC_INDUSTRIAL, LOC_POWER_PLANT):
            risk = _bump(risk, +1)
        elif loc_type == LOC_SCHOOL and ind >= 0.75:   # only bump schools very close to industry
            risk = _bump(risk, +1)
        elif loc_type in (LOC_HOSPITAL, LOC_AMBULANCE):
            risk = _bump(risk, -1)

        # 10% noise injection
        if random.random() < 0.10:
            risk = _bump(risk, random.choice([-1, 1]))

        label_counts[risk] += 1

        dataset.append({
            "population_density":    pop,
            "industrial_proximity":  ind,
            "location_type_encoded": LOCATION_TYPE_ENCODING.get(loc_type, 0),
            "cluster_id":            cid,
            "risk_label":            risk,
        })

    print(f"\n[C5-Dataset] Synthetic dataset generated ({len(dataset)} samples):")
    for label, count in label_counts.items():
        pct = 100 * count / len(dataset)
        print(f"  {label:6s}: {count:4d} ({pct:.1f}%)")

    return dataset



def build_training_dataframe(dataset: List[dict]) -> pd.DataFrame:
    """Converts dataset list to pandas DataFrame for sklearn."""
    return pd.DataFrame(dataset)