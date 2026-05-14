# =============================================================================
# config.py — Global Constants for CityMind
# =============================================================================
# Change GRID_ROWS / GRID_COLS here and the entire system adapts.
# All modules import from this file — never hardcode grid size elsewhere.

GRID_ROWS = 15
GRID_COLS = 15

# ---------------------------------------------------------------------------
# Location Types
# ---------------------------------------------------------------------------
# These string constants are the only legal values for CityNode.location_type.
# Using constants (not raw strings) means a typo like "Hosptial" is caught
# immediately as a NameError instead of silently breaking a constraint check.

LOC_EMPTY        = "Empty"
LOC_RESIDENTIAL  = "Residential"
LOC_HOSPITAL     = "Hospital"
LOC_SCHOOL       = "School"
LOC_INDUSTRIAL   = "Industrial"
LOC_POWER_PLANT  = "Power Plant"
LOC_AMBULANCE    = "Ambulance Depot"

ALL_LOCATION_TYPES = [
    LOC_EMPTY,
    LOC_RESIDENTIAL,
    LOC_HOSPITAL,
    LOC_SCHOOL,
    LOC_INDUSTRIAL,
    LOC_POWER_PLANT,
    LOC_AMBULANCE,
]

# ---------------------------------------------------------------------------
# Risk Levels (Challenge 5 output)
# ---------------------------------------------------------------------------
RISK_LOW    = "Low"
RISK_MEDIUM = "Medium"
RISK_HIGH   = "High"

# Edge weight multipliers applied at traversal time (Challenge 4 / 3 router)
RISK_MULTIPLIER = {
    RISK_LOW:    1.0,
    RISK_MEDIUM: 1.2,
    RISK_HIGH:   1.5,
}

# ---------------------------------------------------------------------------
# Edge Weights (Challenge 2)
# ---------------------------------------------------------------------------
WEIGHT_STANDARD    = 1.0   # Default road cost
WEIGHT_RESIDENTIAL = 0.8   # Roads through residential zones
WEIGHT_BLOCKED     = None  # Sentinel — blocked roads are removed from dict,
                           # but this constant is used in comments / assertions

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
SIM_STEPS = 20   # Total steps in the 20-step evaluation loop (Section 4)

# ---------------------------------------------------------------------------
# UI colours — kept here so the dashboard and the graph both agree
# ---------------------------------------------------------------------------
COLOR_NODE = {
    LOC_EMPTY:       "#E8E8E8",
    LOC_RESIDENTIAL: "#A8D5A2",
    LOC_HOSPITAL:    "#FF6B6B",
    LOC_SCHOOL:      "#FFD93D",
    LOC_INDUSTRIAL:  "#6C6C6C",
    LOC_POWER_PLANT: "#4ECDC4",
    LOC_AMBULANCE:   "#FF8C00",
}

COLOR_RISK = {
    RISK_LOW:    "#FFFFFF",
    RISK_MEDIUM: "#FFB347",
    RISK_HIGH:   "#FF4444",
}
