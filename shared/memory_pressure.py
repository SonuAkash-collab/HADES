"""
Memory pressure monitor for HADES L1 cache budget scaling.
Implements OS-inspired memory pressure detection — dynamically
scales L1 token budgets based on host RAM utilization.
"""
import psutil

# Pressure tiers
PRESSURE_RELAXED  = 0.60  # < 60% RAM used — full budgets
PRESSURE_MODERATE = 0.75  # 60-75% — scale to 70%
PRESSURE_HIGH     = 0.85  # 75-85% — scale to 50%
PRESSURE_CRITICAL = 1.00  # > 85% — scale to 30% (emergency mode)

SCALE_FACTORS = {
    "relaxed":  1.0,
    "moderate": 0.7,
    "high":     0.5,
    "critical": 0.3,
}

def get_memory_pressure() -> tuple[str, float]:
    """
    Returns (tier_name, utilization_fraction).
    tier_name is one of: relaxed, moderate, high, critical
    """
    mem = psutil.virtual_memory()
    ram_percent = mem.percent / 100.0
    
    if ram_percent < PRESSURE_RELAXED:
        tier = "relaxed"
    elif ram_percent < PRESSURE_MODERATE:
        tier = "moderate"
    elif ram_percent < PRESSURE_HIGH:
        tier = "high"
    else:
        tier = "critical"
        
    return tier, ram_percent

def scale_budgets(base_budgets: dict[str, int]) -> dict[str, int]:
    """
    Scale L1 cache budgets based on current memory pressure.
    system budget is never scaled (pinned instructions).
    facts, history, tools, scratch budgets are scaled.
    Enforces a minimum floor of 20 tokens per scalable budget.
    Returns a new dict with scaled values.
    """
    tier, _ = get_memory_pressure()
    scale_factor = SCALE_FACTORS[tier]
    
    scaled_budgets = {}
    for key, val in base_budgets.items():
        if key == "system":
            scaled_budgets[key] = val
        else:
            scaled_val = int(val * scale_factor)
            scaled_budgets[key] = max(20, scaled_val)
            
    return scaled_budgets

def get_pressure_report() -> dict:
    """
    Returns a dict suitable for inclusion in telemetry:
    { tier, ram_percent, scale_factor, budgets_scaled: bool }
    """
    tier, ram_fraction = get_memory_pressure()
    return {
        "tier": tier,
        "ram_percent": round(ram_fraction * 100, 1),
        "scale_factor": SCALE_FACTORS[tier],
        "budgets_scaled": tier != "relaxed"
    }
