from __future__ import annotations

from typing import Dict, List, Optional

from control.candidates import candidate_max_value
from control.constants import CONTROL_KEYS


def calculate_energy_wh(control: Dict[str, float], config: Dict[str, object]) -> float:
    return sum(calculate_device_energy_breakdown(control, config).values())


def calculate_resource_cost(control: Dict[str, float], config: Dict[str, object]) -> float:
    normalizer = float(config["nutrient_resource_normalizer_ml"])
    return float(control["nutrient_pump_ml"]) / normalizer if normalizer > 0.0 else 0.0


def calculate_change_penalty(
    control: Dict[str, float],
    previous_control: Optional[Dict[str, float]],
    config: Dict[str, object],
) -> float:
    if not previous_control:
        return 0.0
    diffs: List[float] = []
    for key in CONTROL_KEYS:
        denominator = candidate_max_value(key, config) or 1.0
        diffs.append(abs(float(control[key]) - float(previous_control.get(key, 0.0))) / denominator)
    return sum(diffs) / len(diffs) if diffs else 0.0


def calculate_over_control_penalty(
    current_inputs: Dict[str, float],
    control: Dict[str, float],
    config: Dict[str, object],
    ec_mode: str,
) -> float:
    penalty = 0.0
    thresholds = config.get("ec_nutrient_penalty_threshold_by_mode", {})
    if isinstance(thresholds, dict) and ec_mode in thresholds:
        ec_threshold = float(thresholds[ec_mode])
    else:
        ec_threshold = float(config["ec_nutrient_penalty_threshold"])
    if float(current_inputs["EC"]) >= ec_threshold and float(control["nutrient_pump_ml"]) > 0.0:
        penalty += float(control["nutrient_pump_ml"]) / float(config["nutrient_resource_normalizer_ml"])
    if float(current_inputs["T"]) <= float(config["low_temperature_cooler_penalty_threshold"]) and float(control["cooler_minutes"]) > 0.0:
        penalty += float(control["cooler_minutes"]) / max(candidate_max_value("cooler_minutes", config), 1.0)
    if float(current_inputs["DO"]) >= float(config["high_do_aeration_penalty_threshold"]) and float(control["aeration_minutes"]) > 0.0:
        penalty += float(control["aeration_minutes"]) / max(candidate_max_value("aeration_minutes", config), 1.0)
    if (
        float(current_inputs["FR"]) >= float(config["normal_fr_circulation_penalty_threshold"])
        and float(control["circulation_pump_minutes"]) >= float(config["excessive_circulation_minutes_threshold"])
    ):
        penalty += float(control["circulation_pump_minutes"]) / max(candidate_max_value("circulation_pump_minutes", config), 1.0)
    if float(control["led_minutes"]) > 0.0:
        penalty += float(control["led_minutes"]) / max(candidate_max_value("led_minutes", config), 1.0)
    return penalty


def calculate_weak_reduction_penalty(
    current_risk_score: float,
    predicted_risk_score: float,
    energy_wh: float,
    resource_cost_norm: float,
    config: Dict[str, object],
) -> float:
    risk_reduction = current_risk_score - predicted_risk_score
    threshold = float(config["risk_reduction_threshold"])
    if risk_reduction < threshold and (energy_wh > 0.0 or resource_cost_norm > 0.0):
        return (threshold - risk_reduction) + (energy_wh / float(config["energy_normalizer_wh"])) + resource_cost_norm
    return 0.0


def calculate_device_energy_breakdown(control: Dict[str, float], config: Dict[str, object]) -> Dict[str, float]:
    nutrient_flow = float(config["estimated_nutrient_pump_flow_ml_per_min"])
    nutrient_minutes = float(control["nutrient_pump_ml"]) / nutrient_flow if nutrient_flow > 0.0 else 0.0
    return {
        "circulation_pump": float(config["estimated_circulation_pump_watt"]) * float(control["circulation_pump_minutes"]) / 60.0,
        "nutrient_pump": float(config["estimated_nutrient_pump_watt"]) * nutrient_minutes / 60.0,
        "aeration_pump": float(config["estimated_aeration_pump_watt"]) * float(control["aeration_minutes"]) / 60.0,
        "peltier_cooler": float(config["estimated_peltier_cooler_watt"]) * float(control["cooler_minutes"]) / 60.0,
        "cooling_fan": float(config["estimated_fan_watt"]) * float(control["fan_minutes"]) / 60.0,
        "grow_led": float(config["estimated_led_watt"]) * float(control["led_minutes"]) / 60.0,
    }
