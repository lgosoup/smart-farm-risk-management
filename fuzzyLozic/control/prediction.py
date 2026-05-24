from __future__ import annotations

from typing import Dict

from control.utils import clamp


def predict_next_inputs(
    current_inputs: Dict[str, float],
    control: Dict[str, float],
    config: Dict[str, object],
) -> Dict[str, float]:
    predicted = {key: float(value) for key, value in current_inputs.items()}
    wt_current = float(current_inputs.get("water_temperature", 20.0))
    fr_current = float(current_inputs.get("flow_ratio", 1.0))
    ec_current = float(current_inputs.get("EC", 0.0))

    cooler_active = 1.0 if float(control["cooler_minutes"]) > 0.0 else 0.0
    wt_next = (
        wt_current
        - float(config["heuristic_temp_drop_per_cooler_min"]) * float(control["cooler_minutes"])
        - float(config["heuristic_temp_drop_per_fan_min_when_cooling"]) * float(control["fan_minutes"]) * cooler_active
        - float(config["heuristic_temp_stabilize_per_circulation_min"]) * float(control["circulation_pump_minutes"])
    )
    wt_next = clamp(wt_next, *config["input_clamps"]["water_temperature"])

    fr_next = fr_current + float(config["heuristic_fr_gain_per_circulation_min"]) * float(control["circulation_pump_minutes"])
    ec_next = ec_current + float(config["heuristic_ec_gain_per_nutrient_ml"]) * float(control["nutrient_pump_ml"])

    predicted["water_temperature"] = wt_next
    predicted["flow_ratio"] = clamp(fr_next, *config["input_clamps"]["flow_ratio"])
    predicted["EC"] = clamp(ec_next, *config["input_clamps"]["EC"])
    predicted["pH"] = clamp(float(current_inputs.get("pH", 7.0)), *config["input_clamps"]["pH"])
    predicted["turbidity"] = clamp(float(current_inputs.get("turbidity", 0.0)), *config["input_clamps"]["turbidity"])
    # air_temperature, humidity는 제어 영향 없음 — 현재값 유지
    if "air_temperature" in current_inputs:
        predicted["air_temperature"] = clamp(float(current_inputs["air_temperature"]), *config["input_clamps"]["air_temperature"])
    if "humidity" in current_inputs:
        predicted["humidity"] = clamp(float(current_inputs["humidity"]), *config["input_clamps"]["humidity"])
    return predicted
