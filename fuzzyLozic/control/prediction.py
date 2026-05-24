from __future__ import annotations

from typing import Dict

from control.utils import clamp


def predict_next_inputs(
    current_inputs: Dict[str, float],
    control: Dict[str, float],
    config: Dict[str, object],
) -> Dict[str, float]:
    predicted = {key: float(value) for key, value in current_inputs.items()}
    t_current = float(current_inputs["T"])
    do_current = float(current_inputs["DO"])
    fr_current = float(current_inputs["FR"])
    ec_current = float(current_inputs["EC"])

    cooler_active = 1.0 if float(control["cooler_minutes"]) > 0.0 else 0.0
    t_next = (
        t_current
        - float(config["heuristic_temp_drop_per_cooler_min"]) * float(control["cooler_minutes"])
        - float(config["heuristic_temp_drop_per_fan_min_when_cooling"]) * float(control["fan_minutes"]) * cooler_active
        - float(config["heuristic_temp_stabilize_per_circulation_min"]) * float(control["circulation_pump_minutes"])
    )
    t_next = clamp(t_next, *config["input_clamps"]["T"])

    do_next = (
        do_current
        + float(config["heuristic_do_gain_per_aeration_min"]) * float(control["aeration_minutes"])
        + float(config["heuristic_do_gain_per_circulation_min"]) * float(control["circulation_pump_minutes"])
        + float(config["heuristic_do_gain_per_temp_drop"]) * max(0.0, t_current - t_next)
    )
    fr_next = fr_current + float(config["heuristic_fr_gain_per_circulation_min"]) * float(control["circulation_pump_minutes"])
    ec_next = ec_current + float(config["heuristic_ec_gain_per_nutrient_ml"]) * float(control["nutrient_pump_ml"])

    predicted["T"] = t_next
    predicted["DO"] = clamp(do_next, *config["input_clamps"]["DO"])
    predicted["FR"] = clamp(fr_next, *config["input_clamps"]["FR"])
    predicted["EC"] = clamp(ec_next, *config["input_clamps"]["EC"])
    predicted["pH"] = clamp(float(current_inputs["pH"]), *config["input_clamps"]["pH"])
    predicted["Turbidity"] = clamp(float(current_inputs["Turbidity"]), *config["input_clamps"]["Turbidity"])
    return predicted
