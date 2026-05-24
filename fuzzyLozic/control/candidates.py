from __future__ import annotations

import math
from itertools import product
from typing import Dict, List

from control.constants import CONTROL_KEYS


def generate_candidates(config: Dict[str, object]) -> List[Dict[str, float]]:
    candidates: List[Dict[str, float]] = []
    limit = config.get("candidate_limit")
    candidate_limit = int(limit) if isinstance(limit, int) and limit > 0 else None
    for values in product(
        config["circulation_pump_minutes_candidates"],
        config["nutrient_pump_ml_candidates"],
        config["aeration_minutes_candidates"],
        config["cooler_minutes_candidates"],
        config["fan_minutes_candidates"],
        config["led_minutes_candidates"],
    ):
        control = dict(zip(CONTROL_KEYS, [float(value) for value in values]))
        if not is_valid_candidate(control, config):
            continue
        candidates.append(control)
        if candidate_limit is not None and len(candidates) >= candidate_limit:
            break
    return candidates


def is_valid_candidate(control: Dict[str, float], config: Dict[str, object]) -> bool:
    if float(control["cooler_minutes"]) > 0.0 and float(control["fan_minutes"]) < float(control["cooler_minutes"]):
        return False
    for key, options_key in control_option_key_map().items():
        if float(control[key]) not in {float(value) for value in config[options_key]}:
            return False
    return True


def control_option_key_map() -> Dict[str, str]:
    return {
        "circulation_pump_minutes": "circulation_pump_minutes_candidates",
        "nutrient_pump_ml": "nutrient_pump_ml_candidates",
        "aeration_minutes": "aeration_minutes_candidates",
        "cooler_minutes": "cooler_minutes_candidates",
        "fan_minutes": "fan_minutes_candidates",
        "led_minutes": "led_minutes_candidates",
    }


def candidate_max_value(control_key: str, config: Dict[str, object]) -> float:
    return max(float(value) for value in config[control_option_key_map()[control_key]])


def raw_candidate_count(config: Dict[str, object]) -> int:
    return math.prod(len(config[key]) for key in control_option_key_map().values())
