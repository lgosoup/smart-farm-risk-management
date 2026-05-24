from __future__ import annotations

from typing import Dict

from control.scoring import calculate_device_energy_breakdown


def build_device_commands(control: Dict[str, float], config: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    energy = calculate_device_energy_breakdown(control, config)
    return {
        "circulation_pump": {
            "enabled": float(control["circulation_pump_minutes"]) > 0.0,
            "duration_sec": int(round(float(control["circulation_pump_minutes"]) * 60.0)),
            "estimated_wh": round(energy["circulation_pump"], 4),
            "purpose": "\uc21c\ud658/\uc720\ub7c9 \uc548\uc815\ud654 \ubc0f DO \ubcf4\uc870",
        },
        "nutrient_pump": {
            "enabled": float(control["nutrient_pump_ml"]) > 0.0,
            "target_ml": round(float(control["nutrient_pump_ml"]), 4),
            "estimated_wh": round(energy["nutrient_pump"], 4),
            "purpose": "EC \ubcf4\uc815",
        },
        "aeration_pump": {
            "enabled": float(control["aeration_minutes"]) > 0.0,
            "duration_sec": int(round(float(control["aeration_minutes"]) * 60.0)),
            "estimated_wh": round(energy["aeration_pump"], 4),
            "purpose": "DO \uc99d\uac00",
        },
        "peltier_cooler": {
            "enabled": float(control["cooler_minutes"]) > 0.0,
            "duration_sec": int(round(float(control["cooler_minutes"]) * 60.0)),
            "estimated_wh": round(energy["peltier_cooler"], 4),
            "purpose": "\uc591\uc561 \uc218\uc628 \ud558\uac15",
        },
        "cooling_fan": {
            "enabled": float(control["fan_minutes"]) > 0.0,
            "duration_sec": int(round(float(control["fan_minutes"]) * 60.0)),
            "estimated_wh": round(energy["cooling_fan"], 4),
            "purpose": "\ud3a0\ud2f0\uc5b4 \ubc29\uc5f4/\ud658\uae30",
        },
        "grow_led": {
            "enabled": float(control["led_minutes"]) > 0.0,
            "duration_sec": int(round(float(control["led_minutes"]) * 60.0)),
            "estimated_wh": round(energy["grow_led"], 4),
            "purpose": "\uad11\ub7c9/\uad11\uc8fc\uae30 \uc81c\uc5b4",
        },
    }
