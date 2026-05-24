from __future__ import annotations

from pathlib import Path
from typing import Dict


def get_default_optimizer_config() -> Dict[str, object]:
    _data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    return {
        "heuristic_estimated_tank_volume_liters": 10.0,
        # 수온 관련 휴리스틱
        "heuristic_temp_drop_per_cooler_min": 0.18,
        "heuristic_temp_drop_per_fan_min_when_cooling": 0.03,
        "heuristic_temp_stabilize_per_circulation_min": 0.02,
        # 유량비 관련 휴리스틱
        "heuristic_fr_gain_per_circulation_min": 0.025,
        # EC 관련 휴리스틱
        "heuristic_ec_gain_per_nutrient_ml": 0.015,
        # 장치별 소비전력 (W) — 실제 하드웨어 기준
        "estimated_main_water_pump_watt": 8.0,    # D5 BTS7960
        "estimated_nutrient_pump_watt": 5.0,
        "estimated_oxygen_pump_watt": 4.0,         # D6
        "estimated_cooling_pump_watt": 5.0,        # D7
        "estimated_cooling_fan_watt": 3.0,         # D8 (2개)
        "estimated_led_bar_watt": 20.0,            # D9
        "estimated_ventilation_fan_watt": 3.0,     # D10
        "estimated_peltier_1_watt": 60.0,          # D11
        "estimated_peltier_2_watt": 60.0,          # D12
        "estimated_nutrient_pump_flow_ml_per_min": 30.0,
        # 제어 후보값
        "circulation_pump_minutes_candidates": [0, 3, 5, 10, 15],
        "nutrient_pump_ml_candidates": [0, 5, 10, 20],
        "aeration_minutes_candidates": [0, 5, 10, 15],
        "cooler_minutes_candidates": [0, 3, 5, 10],
        "fan_minutes_candidates": [0, 5, 10, 15],
        "led_minutes_candidates": [0],
        "candidate_limit": None,
        "top_k_candidates": 5,
        "history_max_records": 2000,
        "energy_normalizer_wh": 10.0,
        "nutrient_resource_normalizer_ml": 20.0,
        "risk_reduction_threshold": 1.0,
        "alpha": 1.0,
        "beta": 8.0,
        "gamma": 5.0,
        "delta": 2.0,
        "eta": 6.0,
        "theta": 5.0,
        "ec_nutrient_penalty_threshold": 0.25,
        "ec_nutrient_penalty_threshold_by_mode": {"water": 0.25, "hydro": 2.5},
        "low_temperature_cooler_penalty_threshold": 15.0,
        "normal_fr_circulation_penalty_threshold": 0.9,
        "excessive_circulation_minutes_threshold": 10.0,
        "input_clamps": {
            "water_temperature": (-5.0, 45.0),
            "flow_ratio": (0.0, 3.0),
            "EC": (0.0, 10.0),
            "pH": (0.0, 14.0),
            "turbidity": (0.0, 100.0),
            "air_temperature": (-10.0, 60.0),
            "humidity": (0.0, 100.0),
        },
        "history_inputs_path": str(_data_dir / "history_inputs.jsonl"),
        "history_packets_path": str(_data_dir / "history_packets.jsonl"),
        "control_history_path": str(_data_dir / "device_control_history.jsonl"),
        "device_control_output_path": str(_data_dir / "device_control_outputs.json"),
        "device_control_history_path": str(_data_dir / "device_control_history.jsonl"),
    }
