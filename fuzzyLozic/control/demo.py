from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from main import run_assessment

from control.config import get_default_optimizer_config
from control.optimizer import optimize_control, should_optimize
from control.output import build_monitor_only_output


def run_demo_scenarios(ec_mode: str = "water") -> List[Dict[str, object]]:
    config = get_default_optimizer_config()
    config["assessment_context"] = {
        "history_inputs": [],
        "recent_risk_labels": [],
        "history_inputs_path": str(Path(config["history_inputs_path"])),
        "history_packets_path": str(Path(config["history_packets_path"])),
    }
    samples = [
        {"T": 18.0, "DO": 6.0, "pH": 6.3, "FR": 1.0, "EC": 0.5, "Turbidity": 1.0},
        {"T": 12.5, "DO": 9.5, "pH": 6.5, "FR": 1.0, "EC": 0.1, "Turbidity": 0.5},
    ]
    results: List[Dict[str, object]] = []
    for sample in samples:
        packet = run_assessment(sample, ec_mode=ec_mode, history_inputs=[], recent_risk_labels=[], save_history=False)
        results.append(optimize_control(sample, packet, ec_mode, config) if should_optimize(packet) else build_monitor_only_output(sample, packet, config))
    return results
