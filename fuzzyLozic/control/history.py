from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from control.constants import CONTROL_KEYS
from control.utils import read_jsonl


def load_previous_control(history_path: str | Path) -> Optional[Dict[str, float]]:
    path = Path(history_path)
    if not path.exists():
        return None
    last_valid: Optional[Dict[str, object]] = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                last_valid = row
    if not isinstance(last_valid, dict):
        return None
    selected = last_valid.get("selected_control")
    if not isinstance(selected, dict):
        return None
    result: Dict[str, float] = {}
    for key in CONTROL_KEYS:
        value = selected.get(key)
        if isinstance(value, (int, float)):
            result[key] = float(value)
    return result or None


def load_assessment_context(config: Dict[str, object]) -> Dict[str, object]:
    cached = config.get("_assessment_context_cache")
    if isinstance(cached, dict):
        return cached
    supplied = config.get("assessment_context")
    if isinstance(supplied, dict):
        context = {
            "history_inputs": list(supplied.get("history_inputs", [])),
            "recent_risk_labels": list(supplied.get("recent_risk_labels", [])),
            "history_inputs_path": str(supplied.get("history_inputs_path", config["history_inputs_path"])),
            "history_packets_path": str(supplied.get("history_packets_path", config["history_packets_path"])),
        }
        config["_assessment_context_cache"] = context
        return context

    history_inputs_path = Path(str(config["history_inputs_path"]))
    history_packets_path = Path(str(config["history_packets_path"]))
    history_inputs = []
    for row in read_jsonl(history_inputs_path):
        sensor = row.get("sensor")
        if isinstance(sensor, dict):
            history_inputs.append({key: float(value) for key, value in sensor.items() if isinstance(value, (int, float))})

    recent_risk_labels = []
    for row in read_jsonl(history_packets_path)[-10:]:
        risk = row.get("risk")
        if isinstance(risk, dict) and isinstance(risk.get("risk_label"), str):
            recent_risk_labels.append(str(risk["risk_label"]))

    context = {
        "history_inputs": history_inputs,
        "recent_risk_labels": recent_risk_labels,
        "history_inputs_path": str(history_inputs_path),
        "history_packets_path": str(history_packets_path),
    }
    config["_assessment_context_cache"] = context
    return context
