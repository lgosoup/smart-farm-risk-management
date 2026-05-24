from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from control.commands import build_device_commands
from control.constants import MONITOR_ONLY
from control.utils import now_iso, risk_label, risk_value, round_numeric_dict, trim_jsonl_keep_last, zero_control


def build_monitor_only_output(
    current_inputs: Dict[str, float],
    current_packet: Dict[str, object],
    config: Dict[str, object],
) -> Dict[str, object]:
    zero = zero_control()
    return {
        "timestamp": now_iso(),
        "control_mode": MONITOR_ONLY,
        "trigger": {
            "enabled": False,
            "reason": "action_key\uac00 \uacbd\uace0 \ubbf8\ub9cc",
            "action_key": current_packet.get("action_key"),
            "risk_label": risk_label(current_packet),
            "risk_score": round(risk_value(current_packet, "risk_score"), 4),
        },
        "current_inputs": round_numeric_dict(current_inputs),
        "selected_control": zero,
        "device_commands": build_device_commands(zero, config),
        "prediction": None,
        "objective": None,
        "candidate_summary": None,
        "estimated_total_wh": 0.0,
        "measured_total_wh": None,
        "measurement_source": None,
        "bandit_feedback": {
            "enabled": False,
            "reward": None,
            "comment": "\ud5a5\ud6c4 \uc2e4\uc81c \uc704\ud5d8\ub3c4 \uac10\uc18c\ub7c9\uacfc \uc2e4\uc81c \uc804\ub825 \uc0ac\uc6a9\ub7c9 \uae30\ubc18 \ubcf4\uc0c1 \uacc4\uc0b0 \uc608\uc815",
        },
        "notes": [
            "\uacbd\uace0 \uc774\uc0c1\uc774 \uc544\ub2c8\ubbc0\ub85c \uc7a5\uce58 \ucd5c\uc801\ud654 \ubbf8\uc218\ud589",
            "outputs.json\uc740 \ud604\uc7ac inputs.json \uae30\uc900\uc73c\ub85c \ucd5c\uc2e0\ud654\ud568",
        ],
    }


def save_control_output(
    result: Dict[str, object],
    output_path: str | Path,
    history_path: str | Path,
    max_records: int,
) -> None:
    output_file = Path(output_path)
    history_file = Path(history_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    with history_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    trim_jsonl_keep_last(history_file, max_records)
