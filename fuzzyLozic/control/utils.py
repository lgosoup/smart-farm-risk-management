from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from control.constants import CONTROL_KEYS


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows: List[Dict[str, object]] = []
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
                rows.append(row)
    return rows


def trim_jsonl_keep_last(path: str | Path, max_records: int) -> None:
    if max_records <= 0:
        return
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return
    rows = read_jsonl(jsonl_path)
    if len(rows) <= max_records:
        return
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows[-max_records:]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def risk_label(packet: Dict[str, object]) -> Optional[str]:
    risk = packet.get("risk")
    return str(risk["risk_label"]) if isinstance(risk, dict) and isinstance(risk.get("risk_label"), str) else None


def risk_value(packet: Dict[str, object], key: str) -> float:
    risk = packet.get("risk")
    return float(risk[key]) if isinstance(risk, dict) and isinstance(risk.get(key), (int, float)) else 0.0


def round_numeric_dict(values: Dict[str, float]) -> Dict[str, float]:
    rounded: Dict[str, float] = {}
    for key, value in values.items():
        numeric = round(float(value), 4)
        rounded[key] = int(numeric) if float(numeric).is_integer() else numeric
    return rounded


def zero_control() -> Dict[str, float]:
    return {key: 0 for key in CONTROL_KEYS}
