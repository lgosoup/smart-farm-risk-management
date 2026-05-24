# main.py
"""
Driver to run the fuzzy risk assessment and emit an evidence packet.

Features:
- Inputs are stored as JSONL history (raw sensor values; no modification).
- Time-series flags are computed from recent input history and included in the packet.
- Sensor integrity flags are computed (flags only; no clipping) and included in the packet.
- Alert escalation is applied to action_key based on recent repeated 위험 labels.
- JSONL retention trimming is applied to keep files bounded.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from explain import build_evidence_packet
from fuzzify import (
    compute_delta_flags,
    compute_integrity_flags,
    compute_timeseries_flags,
    fuzzify_inputs,
    load_default_config,
    preprocess_inputs,
)
from inference import infer
from input_loader import load_inputs

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_crop_ec_mode(cfg: Dict[str, object]) -> str:
    """Get ec_mode from loaded crop config, defaulting to 'water'."""
    return str(cfg.get("ec_mode_active", cfg.get("ec_mode_default", "water")))


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _append_jsonl(path: str, obj: Dict[str, object]) -> None:
    p = Path(path)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _read_jsonl(path: str) -> List[Dict[str, object]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: List[Dict[str, object]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _read_jsonl_last_n(path: str, n: int) -> List[Dict[str, object]]:
    if n <= 0:
        return []
    rows = _read_jsonl(path)
    return rows[-n:]


def _trim_jsonl_keep_last(path: str, max_records: int) -> None:
    if max_records <= 0:
        return
    p = Path(path)
    if not p.exists():
        return
    rows = _read_jsonl(path)
    if len(rows) <= max_records:
        return
    keep = rows[-max_records:]
    with p.open("w", encoding="utf-8") as f:
        for obj in keep:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _extract_sensor_series(history_rows: List[Dict[str, object]]) -> List[Dict[str, Optional[float]]]:
    series: List[Dict[str, Optional[float]]] = []
    for r in history_rows:
        sensor = r.get("sensor")
        if isinstance(sensor, dict):
            series.append({k: (float(v) if isinstance(v, (int, float)) else None) for k, v in sensor.items()})
    return series


def _extract_recent_risk_labels(history_packets: List[Dict[str, object]], max_n: int) -> List[str]:
    labels: List[str] = []
    for pkt in history_packets[-max_n:]:
        risk = pkt.get("risk")
        if isinstance(risk, dict):
            lbl = risk.get("risk_label")
            if isinstance(lbl, str):
                labels.append(lbl)
    return labels


def run_assessment(
    raw_inputs: Dict[str, float],
    ec_mode: str = "water",
    *,
    history_inputs: Optional[List[Dict[str, Optional[float]]]] = None,
    recent_risk_labels: Optional[Sequence[str]] = None,
    history_inputs_path: str = str(_DATA_DIR / "history_inputs.jsonl"),
    history_packets_path: str = str(_DATA_DIR / "history_packets.jsonl"),
    save_history: bool = True,
) -> Dict[str, object]:
    cfg = load_default_config()
    if ec_mode == "water":  # only auto-override if using default
        ec_mode = _load_crop_ec_mode(cfg)
    now = _now_iso()

    log_cfg = cfg.get("logging_policy", {}) or {}
    inputs_max = int(log_cfg.get("inputs_max_records", 2000))
    packets_max = int(log_cfg.get("packets_max_records", 2000))

    if history_inputs is None:
        win = int((cfg.get("timeseries_policy", {}) or {}).get("window", 5))
        hist_rows = _read_jsonl_last_n(history_inputs_path, max(inputs_max, win + 20))
        history_inputs = _extract_sensor_series(hist_rows)

    if recent_risk_labels is None:
        pkt_rows = _read_jsonl_last_n(history_packets_path, max(50, packets_max))
        recent_risk_labels = _extract_recent_risk_labels(pkt_rows, max_n=10)

    previous_inputs = history_inputs[-1] if history_inputs else None

    processed = preprocess_inputs(raw_inputs, cfg, history=history_inputs)
    memberships = fuzzify_inputs(processed, cfg, ec_mode=ec_mode)

    delta_flags = compute_delta_flags(processed, previous_inputs, cfg)
    integrity_flags = compute_integrity_flags(processed, previous_inputs, cfg)
    ts_flags_dict = compute_timeseries_flags(processed, history_inputs, cfg)

    merged_flags: Dict[str, bool] = {}
    merged_flags.update({k: bool(v) for k, v in (delta_flags or {}).items()})
    merged_flags.update({k: True for k in ts_flags_dict.keys()})

    risk_score, risk_label, score_membership, label_peaks, fired_rules = infer(memberships, cfg, delta_flags=merged_flags)

    packet = build_evidence_packet(
        raw_inputs=processed,
        memberships=memberships,
        risk_score=risk_score,
        risk_label=risk_label,
        score_membership=score_membership,
        fired_rules=fired_rules,
        timestamp=now,
        top_n=cfg["rule_execution"]["top_n"],
        aggregated_label_peaks=label_peaks,
        timeseries_flags=list(ts_flags_dict.keys()),
        integrity_flags=integrity_flags,
        recent_risk_labels=list(recent_risk_labels or []),
        alert_policy=cfg.get("alert_policy"),
    )

    if save_history:
        _append_jsonl(history_inputs_path, {"ts": now, "sensor": processed})
        _append_jsonl(history_packets_path, packet)
        _trim_jsonl_keep_last(history_inputs_path, inputs_max)
        _trim_jsonl_keep_last(history_packets_path, packets_max)

    return packet


if __name__ == "__main__":
    inputs = load_inputs(str(_DATA_DIR / "inputs.json"))
    packet = run_assessment(inputs, ec_mode="water")

    out_path = _DATA_DIR / "outputs.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(packet, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote {out_path}")
