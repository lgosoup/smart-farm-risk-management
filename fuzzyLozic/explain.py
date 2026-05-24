from __future__ import annotations

from typing import Dict, Iterable, List, Sequence


VERY_STABLE = "\ub9e4\uc6b0\uc548\uc815"
STABLE = "\uc548\uc815"
BOUNDARY = "\uacbd\uacc4"
CAUTION = "\uc8fc\uc758"
WARNING = "\uacbd\uace0"
DANGER = "\uc704\ud5d8"
EMERGENCY = "\uae34\uae09"


def _round4(value: object) -> object:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {k: _round4(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round4(v) for v in value]
    return value


def _top_rules(fired_rules: Sequence[Dict[str, object]], top_n: int) -> List[Dict[str, object]]:
    ordered = sorted(fired_rules, key=lambda row: float(row.get("firing_strength", 0.0)), reverse=True)
    return [_round4(dict(row)) for row in ordered[:top_n]]


def _sum_drivers(fired_rules: Iterable[Dict[str, object]]) -> Dict[str, float]:
    drivers: Dict[str, float] = {}
    for row in fired_rules:
        label = row.get("conclusion_label")
        if not isinstance(label, str):
            continue
        drivers[label] = round(drivers.get(label, 0.0) + float(row.get("firing_strength", 0.0)), 4)
    return drivers


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _build_recommendations(
    fired_rules: Sequence[Dict[str, object]],
    timeseries_flags: Sequence[str],
    integrity_flags: Dict[str, object],
    action_key: str,
) -> Dict[str, List[str]]:
    scenario_notes: List[str] = []
    checklist: List[str] = []

    out_of_range = integrity_flags.get("out_of_range", {}) if isinstance(integrity_flags, dict) else {}
    spike = integrity_flags.get("spike", {}) if isinstance(integrity_flags, dict) else {}

    if isinstance(out_of_range, dict):
        for var in out_of_range.keys():
            scenario_notes.append(f"{var} sensor is outside the allowed operating range.")
            checklist.append(f"Inspect the {var} sensor reading and the real process condition.")

    if isinstance(spike, dict):
        for var, info in spike.items():
            direction = info.get("direction") if isinstance(info, dict) else None
            delta = info.get("delta") if isinstance(info, dict) else None
            if isinstance(delta, (int, float)):
                scenario_notes.append(f"Single-step spike detected on {var} ({direction}, delta={delta:+.2f}).")
            else:
                scenario_notes.append(f"Single-step spike detected on {var}.")
            checklist.append(f"Verify whether the {var} change is real or caused by sensor noise.")

    for flag in timeseries_flags:
        if flag.startswith("TS_PERSIST_T_HIGH"):
            scenario_notes.append("High water temperature has persisted across recent samples.")
            checklist.append("Prioritize cooling or heat rejection.")
        elif flag.startswith("TS_TREND_T_UP"):
            scenario_notes.append("Water temperature is trending upward.")
            checklist.append("Inspect the cooling path and nearby heat sources.")
        elif flag.startswith("TS_TREND_FR_DOWN"):
            scenario_notes.append("Flow ratio is trending downward.")
            checklist.append("Inspect pump output, filter blockage, and tubing.")
        elif flag.startswith("TS_RECOVERY_FAIL_FR"):
            scenario_notes.append("Flow recovery has not appeared in recent samples.")
        elif flag.startswith("TS_VOL_"):
            scenario_notes.append(f"Short-window variability increased ({flag}).")
        elif flag.startswith("TS_SPIKE_"):
            scenario_notes.append(f"Time-series spike flag was raised ({flag}).")

    for row in fired_rules:
        rule_id = row.get("rule_id")
        if rule_id == "R2":
            scenario_notes.append("Emergency-level water temperature is driving the current risk state.")
            checklist.append("Urgently reduce water temperature.")
        elif rule_id == "R3":
            scenario_notes.append("Flow stagnation is driving the current risk state.")
            checklist.append("Urgently restore circulation.")
        elif rule_id == "R6":
            scenario_notes.append("High temperature combined with low flow is a major composite risk.")
        elif rule_id == "R7":
            scenario_notes.append("High EC combined with low flow is a major composite risk.")
        elif rule_id == "R8":
            scenario_notes.append("Low flow and poor water clarity are increasing contamination risk.")
        elif rule_id == "R9":
            scenario_notes.append("High EC combined with borderline flow is elevating risk.")
        elif rule_id == "R16":
            checklist.append("Prioritize temperature reduction first.")
        elif rule_id == "R17":
            checklist.append("Inspect circulation recovery first.")
        elif rule_id == "R18":
            checklist.append("Inspect water clarity and contamination sources.")
        elif rule_id == "R19":
            checklist.append("Inspect EC increase sources before adding more nutrient.")

    if action_key in {WARNING, DANGER, EMERGENCY}:
        checklist.append("Confirm that actuators are responding after intervention.")

    return {
        "scenario_notes": _dedupe_keep_order(scenario_notes),
        "checklist": _dedupe_keep_order(checklist),
    }


def _resolve_packet_risk_label(
    inferred_risk_label: str,
    risk_score: float,
    fired_rules: Sequence[Dict[str, object]],
    aggregated_label_peaks: Dict[str, float],
) -> str:
    if inferred_risk_label != WARNING:
        return inferred_risk_label

    strong_danger_rules = 0
    for row in fired_rules:
        if row.get("conclusion_label") == DANGER and float(row.get("firing_strength", 0.0)) >= 0.2:
            strong_danger_rules += 1

    danger_peak = float(aggregated_label_peaks.get(DANGER, 0.0))
    if risk_score >= 75.0 and strong_danger_rules >= 3 and danger_peak >= 0.2:
        return DANGER
    return inferred_risk_label


def _resolve_action_key(
    risk_label: str,
    recent_risk_labels: Sequence[str],
    alert_policy: Dict[str, object] | None,
) -> str:
    if not isinstance(risk_label, str) or not risk_label:
        return BOUNDARY

    action_key = risk_label
    if not isinstance(alert_policy, dict) or not alert_policy.get("enabled"):
        return action_key

    danger_cfg = alert_policy.get("danger_escalation", {})
    if not isinstance(danger_cfg, dict):
        return action_key

    label = danger_cfg.get("label")
    escalated = danger_cfg.get("escalated_action_key")
    count = int(danger_cfg.get("count", 0))
    if not isinstance(label, str) or not isinstance(escalated, str) or count <= 1:
        return action_key

    if risk_label != label:
        return action_key

    consecutive = 1
    for previous in reversed(list(recent_risk_labels)):
        if previous == label:
            consecutive += 1
        else:
            break

    if consecutive >= count:
        return escalated
    return action_key


def build_evidence_packet(
    *,
    raw_inputs: Dict[str, object],
    memberships: Dict[str, Dict[str, float]],
    risk_score: float,
    risk_label: str,
    score_membership: Dict[str, float],
    fired_rules: Sequence[Dict[str, object]],
    timestamp: str,
    top_n: int,
    aggregated_label_peaks: Dict[str, float],
    timeseries_flags: Sequence[str],
    integrity_flags: Dict[str, object],
    recent_risk_labels: Sequence[str],
    alert_policy: Dict[str, object] | None,
) -> Dict[str, object]:
    packet_risk_label = _resolve_packet_risk_label(risk_label, float(risk_score), fired_rules, aggregated_label_peaks)
    action_key = _resolve_action_key(packet_risk_label, recent_risk_labels, alert_policy)
    packet = {
        "inputs": {**dict(raw_inputs), "timestamp": timestamp},
        "memberships": _round4(memberships),
        "risk": {
            "risk_score": round(float(risk_score), 4),
            "risk_label": packet_risk_label,
            "output_membership": _round4(score_membership),
            "aggregated_label_peaks": _round4(aggregated_label_peaks),
        },
        "rules_fired_topN": _top_rules(fired_rules, top_n),
        "drivers": _sum_drivers(fired_rules),
        "timeseries": {"flags": list(timeseries_flags)},
        "sensor_integrity": integrity_flags,
        "recommendations": _build_recommendations(fired_rules, timeseries_flags, integrity_flags, action_key),
        "action_key": action_key,
    }
    return packet
