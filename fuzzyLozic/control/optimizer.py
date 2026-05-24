from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

try:
    from main import run_assessment
except ModuleNotFoundError:  # pragma: no cover
    import sys

    BASE_DIR = Path(__file__).resolve().parent.parent
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    from main import run_assessment

from control.candidates import generate_candidates, raw_candidate_count
from control.commands import build_device_commands
from control.config import get_default_optimizer_config
from control.constants import DANGER, EMERGENCY, OPTIMIZE, WARNING
from control.history import load_assessment_context, load_previous_control
from control.prediction import predict_next_inputs
from control.scoring import (
    calculate_change_penalty,
    calculate_energy_wh,
    calculate_over_control_penalty,
    calculate_resource_cost,
    calculate_weak_reduction_penalty,
)
from control.utils import now_iso, risk_label, risk_value, round_numeric_dict


def should_optimize(packet: Dict[str, object]) -> bool:
    return packet.get("action_key") in {WARNING, DANGER, EMERGENCY}


def evaluate_candidate(
    current_inputs: Dict[str, float],
    current_packet: Dict[str, object],
    control: Dict[str, float],
    previous_control: Optional[Dict[str, float]],
    ec_mode: str,
    config: Dict[str, object],
) -> Dict[str, object]:
    predicted_inputs = predict_next_inputs(current_inputs, control, config)
    context = load_assessment_context(config)
    predicted_packet = run_assessment(
        predicted_inputs,
        ec_mode=ec_mode,
        history_inputs=context["history_inputs"],
        recent_risk_labels=context["recent_risk_labels"],
        history_inputs_path=context["history_inputs_path"],
        history_packets_path=context["history_packets_path"],
        save_history=False,
    )

    current_risk_score = risk_value(current_packet, "risk_score")
    predicted_risk_score = risk_value(predicted_packet, "risk_score")
    energy_wh = calculate_energy_wh(control, config)
    energy_cost_norm = energy_wh / float(config["energy_normalizer_wh"])
    resource_cost_norm = calculate_resource_cost(control, config)
    change_penalty = calculate_change_penalty(control, previous_control, config)
    over_control_penalty = calculate_over_control_penalty(current_inputs, control, config, ec_mode)
    weak_reduction_penalty = calculate_weak_reduction_penalty(
        current_risk_score,
        predicted_risk_score,
        energy_wh,
        resource_cost_norm,
        config,
    )
    risk_reduction = current_risk_score - predicted_risk_score

    objective = (
        float(config["alpha"]) * predicted_risk_score
        + float(config["beta"]) * energy_cost_norm
        + float(config["gamma"]) * resource_cost_norm
        + float(config["delta"]) * change_penalty
        + float(config["eta"]) * over_control_penalty
        + float(config["theta"]) * weak_reduction_penalty
    )

    return {
        "control": dict(control),
        "predicted_inputs": predicted_inputs,
        "predicted_packet": predicted_packet,
        "predicted_risk_score": predicted_risk_score,
        "predicted_risk_label": risk_label(predicted_packet),
        "predicted_action_key": predicted_packet.get("action_key"),
        "risk_reduction": risk_reduction,
        "energy_wh": energy_wh,
        "energy_cost_norm": energy_cost_norm,
        "resource_cost_norm": resource_cost_norm,
        "change_penalty": change_penalty,
        "over_control_penalty": over_control_penalty,
        "weak_reduction_penalty": weak_reduction_penalty,
        "J": objective,
    }


def optimize_control(
    current_inputs: Dict[str, float],
    current_packet: Dict[str, object],
    ec_mode: str = "water",
    config: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    merged = merged_config(config)
    previous_control = load_previous_control(merged["control_history_path"])

    total_candidates = raw_candidate_count(merged)
    candidates = generate_candidates(merged)
    evaluated = [
        evaluate_candidate(current_inputs, current_packet, candidate, previous_control, ec_mode, merged)
        for candidate in candidates
    ]
    evaluated_sorted = sorted(
        evaluated,
        key=lambda item: (
            float(item["J"]),
            float(item["predicted_risk_score"]),
            float(item["energy_wh"]),
            float(item["resource_cost_norm"]),
        ),
    )

    current_risk_score = risk_value(current_packet, "risk_score")
    improving = [item for item in evaluated_sorted if float(item["predicted_risk_score"]) < current_risk_score]
    selected = improving[0] if improving else evaluated_sorted[0]

    top_k = [
        candidate_brief(item, rank)
        for rank, item in enumerate(evaluated_sorted[: int(merged["top_k_candidates"])], start=1)
    ]

    return {
        "timestamp": now_iso(),
        "control_mode": OPTIMIZE,
        "trigger": {
            "enabled": True,
            "reason": "action_key\uac00 \uacbd\uace0 \uc774\uc0c1",
            "action_key": current_packet.get("action_key"),
            "risk_label": risk_label(current_packet),
            "risk_score": round(current_risk_score, 4),
        },
        "current_inputs": round_numeric_dict(current_inputs),
        "selected_control": round_numeric_dict(selected["control"]),
        "device_commands": build_device_commands(selected["control"], merged),
        "prediction": {
            "predicted_inputs": round_numeric_dict(selected["predicted_inputs"]),
            "predicted_risk": {
                "risk_score": round(float(selected["predicted_risk_score"]), 4),
                "risk_label": selected["predicted_risk_label"],
                "action_key": selected["predicted_action_key"],
            },
        },
        "objective": {
            "J": round(float(selected["J"]), 4),
            "predicted_risk_score": round(float(selected["predicted_risk_score"]), 4),
            "risk_reduction": round(float(selected["risk_reduction"]), 4),
            "energy_wh": round(float(selected["energy_wh"]), 4),
            "energy_cost_norm": round(float(selected["energy_cost_norm"]), 4),
            "resource_cost_norm": round(float(selected["resource_cost_norm"]), 4),
            "change_penalty": round(float(selected["change_penalty"]), 4),
            "over_control_penalty": round(float(selected["over_control_penalty"]), 4),
            "weak_reduction_penalty": round(float(selected["weak_reduction_penalty"]), 4),
            "weights": {
                "alpha": float(merged["alpha"]),
                "beta": float(merged["beta"]),
                "gamma": float(merged["gamma"]),
                "delta": float(merged["delta"]),
                "eta": float(merged["eta"]),
                "theta": float(merged["theta"]),
            },
        },
        "candidate_summary": {
            "total_candidates": total_candidates,
            "valid_candidates": len(candidates),
            "top_k": top_k,
        },
        "estimated_total_wh": round(float(selected["energy_wh"]), 4),
        "measured_total_wh": None,
        "measurement_source": None,
        "bandit_feedback": {
            "enabled": False,
            "reward": None,
            "comment": "\ud5a5\ud6c4 \uc2e4\uc81c \uc704\ud5d8\ub3c4 \uac10\uc18c\ub7c9\uacfc \uc2e4\uc81c \uc804\ub825 \uc0ac\uc6a9\ub7c9 \uae30\ubc18 \ubcf4\uc0c1 \uacc4\uc0b0 \uc608\uc815",
        },
        "notes": [
            "outputs.json\uc740 \ud604\uc7ac inputs.json \uae30\uc900\uc73c\ub85c \ucd5c\uc2e0\ud654\ud568",
            "\uae30\uc874 history_inputs.jsonl\uacfc history_packets.jsonl\uc740 \uc218\uc815\ud558\uc9c0 \uc54a\uc74c",
            "\uc608\uce21 \ubaa8\ub378\uc740 \uc18c\ud615 \uc218\uacbd\uc7ac\ubc30 \ubc15\uc2a4 \uae30\uc900 \ucd08\uae30 \ud734\ub9ac\uc2a4\ud2f1 \uc120\ud615 \uadfc\uc0ac\uc774\uba70 \ud5a5\ud6c4 \uc2e4\uce21 \ub370\uc774\ud130\ub85c \ubcf4\uc815 \ud544\uc694",
            "\uc608\uc0c1 \uc804\ub825\uc740 \ucd94\uc815\uac12\uc774\uba70 \uc2e4\uc81c \uc804\ub825 \uce21\uc815\uac12\uacfc \ube44\uad50 \uac00\ub2a5",
        ],
    }


def candidate_brief(item: Dict[str, object], rank: int) -> Dict[str, object]:
    return {
        "rank": rank,
        "J": round(float(item["J"]), 4),
        "control": round_numeric_dict(item["control"]),
        "predicted_risk_score": round(float(item["predicted_risk_score"]), 4),
        "predicted_risk_label": item["predicted_risk_label"],
        "risk_reduction": round(float(item["risk_reduction"]), 4),
        "energy_wh": round(float(item["energy_wh"]), 4),
        "resource_cost_norm": round(float(item["resource_cost_norm"]), 4),
    }


def merged_config(config: Optional[Dict[str, object]]) -> Dict[str, object]:
    merged = get_default_optimizer_config()
    if config:
        merged.update(config)
    return merged
