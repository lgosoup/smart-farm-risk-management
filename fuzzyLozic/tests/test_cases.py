# test_cases.py
import pytest

from fuzzify import (
    compute_timeseries_flags,
    load_default_config,
    preprocess_inputs,
    fuzzify_inputs,
)
from inference import infer
from main import run_assessment


def _run(inputs, ec_mode=None):
    cfg = load_default_config()
    processed = preprocess_inputs(inputs, cfg)
    memberships = fuzzify_inputs(processed, cfg, ec_mode=ec_mode)
    return infer(memberships, cfg)


def test_normal_stable():
    inputs = {"water_temperature": 12.5, "pH": 6.5, "flow_ratio": 1.0, "EC": 0.12, "turbidity": 0.5, "air_temperature": 22.0, "humidity": 55.0}
    score, risk_label, *_ = _run(inputs, ec_mode="water")
    assert risk_label in {"매우안정", "안정", "경계", "주의"}


def test_high_temp_danger():
    # water_temperature emergency level → R2 fires → 위험
    inputs = {"water_temperature": 34.0, "pH": 6.6, "flow_ratio": 1.0, "EC": 0.1, "turbidity": 0.5, "air_temperature": 35.0, "humidity": 60.0}
    _, risk_label, *_ = _run(inputs, ec_mode="water")
    assert risk_label in {"경고", "위험"}


def test_temp_rising_and_flow_low():
    # 고온 + 유량 저하 → R6 발화 → 경고 이상
    inputs = {"water_temperature": 31.0, "pH": 6.6, "flow_ratio": 0.5, "EC": 0.1, "turbidity": 0.8, "air_temperature": 30.0, "humidity": 65.0}
    _, risk_label, *_ = _run(inputs, ec_mode="water")
    assert risk_label in {"경고", "위험"}


def test_flow_low_and_turbid():
    inputs = {"water_temperature": 12.0, "pH": 6.5, "flow_ratio": 0.55, "EC": 0.12, "turbidity": 10.0, "air_temperature": 22.0, "humidity": 55.0}
    _, risk_label, *_ = _run(inputs, ec_mode="water")
    assert risk_label in {"주의", "경고", "위험"}


def test_ec_mode_branching():
    inputs = {"water_temperature": 12.0, "pH": 6.5, "flow_ratio": 1.0, "EC": 3.5, "turbidity": 0.5, "air_temperature": 22.0, "humidity": 55.0}
    score_water, label_water, *_ = _run(inputs, ec_mode="water")
    score_hydro, label_hydro, *_ = _run(inputs, ec_mode="hydro")
    assert (label_water != label_hydro) or (abs(score_water - score_hydro) > 1e-3)


def test_timeseries_flags_emitted():
    cfg = load_default_config()
    history = [
        {"water_temperature": 12.0, "flow_ratio": 1.0},
        {"water_temperature": 12.4, "flow_ratio": 0.95},
        {"water_temperature": 13.0, "flow_ratio": 0.90},
        {"water_temperature": 13.8, "flow_ratio": 0.82},
    ]
    current = {"water_temperature": 14.2, "pH": 6.5, "flow_ratio": 0.78, "EC": 0.12, "turbidity": 0.8, "air_temperature": 24.0, "humidity": 58.0}
    flags = compute_timeseries_flags(current, history, cfg)
    assert any(k.startswith("TS_TREND_") for k in flags.keys())


def test_alert_escalation_action_key():
    inputs = {"water_temperature": 20.0, "pH": 6.6, "flow_ratio": 0.6, "EC": 0.12, "turbidity": 0.8, "air_temperature": 28.0, "humidity": 62.0}
    packet = run_assessment(
        inputs,
        ec_mode="water",
        history_inputs=[],
        recent_risk_labels=["위험", "위험"],
        save_history=False,
    )
    assert packet["risk"]["risk_label"] == "위험"
    assert packet["action_key"] == "긴급"


def test_driver_end_to_end_packet_fields():
    inputs = {"water_temperature": 13.0, "pH": 6.6, "flow_ratio": 1.05, "EC": 0.15, "turbidity": 0.9, "air_temperature": 23.0, "humidity": 57.0}
    packet = run_assessment(inputs, ec_mode="water", history_inputs=[], recent_risk_labels=[], save_history=False)
    assert "risk" in packet and "rules_fired_topN" in packet
    assert "timeseries" in packet and "sensor_integrity" in packet and "recommendations" in packet
    assert packet["risk"]["risk_label"] in {"매우안정", "안정", "경계", "주의", "경고", "위험"}
