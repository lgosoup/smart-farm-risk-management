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
    inputs = {"T": 12.5, "DO": 9.5, "pH": 6.5, "FR": 1.0, "EC": 0.12, "Turbidity": 0.5}
    score, risk_label, *_ = _run(inputs, ec_mode="water")
    assert risk_label in {"매우안정", "안정", "경계", "주의"}


def test_do_critical_is_danger():
    # DO=3.0: 바질 기준 치명적 낮음 구간(0~3.5) 내 → R1 발화 → 위험
    inputs = {"T": 12.0, "DO": 3.0, "pH": 6.6, "FR": 1.0, "EC": 0.1, "Turbidity": 0.5}
    _, risk_label, *_ = _run(inputs, ec_mode="water")
    assert risk_label == "위험"


def test_temp_rising_and_do_boundary():
    # 바질 기준: T=31.0(경고(고온) 진입), DO=6.5(낮음 중심) → 경고 이상
    inputs = {"T": 31.0, "DO": 6.5, "pH": 6.6, "FR": 1.0, "EC": 0.1, "Turbidity": 0.8}
    _, risk_label, *_ = _run(inputs, ec_mode="water")
    assert risk_label in {"경고", "위험"}


def test_flow_low_and_turbid():
    inputs = {"T": 12.0, "DO": 9.0, "pH": 6.5, "FR": 0.55, "EC": 0.12, "Turbidity": 10.0}
    _, risk_label, *_ = _run(inputs, ec_mode="water")
    assert risk_label in {"주의", "경고", "위험"}


def test_ec_mode_branching():
    inputs = {"T": 12.0, "DO": 8.0, "pH": 6.5, "FR": 1.0, "EC": 3.5, "Turbidity": 0.5}
    score_water, label_water, *_ = _run(inputs, ec_mode="water")
    score_hydro, label_hydro, *_ = _run(inputs, ec_mode="hydro")
    assert (label_water != label_hydro) or (abs(score_water - score_hydro) > 1e-3)


def test_timeseries_flags_emitted():
    cfg = load_default_config()
    history = [
        {"T": 12.0, "DO": 9.0, "FR": 1.0},
        {"T": 12.4, "DO": 8.6, "FR": 0.95},
        {"T": 13.0, "DO": 8.2, "FR": 0.90},
        {"T": 13.8, "DO": 7.8, "FR": 0.82},
    ]
    current = {"T": 14.2, "DO": 7.5, "pH": 6.5, "FR": 0.78, "EC": 0.12, "Turbidity": 0.8}
    flags = compute_timeseries_flags(current, history, cfg)
    assert any(k.startswith("TS_TREND_") for k in flags.keys())


def test_alert_escalation_action_key():
    inputs = {"T": 20.0, "DO": 4.8, "pH": 6.6, "FR": 0.6, "EC": 0.12, "Turbidity": 0.8}
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
    inputs = {"T": 13.0, "DO": 9.2, "pH": 6.6, "FR": 1.05, "EC": 0.15, "Turbidity": 0.9}
    packet = run_assessment(inputs, ec_mode="water", history_inputs=[], recent_risk_labels=[], save_history=False)
    assert "risk" in packet and "rules_fired_topN" in packet
    assert "timeseries" in packet and "sensor_integrity" in packet and "recommendations" in packet
    assert packet["risk"]["risk_label"] in {"매우안정", "안정", "경계", "주의", "경고", "위험"}
