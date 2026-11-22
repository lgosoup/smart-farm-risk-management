# test_regressions.py
"""오늘 감사에서 고친 버그들의 회귀 테스트."""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from config import load_config
from explain import build_evidence_packet
from fuzzify import fuzzify_inputs, load_default_config
from inference import infer
from main import run_assessment
from membership import trap, tri
from rules import build_rules


STABLE = {"water_temperature": 12.5, "pH": 6.5, "flow_ratio": 1.0, "EC": 0.12, "turbidity": 0.3}


def _infer(**overrides):
    cfg = load_default_config()
    x = dict(STABLE)
    x.update(overrides)
    m = fuzzify_inputs(x, cfg, ec_mode="water")
    return m, infer(m, cfg)


# ---------- membership: 숄더 끝점 ----------

def test_trap_left_and_right_shoulder_endpoints_are_full_membership():
    assert trap(0.0, 0.0, 0.35, 0.45, 0.0) == 1.0   # 유량 정체 at FR=0
    assert trap(88.0, 95.0, 100.0, 100.0, 100.0) == 1.0  # 습도 100%
    assert tri(0.0, 0.0, 1.0, 0.0) == 1.0
    assert tri(0.0, 1.0, 1.0, 1.0) == 1.0
    # 일반 모양은 그대로
    assert trap(1.0, 2.0, 3.0, 4.0, 1.0) == 0.0
    assert trap(1.0, 2.0, 3.0, 4.0, 1.5) == pytest.approx(0.5)
    assert tri(1.0, 2.0, 3.0, 3.0) == 0.0


@pytest.mark.parametrize(
    "overrides, var, label",
    [
        ({"flow_ratio": 0.0}, "flow_ratio", "정체(위험)"),
        ({"pH": 0.0}, "pH", "강산성(위험)"),
        ({"EC": 0.0}, "EC", "매우낮음"),
        ({"turbidity": 0.0}, "turbidity", "맑음(안정)"),
        ({"water_temperature": -5.0}, "water_temperature", "매우낮음"),
    ],
)
def test_range_endpoint_values_get_extreme_set(overrides, var, label):
    m, _ = _infer(**overrides)
    assert m[var][label] == 1.0


def test_flow_zero_is_danger_not_fallback():
    # 펌프 완전 정지(FR=0.0)가 아무 규칙도 발화하지 않아 50점 '경계'로 나오던 버그
    _, (score, label, _, _, fired) = _infer(flow_ratio=0.0)
    assert label == "위험"
    assert any(r["rule_id"] == "R3" for r in fired)


def test_ph_zero_is_danger_not_fallback():
    _, (score, label, _, _, fired) = _infer(pH=0.0)
    assert label == "위험"
    assert fired


# ---------- R10/R11 이중 계산 ----------

def test_r10_r11_identical_rules_do_not_double_count_into_r12():
    _, (_, _, _, _, fired) = _infer(water_temperature=16.0, flow_ratio=0.85)
    ids = {r["rule_id"] for r in fired}
    assert {"R10", "R11"} <= ids
    assert "R12" not in ids


def test_r12_still_promotes_on_two_distinct_warning_rules():
    # 수온 고온(R16 경고) + 유량 매우낮음(R17 경고) — 서로 다른 경고 규칙 2개
    _, (_, label, _, _, fired) = _infer(water_temperature=20.0, flow_ratio=0.45)
    assert "R12" in {r["rule_id"] for r in fired}
    assert label == "위험"


def test_r12_stub_premise_is_not_turbidity():
    cfg = load_default_config()
    m = fuzzify_inputs(dict(STABLE, turbidity=1.5), cfg, ec_mode="water")
    assert m["turbidity"]["약간 탁함(경계)"] > 0.9
    assert build_rules(cfg)["R12"].evaluate(m) == 0.0


# ---------- explain: 시계열 플래그 이름 ----------

def test_timeseries_recommendations_match_current_flag_names():
    pkt = build_evidence_packet(
        raw_inputs={},
        memberships={},
        risk_score=50.0,
        risk_label="주의",
        score_membership={},
        fired_rules=[],
        timestamp="t",
        top_n=5,
        aggregated_label_peaks={},
        timeseries_flags=[
            "TS_PERSIST_water_temperature_HIGH",
            "TS_TREND_water_temperature_UP",
            "TS_TREND_flow_ratio_DOWN",
            "TS_RECOVERY_FAIL_flow_ratio",
        ],
        integrity_flags={},
        recent_risk_labels=[],
        alert_policy=None,
    )
    notes = " ".join(pkt["recommendations"]["scenario_notes"])
    assert "persisted" in notes
    assert "trending upward" in notes
    assert "Flow ratio is trending downward" in notes
    assert "Flow recovery" in notes


# ---------- config: 잘못된 override ----------

def test_bad_membership_override_is_ignored(tmp_path):
    state = {"overrides": {"memberships": {"pH": {"안정(최적)": [6.0, 6.5]}, "flow_ratio": {"안정": [1.1, 1.0, 1.05, 1.2]}}}}
    sp = tmp_path / "state.json"
    sp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    cfg = load_config(state_path=str(sp))
    m = fuzzify_inputs(dict(STABLE), cfg, ec_mode="water")  # ValueError가 나면 안 된다
    assert m["pH"]["안정(최적)"] == 1.0
    shapes = {s.label: s.points for s in cfg["variables"]["flow_ratio"].shapes}
    assert shapes["안정"] == (0.9, 1.0, 1.05, 1.1)


# ---------- ec_mode: 패킷 ↔ 최적화 일치 ----------

def test_packet_records_effective_ec_mode(tmp_path, monkeypatch):
    import config

    crop = tmp_path / "crop.json"
    crop.write_text(json.dumps({"crop": "x", "ec_mode": "hydro"}), encoding="utf-8")
    monkeypatch.setattr(config, "CROP_CONFIG_PATH", str(crop))
    pkt = run_assessment(dict(STABLE, EC=1.5), ec_mode="water", history_inputs=[], recent_risk_labels=[], save_history=False)
    assert pkt["ec_mode"] == "hydro"


def test_optimizer_uses_packet_ec_mode(monkeypatch):
    import control.optimizer as opt

    seen = []
    real = opt.calculate_over_control_penalty

    def spy(current_inputs, control, config, ec_mode):
        seen.append(ec_mode)
        return real(current_inputs, control, config, ec_mode)

    monkeypatch.setattr(opt, "calculate_over_control_penalty", spy)
    inputs = dict(STABLE, flow_ratio=0.0, air_temperature=22.0, humidity=55.0)
    pkt = run_assessment(inputs, ec_mode="hydro", history_inputs=[], recent_risk_labels=[], save_history=False)
    cfg = {
        "assessment_context": {"history_inputs": [], "recent_risk_labels": []},
        "control_history_path": "__nonexistent__.jsonl",
        "circulation_pump_minutes_candidates": [0, 5],
        "nutrient_pump_ml_candidates": [0],
        "aeration_minutes_candidates": [0],
        "cooler_minutes_candidates": [0],
        "fan_minutes_candidates": [0],
    }
    opt.optimize_control(inputs, pkt, ec_mode="water", config=cfg)
    assert seen and set(seen) == {"hydro"}


# ---------- run_control_optimizer: 상대 경로 ----------

def test_run_control_optimizer_resolves_bare_input_name_to_data_dir():
    import run_control_optimizer as rco

    assert rco.resolve_path("inputs.json") == (rco.DATA_DIR / "inputs.json").resolve()


# ---------- improvement ----------

def test_vision_reward_root_points_at_project_root():
    from improvement import reward

    assert (reward._ROOT_DIR / "smartfarm").is_dir()
    assert (reward._ROOT_DIR / "fuzzyLozic").is_dir()


def test_llm_gate_falls_back_without_gpu_or_transformers(monkeypatch):
    from improvement import llm_gate

    monkeypatch.setattr(llm_gate, "_llm_runtime_available", lambda: False)
    out = llm_gate.llm_gate(current_packet={}, recent_packets=[], action={"action_type": "rule_weight"})
    assert out["decision"] in {"APPROVE", "HOLD"}


def test_llm_gate_module_does_not_import_torch():
    import importlib

    sys.modules.pop("improvement.llm_gate", None)
    had_torch = "torch" in sys.modules
    importlib.import_module("improvement.llm_gate")
    if not had_torch:
        assert "torch" not in sys.modules


def test_membership_delta_cap_is_cumulative():
    from improvement.apply_action import apply_action_to_state
    from improvement.types import Action

    cfg = load_default_config()
    base = load_default_config()
    state = {"overrides": {"memberships": {}, "rule_weights": {}}}
    act = Action(
        action_id="M_WT_OPT_P1_UP",
        action_type="membership_points",
        var_key="water_temperature",
        shape_label="안정(최적)",
        point_index=1,
        delta=0.2,
    )
    results = [apply_action_to_state(cfg=cfg, state=state, action=act, base_cfg=base) for _ in range(10)]
    pts = state["overrides"]["memberships"]["water_temperature"]["안정(최적)"]
    cap = cfg["improvement"]["membership_delta_cap"]["water_temperature"]
    assert pts[1] <= 12.0 + cap + 1e-9
    assert results[-1]["ok"] is False


def test_reward_uses_only_packets_after_action(tmp_path):
    from improvement.run_cycle import run_improvement_cycle

    now = datetime.now().astimezone()
    applied = now - timedelta(minutes=10)

    def pkt(ts, action_key):
        return {"inputs": {"timestamp": ts.isoformat(timespec="seconds")}, "action_key": action_key, "risk": {"risk_score": 30.0}}

    # 적용 전에는 경보만, 적용 후에는 정상(+경보 10%)만
    rows = [pkt(applied - timedelta(minutes=i + 1), "위험") for i in range(20)]
    rows += [pkt(applied + timedelta(minutes=1), "위험")]
    rows += [pkt(applied + timedelta(minutes=i + 2), "안정") for i in range(9)]
    hp = tmp_path / "packets.jsonl"
    hp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    sp = tmp_path / "state.json"
    state = {
        "overrides": {"memberships": {}, "rule_weights": {}},
        "bandit": {"algo": "ucb1", "c": 0.35, "actions": {}},
        "pending": {"action_id": "W_R6_UP", "applied_at": applied.isoformat(timespec="seconds")},
    }
    sp.write_text(json.dumps(state), encoding="utf-8")

    out = run_improvement_cycle(
        state_path=str(sp),
        history_packets_path=str(hp),
        use_llm_gate=False,
        vision_capture_dir=str(tmp_path / "no_vision"),
    )
    assert out["reward_updated"]
    # 적용 후 10개 중 경보 1개 → alarm_rate 0.1 (목표 범위 안)
    assert out["reward_details"]["alarm_rate"] == pytest.approx(0.1)


def test_strong_warning_rule_not_diluted_by_boundary_rules():
    """수온 경계 + 유량 경계(R10 경고 1.0)가 R22/R25 경계 규칙과 평균돼 '주의'로 내려가면 안 된다."""
    from fuzzify import fuzzify_inputs, load_default_config
    from inference import infer

    cfg = load_default_config()
    inputs = {"water_temperature": 16.0, "pH": 6.5, "flow_ratio": 0.85, "EC": 0.12, "turbidity": 0.5}
    _, label, *_ = infer(fuzzify_inputs(inputs, cfg, ec_mode="water"), cfg)
    assert label in {"경고", "위험"}
