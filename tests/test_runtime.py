"""Runtime regression tests for the entry points (orchestrator / app / streamlit_app / llm).

None of these tests write into the repository's data/ directory:
- orchestrator paths are monkeypatched to tmp_path,
- app.py / streamlit_app.py are exercised on a temporary copy of the repo.
The LLM is always forced "unavailable" so no model is ever loaded.
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_COPY_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "tests", "*.jsonl")


def _copy_repo(dst: Path) -> Path:
    """Copy just what the entry points need into dst (no history files)."""
    for name in ("fuzzyLozic", "llm", "data"):
        shutil.copytree(ROOT / name, dst / name, ignore=_COPY_IGNORE)
    for name in ("app.py", "streamlit_app.py", "orchestrator.py"):
        shutil.copy2(ROOT / name, dst / name)
    # runtime files that may exist locally must not leak into the copy
    for name in ("system_status.json", "llm_request.json", "llm_response.json", "vision_capture_request.json"):
        (dst / "data" / name).unlink(missing_ok=True)
    return dst


def _force_llm_unavailable(monkeypatch):
    import llm.loader as loader

    def _unavailable():
        raise loader.LLMUnavailableError("test: LLM disabled")

    monkeypatch.setattr(loader, "check_llm_available", _unavailable)
    return loader


# ════════════════════════════════════════════════════════════════════════════
# llm/
# ════════════════════════════════════════════════════════════════════════════

def test_llm_modules_import_without_gpu_stack():
    # must not import torch/transformers at module import time
    import llm.generate  # noqa: F401
    import llm.loader
    import llm.prompts  # noqa: F401
    import llm.improvement_prompts  # noqa: F401

    assert hasattr(llm.loader, "LLMUnavailableError")


def test_check_llm_available_reports_missing_packages(monkeypatch):
    import importlib.util as iu

    import llm.loader as loader

    real_find_spec = iu.find_spec
    monkeypatch.setattr(
        loader.importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name in {"transformers", "bitsandbytes"} else real_find_spec(name, *a, **k),
    )
    with pytest.raises(loader.LLMUnavailableError) as ei:
        loader.check_llm_available()
    msg = str(ei.value)
    assert "transformers" in msg and "requirements-gpu.txt" in msg

    # get_llm surfaces the same clear error (and leaves the cache empty)
    with pytest.raises(loader.LLMUnavailableError):
        loader.get_llm()
    assert loader._model is None and loader._tokenizer is None


def test_generate_without_llm_raises_clear_error(monkeypatch):
    _force_llm_unavailable(monkeypatch)
    from llm.generate import run_smartfarm_llm_timeseries
    from llm.loader import LLMUnavailableError

    with pytest.raises(LLMUnavailableError):
        run_smartfarm_llm_timeseries(current_packet={"action_key": "안정"}, history_packets=[])


@pytest.mark.parametrize(
    "src, expected",
    [
        ("EC가 낮음", "전기전도도(EC)가 낮음"),
        ("EC 높음", "전기전도도(EC) 높음"),
        ("전기전도도(EC) 정상", "전기전도도(EC) 정상"),
        ("EC(전기전도도) 정상", "EC(전기전도도) 정상"),
        ("ECO 모드", "ECO 모드"),
    ],
)
def test_postprocess_korean_terms(src, expected):
    from llm.generate import _postprocess_korean_terms

    assert _postprocess_korean_terms(src) == expected
    # keys are never touched
    assert _postprocess_korean_terms({"EC": src}) == {"EC": expected}


def test_ensure_schema_json_handles_think_and_trailing_comma():
    from llm.generate import _ensure_schema_json

    raw = '<think>blah</think>\n```json\n{"상태": "주의", "주요 원인": ["a",],}\n```'
    assert json.loads(_ensure_schema_json(raw)) == {"상태": "주의", "주요 원인": ["a"]}


# ════════════════════════════════════════════════════════════════════════════
# orchestrator.py
# ════════════════════════════════════════════════════════════════════════════

class _StopLoop(Exception):
    pass


class _DictSensor:
    def __init__(self, values):
        self.values = values

    def collect(self):
        return dict(self.values)


@pytest.fixture
def orch(tmp_path, monkeypatch):
    """orchestrator module with every data path redirected into tmp_path."""
    import orchestrator as o

    data = tmp_path / "data"
    data.mkdir()
    # improvement_state.json is a runtime file (not in git); start without it like a fresh clone
    for attr, name in {
        "INPUTS_PATH": "inputs.json",
        "HISTORY_INPUTS_PATH": "history_inputs.jsonl",
        "HISTORY_PACKETS_PATH": "history_packets.jsonl",
        "CONTROL_OUTPUT_PATH": "device_control_outputs.json",
        "CONTROL_HISTORY_PATH": "device_control_history.jsonl",
        "IMPROVEMENT_STATE_PATH": "improvement_state.json",
        "ALERTS_PATH": "alerts.jsonl",
        "SYSTEM_STATUS_PATH": "system_status.json",
        "LLM_REQUEST_PATH": "llm_request.json",
        "LLM_RESPONSE_PATH": "llm_response.json",
        "VISION_CAPTURE_DIR": "vision_captures",
        "VISION_CAPTURE_REQUEST_PATH": "vision_capture_request.json",
    }.items():
        monkeypatch.setattr(o, attr, data / name)
    monkeypatch.setattr(o, "_setup_logging", lambda *a, **k: None)
    _force_llm_unavailable(monkeypatch)

    def fake_sleep(sec):
        if sec >= 60:  # end-of-cycle / error sleep -> stop after one cycle
            raise _StopLoop
        # vision capture polling: return immediately

    monkeypatch.setattr(o.time, "sleep", fake_sleep)
    o._test_data_dir = data
    return o


def _run_one_cycle(o, inputs, **kw):
    with pytest.raises(_StopLoop):
        o.run(sensor_adapter=_DictSensor(inputs), **kw)


def test_orchestrator_import_has_no_log_side_effect():
    import logging

    import orchestrator as o

    log_files = [
        getattr(h, "baseFilename", "") for h in logging.getLogger().handlers + logging.getLogger("orchestrator").handlers
    ]
    assert not any(f.endswith("orchestrator.log") for f in log_files), "import must not attach orchestrator.log handler"
    assert o.LOG_PATH.name == "orchestrator.log"


def test_orchestrator_one_cycle_danger(orch):
    o = orch
    d = o._test_data_dir
    (d / "llm_request.json").write_text(json.dumps({"query": "설명해줘"}, ensure_ascii=False), encoding="utf-8")
    danger = {
        "water_temperature": 36.0, "pH": 6.2, "EC": 1.5, "flow_ratio": 0.2,
        "turbidity": 2.0, "air_temperature": 30.0, "humidity": 70.0,
    }
    _run_one_cycle(o, danger, bandit_hour=0)

    status = json.loads((d / "system_status.json").read_text(encoding="utf-8"))
    assert status["action_key"] in o.ALERT_LEVELS
    assert status["inputs"]["water_temperature"] == 36.0
    assert (d / "alerts.jsonl").read_text(encoding="utf-8").strip()

    control = json.loads((d / "device_control_outputs.json").read_text(encoding="utf-8"))
    assert control["control_mode"] == "optimize"
    assert control["trigger"]["action_key"] == status["action_key"]

    # history appended exactly once
    assert len((d / "history_packets.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert len((d / "history_inputs.jsonl").read_text(encoding="utf-8").splitlines()) == 1

    # LLM request consumed; unavailable LLM -> clear error response, loop survives
    assert not (d / "llm_request.json").exists()
    resp = json.loads((d / "llm_response.json").read_text(encoding="utf-8"))
    assert "error" in resp and "LLM" in resp["error"]

    # bandit ran (heuristic gate) and recorded last_cycle
    state = json.loads((d / "improvement_state.json").read_text(encoding="utf-8"))
    assert state["last_cycle"] and state["last_cycle"]["ts"]


def test_orchestrator_monitor_only_cycle(orch):
    o = orch
    d = o._test_data_dir
    safe = {
        "water_temperature": 22.5, "pH": 6.15, "EC": 1.5, "flow_ratio": 1.0,
        "turbidity": 1.0, "air_temperature": 24.0, "humidity": 60.0,
    }
    _run_one_cycle(o, safe, bandit_hour=0)
    status = json.loads((d / "system_status.json").read_text(encoding="utf-8"))
    control = json.loads((d / "device_control_outputs.json").read_text(encoding="utf-8"))
    if status["action_key"] not in o.ALERT_LEVELS:
        assert control["control_mode"] == "monitor_only"
        assert not (d / "alerts.jsonl").exists()


def test_orchestrator_bandit_not_rerun_after_restart_same_day(orch, monkeypatch):
    o = orch
    d = o._test_data_dir
    sp = d / "improvement_state.json"
    state = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    state["last_cycle"] = {"ts": datetime.now().astimezone().isoformat(timespec="seconds")}
    (d / "improvement_state.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

    called = []
    monkeypatch.setattr(o, "run_improvement_cycle", lambda **kw: called.append(kw) or {})
    monkeypatch.setattr(o, "_request_vision_capture", lambda *a, **k: None)
    _run_one_cycle(o, {"water_temperature": 22.0, "pH": 6.2, "EC": 1.5, "flow_ratio": 1.0,
                       "turbidity": 1.0, "air_temperature": 24.0, "humidity": 60.0}, bandit_hour=0)
    assert called == [], "bandit must not run twice on the same day after a restart"


def test_load_last_bandit_date(tmp_path):
    import orchestrator as o

    assert o._load_last_bandit_date(tmp_path / "missing.json") is None
    p = tmp_path / "s.json"
    p.write_text("{not json", encoding="utf-8")
    assert o._load_last_bandit_date(p) is None
    p.write_text(json.dumps({"last_cycle": None}), encoding="utf-8")
    assert o._load_last_bandit_date(p) is None
    p.write_text(json.dumps({"last_cycle": {"ts": "2026-01-02T03:04:05+09:00"}}), encoding="utf-8")
    assert o._load_last_bandit_date(p) is not None


def test_orchestrator_sensor_failure_skips_cycle(orch):
    o = orch

    class Broken:
        def collect(self):
            raise FileNotFoundError("no inputs.json")

    with pytest.raises(_StopLoop):
        o.run(sensor_adapter=Broken())
    assert not (o._test_data_dir / "system_status.json").exists()


# ════════════════════════════════════════════════════════════════════════════
# app.py
# ════════════════════════════════════════════════════════════════════════════

def test_app_runs_fuzzy_main_from_other_cwd_with_utf8_env(tmp_path, monkeypatch):
    """subprocess call to fuzzyLozic/main.py (on a repo copy) must decode Korean output.

    The copy lives under a path with Korean characters so main.py prints non-ASCII.
    """
    repo = _copy_repo(tmp_path / "저장소")
    spec = importlib.util.spec_from_file_location("app_copy", repo / "app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)

    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")  # mismatching parent setting used to break decoding
    monkeypatch.chdir(tmp_path)
    app.run_fuzzy_main()
    assert (repo / "data" / "outputs.json").exists()
    assert (repo / "data" / "history_packets.jsonl").exists()


def test_app_main_degrades_without_llm(tmp_path, monkeypatch, capsys):
    import app

    _force_llm_unavailable(monkeypatch)
    hp = tmp_path / "history_packets.jsonl"
    pkt = {
        "action_key": "경고",
        "risk": {"risk_score": 70.0, "risk_label": "경고"},
        "rules_fired_topN": [{"human_readable_reason": "수온 높음"}],
    }
    hp.write_text(json.dumps(pkt, ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setattr(app, "HISTORY_PACKETS_PATH", hp)
    monkeypatch.setattr(app, "run_fuzzy_main", lambda: None)
    monkeypatch.setattr(sys, "argv", ["app.py"])

    with pytest.raises(SystemExit) as ei:
        app.main()
    assert ei.value.code == 2
    out = capsys.readouterr().out
    assert "[상태] 경고" in out and "LLM 해설 생략" in out and "수온 높음" in out


def test_app_prints_prompt_schema_evidence(capsys):
    import app

    app._print_human_readable(
        {"상태": "주의", "요약": "s", "판정 근거": {"위험 점수": 50, "핵심 근거": ["근거A"]}}
    )
    assert "근거A" in capsys.readouterr().out


# ════════════════════════════════════════════════════════════════════════════
# streamlit_app.py
# ════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def st_app():
    pytest.importorskip("streamlit")
    import streamlit_app

    return streamlit_app


def test_parse_ts_flags_with_underscore_vars(st_app):
    p = st_app._parse_ts_flag
    assert p("TS_TREND_water_temperature_UP_FAST") == ("TREND", "water_temperature", ["UP", "FAST"])
    assert p("TS_SPIKE_flow_ratio_DOWN") == ("SPIKE", "flow_ratio", ["DOWN"])
    assert p("TS_PERSIST_air_temperature_HIGH") == ("PERSIST", "air_temperature", ["HIGH"])
    assert p("TS_VOL_pH") == ("VOL", "pH", [])
    assert p("TS_RECOVERY_FAIL_flow_ratio") == ("RECOVERY_FAIL", "flow_ratio", [])
    assert p("TS_SPIKE_new_var_UP") == ("SPIKE", "new_var", ["UP"])
    assert p("DELTA") == (None, "", [])


def test_describe_ts_flag_direction_and_variable(st_app):
    from config import load_config

    cfg = load_config()
    d = st_app._describe_ts_flag("TS_TREND_water_temperature_UP", cfg)
    assert d["variable"] == "수온" and d["meaning"].startswith("추세 상승")
    d = st_app._describe_ts_flag("TS_RECOVERY_FAIL_flow_ratio", cfg)
    assert d["variable"] == "유량비" and d["meaning"] == "회복 실패"
    d = st_app._describe_ts_flag("TS_PERSIST_water_temperature_HIGH", cfg)
    assert d["meaning"] == "지속 높음"


def test_display_label_covers_config_labels(st_app):
    from config import load_config

    cfg = load_config()
    labels = set(cfg["output_risk_sets"].keys()) if isinstance(cfg["output_risk_sets"], dict) else set()
    for v in cfg["variables"].values():
        for vc in (v.values() if isinstance(v, dict) else [v]):
            labels.update(s.label for s in vc.shapes)
    src = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    missing = [lbl for lbl in labels if f'"{lbl}":' not in src]
    assert not missing, f"labels missing from _display_label map: {missing}"
    assert st_app._display_label("매우안정") == "매우 안전"
    assert st_app._label_color("매우안정") == "#2E7D32"


def _app_test(repo: Path):
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(repo / "streamlit_app.py"), default_timeout=120)


def test_streamlit_smoke_and_out_of_range_input(tmp_path, st_app):
    repo = _copy_repo(tmp_path / "repo")
    inputs = json.loads((repo / "data" / "inputs.json").read_text(encoding="utf-8"))
    inputs["water_temperature"] = 80.0  # far outside integrity range (-5..45)
    (repo / "data" / "inputs.json").write_text(json.dumps(inputs), encoding="utf-8")

    at = _app_test(repo).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.metric, "risk metric should render"


def test_streamlit_session_history_uses_previous_sample(tmp_path, st_app):
    repo = _copy_repo(tmp_path / "repo")
    at = _app_test(repo).run()
    at.sidebar.selectbox[1].select("세션(임시)").run()  # appends sample #1 (water_temperature=default)
    wt = at.sidebar.number_input(key="input_water_temperature")
    wt.set_value(float(wt.value) + 10.0).run()  # +10 C jump vs previous session sample
    assert not at.exception, [e.value for e in at.exception]
    flag_frames = [df.value for df in at.dataframe if "flag" in getattr(df.value, "columns", [])]
    flags = [str(x) for frame in flag_frames for x in frame["flag"].tolist()]
    assert "SPIKE_water_temperature" in flags, flags
