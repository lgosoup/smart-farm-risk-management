"""
Smart Farm Risk Management System - Orchestrator

스케줄:
  - 퍼지 평가:   매 EVAL_INTERVAL_MINUTES (기본 60분)
  - 밴딧 개선:   매일 BANDIT_HOUR 이후 첫 사이클 (기본 03:00)

센서 어댑터 (SensorAdapter):
  - FileSensorAdapter (기본):
      inputs.json을 외부 센서 모듈이 이미 갱신한다고 가정하고 읽기만 함.
  - SubprocessSensorAdapter:
      외부 센서 수집 스크립트를 호출 → 스크립트가 inputs.json 작성 → 읽기.
      센서 모듈이 준비되면 __main__ 블록에서 이걸로 교체.

웹 인터페이스 브릿지 (미래 확장용 파일 인터페이스):
  llm_request.json   웹이 {"query": "..."} 을 써두면 오케스트레이터가 소비
  llm_response.json  LLM 결과를 여기 저장 → 웹이 읽음
  system_status.json 최신 평가 스냅샷 → 웹 대시보드용
  alerts.jsonl       경고/위험/긴급 누적 로그
"""
from __future__ import annotations

import json
import logging
import sys
import time
from abc import ABC, abstractmethod
from datetime import date, datetime
from pathlib import Path
from typing import Optional

# ── sys.path 설정 ────────────────────────────────────────────────────────────
ROOT_DIR  = Path(__file__).resolve().parent
FUZZY_DIR = ROOT_DIR / "fuzzyLozic"

for _p in [str(FUZZY_DIR), str(ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── fuzzyLozic 내부 임포트 (sys.path 설정 후) ────────────────────────────────
from main import run_assessment                          # noqa: E402
from control import (                                   # noqa: E402
    build_monitor_only_output,
    get_default_optimizer_config,
    optimize_control,
    save_control_output,
    should_optimize,
)
from improvement.run_cycle import run_improvement_cycle  # noqa: E402

# ── 파일 경로 ─────────────────────────────────────────────────────────────────
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

INPUTS_PATH            = DATA_DIR / "inputs.json"
HISTORY_INPUTS_PATH    = DATA_DIR / "history_inputs.jsonl"
HISTORY_PACKETS_PATH   = DATA_DIR / "history_packets.jsonl"
CONTROL_OUTPUT_PATH    = DATA_DIR / "device_control_outputs.json"
CONTROL_HISTORY_PATH   = DATA_DIR / "device_control_history.jsonl"
IMPROVEMENT_STATE_PATH = DATA_DIR / "improvement_state.json"

# 웹 브릿지 파일 (현재는 오케스트레이터가 관리, 이후 웹 서버가 읽고 씀)
ALERTS_PATH               = DATA_DIR / "alerts.jsonl"
SYSTEM_STATUS_PATH        = DATA_DIR / "system_status.json"
LLM_REQUEST_PATH          = DATA_DIR / "llm_request.json"
LLM_RESPONSE_PATH         = DATA_DIR / "llm_response.json"
VISION_CAPTURE_DIR        = DATA_DIR / "vision_captures"
VISION_CAPTURE_REQUEST_PATH = DATA_DIR / "vision_capture_request.json"

# ── 스케줄 설정 ───────────────────────────────────────────────────────────────
EVAL_INTERVAL_MINUTES = 60
BANDIT_HOUR           = 3        # 03:00에 밴딧 실행
ALERT_LEVELS          = {"경고", "위험", "긴급"}

# ── 로깅 ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT_DIR / "orchestrator.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("orchestrator")


# ════════════════════════════════════════════════════════════════════════════════
# 센서 어댑터
# ════════════════════════════════════════════════════════════════════════════════

class SensorAdapter(ABC):
    """센서 데이터 수집 인터페이스.

    센서 모듈이 준비되면 이 클래스를 상속해서 collect()를 구현.
    orchestrator는 어댑터만 교체하면 되고, 나머지 로직은 그대로.
    """

    @abstractmethod
    def collect(self) -> dict[str, float]:
        """현재 센서값을 반환.
        {water_temperature, pH, EC, flow_ratio, turbidity, air_temperature, humidity} 포함 필수."""


class FileSensorAdapter(SensorAdapter):
    """기본 어댑터: inputs.json을 외부에서 이미 갱신했다고 가정하고 읽기만 함.

    외부 센서 모듈이 주기적으로 inputs.json을 덮어쓰는 구조라면 이것만으로 충분.
    """

    def __init__(self, path: Path = INPUTS_PATH) -> None:
        self.path = path

    def collect(self) -> dict[str, float]:
        if not self.path.exists():
            raise FileNotFoundError(f"inputs.json 없음: {self.path}")
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("inputs.json은 JSON object여야 함")
        return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}


class SubprocessSensorAdapter(SensorAdapter):
    """외부 센서 수집 스크립트를 subprocess로 실행하는 어댑터.

    흐름:
      오케스트레이터 → script_path 실행 → 스크립트가 inputs.json 작성 → 읽기

    사용 예::

        sensor = SubprocessSensorAdapter(
            script_path="sensors/collector.py",
            args=["--port", "COM3"],
        )

    센서 스크립트 규약:
      - 실행 후 inputs.json에 {"water_temperature":..., "pH":..., "EC":..., "flow_ratio":..., "turbidity":..., "air_temperature":..., "humidity":...} 작성
      - 성공 시 returncode=0, 실패 시 returncode!=0
    """

    def __init__(
        self,
        script_path: str | Path,
        args: Optional[list[str]] = None,
        timeout: int = 30,
        inputs_path: Path = INPUTS_PATH,
    ) -> None:
        self.script_path = Path(script_path)
        self.args = args or []
        self.timeout = timeout
        self._file_adapter = FileSensorAdapter(inputs_path)

    def collect(self) -> dict[str, float]:
        import subprocess

        cmd = [sys.executable, str(self.script_path)] + self.args
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        if result.returncode != 0:
            raise RuntimeError(
                f"센서 스크립트 실패 (rc={result.returncode}): {result.stderr[:300]}"
            )
        return self._file_adapter.collect()


# ════════════════════════════════════════════════════════════════════════════════
# 알람 / 상태 관리
# ════════════════════════════════════════════════════════════════════════════════

def _append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def save_alert(packet: dict, action_key: str) -> None:
    """경고 이상 발생 시 alerts.jsonl에 기록.

    이후 확장 포인트:
      - 이메일: smtplib.sendmail(...)
      - 슬랙: requests.post(SLACK_WEBHOOK, ...)
      - 카카오: requests.post(KAKAO_API, ...)
    """
    risk = packet.get("risk") or {}
    alert = {
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "level": action_key,
        "risk_score": risk.get("risk_score"),
        "risk_label": risk.get("risk_label"),
        "inputs": packet.get("inputs"),
        "top_rules": (packet.get("rules_fired_topN") or [])[:3],
        "timeseries_flags": (packet.get("timeseries") or {}).get("flags", []),
        "recommendations": packet.get("recommendations") or {},
    }
    _append_jsonl(ALERTS_PATH, alert)
    log.warning(
        "ALERT [%s] risk_score=%.1f",
        action_key,
        risk.get("risk_score", 0.0),
    )


def update_system_status(packet: dict, cycle_ts: str) -> None:
    """system_status.json 갱신. 미래 웹 대시보드가 이 파일을 폴링."""
    risk = packet.get("risk") or {}
    SYSTEM_STATUS_PATH.write_text(
        json.dumps(
            {
                "last_eval": cycle_ts,
                "action_key": packet.get("action_key"),
                "risk_score": risk.get("risk_score"),
                "risk_label": risk.get("risk_label"),
                "inputs": packet.get("inputs"),
                "timeseries_flags": (packet.get("timeseries") or {}).get("flags", []),
                "sensor_integrity": packet.get("sensor_integrity"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ════════════════════════════════════════════════════════════════════════════════
# LLM 요청 처리 (사용자 트리거)
# ════════════════════════════════════════════════════════════════════════════════

def check_and_handle_llm_request(packet: dict) -> None:
    """llm_request.json이 있으면 LLM 실행 후 llm_response.json에 결과 저장.

    웹 인터페이스 연동 방법 (미래):
      요청: POST /llm-query → llm_request.json 작성
      응답: GET  /llm-result → llm_response.json 읽기

    지금 당장 테스트하려면:
      echo {"query": "현재 상태 설명해줘"} > llm_request.json
    """
    if not LLM_REQUEST_PATH.exists():
        return

    try:
        req = json.loads(LLM_REQUEST_PATH.read_text(encoding="utf-8"))
        query = str(req.get("query", "현재 상태를 근거와 함께 자세히 설명"))
    except Exception as e:
        log.error("llm_request.json 읽기 실패: %s", e)
        LLM_REQUEST_PATH.unlink(missing_ok=True)
        return

    LLM_REQUEST_PATH.unlink(missing_ok=True)  # 요청 소비
    log.info("LLM 요청 수신: %s", query[:60])

    try:
        from llm.generate import run_smartfarm_llm_timeseries

        history = _read_jsonl_tail(HISTORY_PACKETS_PATH, 60)
        result = run_smartfarm_llm_timeseries(
            current_packet=packet,
            history_packets=history[:-1],
            user_query=query,
        )
        LLM_RESPONSE_PATH.write_text(result, encoding="utf-8")
        log.info("LLM 응답 저장 완료: %s", LLM_RESPONSE_PATH.name)
    except Exception as e:
        LLM_RESPONSE_PATH.write_text(
            json.dumps({"error": str(e)}, ensure_ascii=False), encoding="utf-8"
        )
        log.error("LLM 추론 실패: %s", e)


# ════════════════════════════════════════════════════════════════════════════════
# 제어 최적화
# ════════════════════════════════════════════════════════════════════════════════

def run_control_step(inputs: dict[str, float], packet: dict, ec_mode: str) -> None:
    config = get_default_optimizer_config()
    config.update(
        {
            "history_inputs_path": str(HISTORY_INPUTS_PATH),
            "history_packets_path": str(HISTORY_PACKETS_PATH),
            "control_history_path": str(CONTROL_HISTORY_PATH),
            "device_control_output_path": str(CONTROL_OUTPUT_PATH),
            "device_control_history_path": str(CONTROL_HISTORY_PATH),
        }
    )

    if should_optimize(packet):
        result = optimize_control(inputs, packet, ec_mode=ec_mode, config=config)
        pred = (result.get("prediction") or {}).get("predicted_risk") or {}
        log.info(
            "제어 최적화 완료 → predicted_risk=%.1f J=%.4f",
            pred.get("risk_score", 0.0),
            (result.get("objective") or {}).get("J", 0.0),
        )
    else:
        result = build_monitor_only_output(inputs, packet, config)

    save_control_output(
        result,
        output_path=CONTROL_OUTPUT_PATH,
        history_path=CONTROL_HISTORY_PATH,
        max_records=int(config["history_max_records"]),
    )


# ════════════════════════════════════════════════════════════════════════════════
# 헬퍼
# ════════════════════════════════════════════════════════════════════════════════

def _read_jsonl_tail(path: Path, n: int) -> list[dict]:
    if not path.exists() or n <= 0:
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except json.JSONDecodeError:
                continue
    return rows[-n:]


def _should_run_bandit(last_bandit_date: Optional[date], now: datetime, bandit_hour: int) -> bool:
    return last_bandit_date != now.date() and now.hour >= bandit_hour


def _request_vision_capture(
    capture_dir: Path,
    timeout_sec: int = 60,
    poll_interval_sec: float = 5.0,
) -> Optional[Path]:
    """
    비전 캡처 요청 파일을 작성하고 센서가 이미지를 저장할 때까지 폴링.

    프로토콜:
      1. vision_capture_request.json 작성 → 센서 모듈이 읽어서 촬영
      2. 센서가 YYYYMMDD.jpg를 capture_dir에 저장 후 request.json 삭제
      3. timeout 내에 파일 확인되면 Path 반환, 초과 시 None
    """
    capture_dir.mkdir(parents=True, exist_ok=True)
    today_str = date.today().strftime("%Y%m%d")
    save_path = capture_dir / f"{today_str}.jpg"

    request = {
        "date": today_str,
        "save_path": str(save_path),
    }
    VISION_CAPTURE_REQUEST_PATH.write_text(
        json.dumps(request, ensure_ascii=False), encoding="utf-8"
    )
    log.info("비전 캡처 요청 작성: %s", today_str)

    elapsed = 0.0
    while elapsed < timeout_sec:
        if save_path.exists():
            log.info("비전 이미지 수신 확인: %s", save_path.name)
            return save_path
        time.sleep(poll_interval_sec)
        elapsed += poll_interval_sec

    log.warning("비전 캡처 timeout (%ds) — 휴리스틱으로 fallback", timeout_sec)
    VISION_CAPTURE_REQUEST_PATH.unlink(missing_ok=True)
    return None


# ════════════════════════════════════════════════════════════════════════════════
# 메인 루프
# ════════════════════════════════════════════════════════════════════════════════

def run(
    sensor_adapter: Optional[SensorAdapter] = None,
    eval_interval_minutes: int = EVAL_INTERVAL_MINUTES,
    bandit_hour: int = BANDIT_HOUR,
    ec_mode: str = "water",
) -> None:
    """오케스트레이터 메인 루프. Ctrl+C로 종료.

    Parameters
    ----------
    sensor_adapter:
        None이면 FileSensorAdapter 사용 (inputs.json 읽기만).
    eval_interval_minutes:
        퍼지 평가 주기 (분).
    bandit_hour:
        밴딧 개선을 실행할 시각 (0~23). 해당 시간 이후 첫 사이클에 실행.
    ec_mode:
        EC 모드. "water" 이면 crop_config.json의 설정을 자동으로 사용.
    """
    adapter = sensor_adapter or FileSensorAdapter()
    eval_interval_sec = eval_interval_minutes * 60
    last_bandit_date: Optional[date] = None

    log.info(
        "오케스트레이터 시작. eval_interval=%dmin, bandit_hour=%02d:00",
        eval_interval_minutes,
        bandit_hour,
    )

    while True:
        cycle_start = datetime.now()
        cycle_ts = cycle_start.astimezone().isoformat(timespec="seconds")

        # ── 1. 센서 수집 ──────────────────────────────────────────────────────
        try:
            inputs = adapter.collect()
            log.info(
                "센서 수집 완료: wt=%.1f pH=%.2f EC=%.3f fr=%.2f turb=%.1f at=%.1f hum=%.1f",
                inputs.get("water_temperature", 0), inputs.get("pH", 0),
                inputs.get("EC", 0), inputs.get("flow_ratio", 0),
                inputs.get("turbidity", 0), inputs.get("air_temperature", 0),
                inputs.get("humidity", 0),
            )
        except Exception as e:
            log.error("센서 수집 실패: %s — 이번 사이클 건너뜀", e)
            time.sleep(eval_interval_sec)
            continue

        # ── 2. 퍼지 평가 ──────────────────────────────────────────────────────
        try:
            packet = run_assessment(
                inputs,
                ec_mode=ec_mode,
                history_inputs_path=str(HISTORY_INPUTS_PATH),
                history_packets_path=str(HISTORY_PACKETS_PATH),
                save_history=True,
            )
            action_key = str(packet.get("action_key", ""))
            risk_score = (packet.get("risk") or {}).get("risk_score", 0.0)
            log.info("퍼지 평가: action_key=%s risk_score=%.1f", action_key, risk_score)
        except Exception as e:
            log.error("퍼지 평가 실패: %s", e)
            time.sleep(eval_interval_sec)
            continue

        # ── 3. 시스템 상태 갱신 (웹 브릿지) ──────────────────────────────────
        try:
            update_system_status(packet, cycle_ts)
        except Exception as e:
            log.error("system_status.json 갱신 실패: %s", e)

        # ── 4. 경보 저장 ──────────────────────────────────────────────────────
        if action_key in ALERT_LEVELS:
            try:
                save_alert(packet, action_key)
            except Exception as e:
                log.error("알람 저장 실패: %s", e)

        # ── 5. 제어 최적화 ────────────────────────────────────────────────────
        try:
            run_control_step(inputs, packet, ec_mode)
        except Exception as e:
            log.error("제어 최적화 실패: %s", e)

        # ── 6. LLM 요청 확인 (사용자 트리거) ─────────────────────────────────
        try:
            check_and_handle_llm_request(packet)
        except Exception as e:
            log.error("LLM 요청 처리 실패: %s", e)

        # ── 7. 일별 밴딧 개선 ────────────────────────────────────────────────
        if _should_run_bandit(last_bandit_date, cycle_start, bandit_hour):
            try:
                _request_vision_capture(VISION_CAPTURE_DIR)
                result = run_improvement_cycle(
                    state_path=str(IMPROVEMENT_STATE_PATH),
                    history_packets_path=str(HISTORY_PACKETS_PATH),
                    reward_window=48,   # 1시간 주기 기준 48시간치
                    vision_capture_dir=str(VISION_CAPTURE_DIR),
                )
                last_bandit_date = cycle_start.date()
                reward_str = (
                    f"{result['reward']:.4f}" if result.get("reward") is not None else "N/A"
                )
                log.info(
                    "밴딧 사이클: action=%s decision=%s reward=%s",
                    (result.get("selected_action") or {}).get("action_id", "-"),
                    result.get("decision", "-"),
                    reward_str,
                )
            except Exception as e:
                log.error("밴딧 개선 실패: %s", e)

        # ── 8. 다음 사이클까지 대기 ───────────────────────────────────────────
        elapsed = (datetime.now() - cycle_start).total_seconds()
        sleep_sec = max(0.0, eval_interval_sec - elapsed)
        log.info(
            "사이클 완료 (%.1fs 소요). 다음 실행까지 %.0fs (%.1fmin)",
            elapsed, sleep_sec, sleep_sec / 60,
        )
        time.sleep(sleep_sec)


# ════════════════════════════════════════════════════════════════════════════════
# 진입점
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── 센서 어댑터 설정 ──────────────────────────────────────────────────────
    #
    # [현재] inputs.json을 외부에서 갱신한다고 가정, 읽기만:
    sensor = FileSensorAdapter()
    #
    # [센서 모듈 준비 후] 아래로 교체:
    # sensor = SubprocessSensorAdapter(
    #     script_path="sensors/collector.py",  # 센서 수집 스크립트 경로
    #     args=["--port", "COM3"],              # 필요한 인자
    #     timeout=30,                           # 최대 대기 시간 (초)
    # )

    run(
        sensor_adapter=sensor,
        eval_interval_minutes=EVAL_INTERVAL_MINUTES,  # 60분
        bandit_hour=BANDIT_HOUR,                      # 03:00
        ec_mode="water",  # crop_config.json 있으면 자동 반영됨
    )
