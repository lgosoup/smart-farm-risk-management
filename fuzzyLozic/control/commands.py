from __future__ import annotations

from typing import Dict

from control.scoring import calculate_device_energy_breakdown

# 펠티어 보호 시퀀스 파라미터
_PELTIER_STARTUP_DELAY_SEC = 5   # 냉각펌프/팬 선행 가동 후 펠티어 ON까지 대기
_PELTIER_COOLDOWN_SEC = 45       # 펠티어 OFF 후 냉각펌프/팬 유지 시간


def build_device_commands(control: Dict[str, float], config: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    """
    제어 변수 → 실제 하드웨어 명령 변환

    핀 맵:
      D5  : BTS7960 메인 물펌프 (main_water_pump) — circulation_pump_minutes
      D6  : 산소펌프 (oxygen_pump)               — aeration_minutes
      D7  : 냉각펌프 (cooling_pump)               — fan_minutes (펠티어 보호 시퀀스)
      D8  : 냉각팬 2개 (cooling_fan)              — fan_minutes (펠티어 보호 시퀀스)
      D9  : 12V LED바 (led_bar)                   — led_minutes
      D10 : 환풍기팬 (ventilation_fan)            — 향후 공기온습도 기반 제어 예정
      D11 : 펠티어 1번 (peltier_1)               — cooler_minutes
      D12 : 펠티어 2번 (peltier_2)               — cooler_minutes

    펠티어 보호 시퀀스:
      가동 시작: cooling_pump ON → cooling_fan ON → (5초 대기) → peltier_1 ON → peltier_2 ON
      가동 종료: peltier_1 OFF → peltier_2 OFF → (45초 유지) → cooling_pump OFF → cooling_fan OFF
    """
    energy = calculate_device_energy_breakdown(control, config)
    cooler_active = float(control["cooler_minutes"]) > 0.0
    fan_active = float(control["fan_minutes"]) > 0.0

    # 냉각펌프/팬: 펠티어 사용 시 cooldown 시간만큼 추가 가동
    fan_base_sec = int(round(float(control["fan_minutes"]) * 60.0))
    fan_total_sec = fan_base_sec + (_PELTIER_COOLDOWN_SEC if cooler_active else 0)
    cooling_enabled = fan_active or cooler_active

    return {
        "main_water_pump": {
            "pin": "D5",
            "type": "BTS7960_PWM",
            "enabled": float(control["circulation_pump_minutes"]) > 0.0,
            "duration_sec": int(round(float(control["circulation_pump_minutes"]) * 60.0)),
            "estimated_wh": round(energy["main_water_pump"], 4),
            "purpose": "메인 물순환 및 유량 안정화",
        },
        "nutrient_pump": {
            "pin": None,
            "enabled": float(control["nutrient_pump_ml"]) > 0.0,
            "target_ml": round(float(control["nutrient_pump_ml"]), 4),
            "estimated_wh": round(energy["nutrient_pump"], 4),
            "purpose": "EC 보정 (양액 보충)",
        },
        "oxygen_pump": {
            "pin": "D6",
            "enabled": float(control["aeration_minutes"]) > 0.0,
            "duration_sec": int(round(float(control["aeration_minutes"]) * 60.0)),
            "estimated_wh": round(energy["oxygen_pump"], 4),
            "purpose": "수중 산소 공급",
        },
        "cooling_pump": {
            "pin": "D7",
            "enabled": cooling_enabled,
            "duration_sec": fan_total_sec,
            "estimated_wh": round(energy["cooling_pump_fan"] * 0.5, 4),
            "purpose": "펠티어 냉각수 순환 (D7)",
            "peltier_sequence": {
                "role": "pre_post_cooling",
                "startup_lead_sec": _PELTIER_STARTUP_DELAY_SEC,
                "cooldown_after_peltier_sec": _PELTIER_COOLDOWN_SEC,
            },
        },
        "cooling_fan": {
            "pin": "D8",
            "enabled": cooling_enabled,
            "duration_sec": fan_total_sec,
            "estimated_wh": round(energy["cooling_pump_fan"] * 0.5, 4),
            "purpose": "펠티어 방열 팬 2개 (D8)",
            "peltier_sequence": {
                "role": "pre_post_cooling",
                "startup_lead_sec": _PELTIER_STARTUP_DELAY_SEC,
                "cooldown_after_peltier_sec": _PELTIER_COOLDOWN_SEC,
            },
        },
        "led_bar": {
            "pin": "D9",
            "enabled": float(control["led_minutes"]) > 0.0,
            "duration_sec": int(round(float(control["led_minutes"]) * 60.0)),
            "estimated_wh": round(energy["led_bar"], 4),
            "purpose": "12V LED 성장 조명 (D9)",
        },
        "ventilation_fan": {
            "pin": "D10",
            "enabled": False,
            "duration_sec": 0,
            "estimated_wh": 0.0,
            "purpose": "환기 팬 (D10) — 공기 온습도 기반 제어 예정",
        },
        "peltier_1": {
            "pin": "D11",
            "enabled": cooler_active,
            "duration_sec": int(round(float(control["cooler_minutes"]) * 60.0)),
            "estimated_wh": round(energy["peltier"] * 0.5, 4),
            "purpose": "펠티어 냉각 소자 1번 (D11)",
            "peltier_sequence": {
                "requires_before_on": ["cooling_pump", "cooling_fan"],
                "startup_delay_sec": _PELTIER_STARTUP_DELAY_SEC,
                "cooldown_sec_after_off": _PELTIER_COOLDOWN_SEC,
                "sequence_on": "cooling_pump ON → cooling_fan ON → wait 5s → peltier_1 ON",
                "sequence_off": "peltier_1 OFF → wait 45s → cooling_pump OFF → cooling_fan OFF",
            },
        },
        "peltier_2": {
            "pin": "D12",
            "enabled": cooler_active,
            "duration_sec": int(round(float(control["cooler_minutes"]) * 60.0)),
            "estimated_wh": round(energy["peltier"] * 0.5, 4),
            "purpose": "펠티어 냉각 소자 2번 (D12)",
            "peltier_sequence": {
                "requires_before_on": ["cooling_pump", "cooling_fan"],
                "startup_delay_sec": _PELTIER_STARTUP_DELAY_SEC,
                "cooldown_sec_after_off": _PELTIER_COOLDOWN_SEC,
                "sequence_on": "cooling_pump ON → cooling_fan ON → wait 5s → peltier_2 ON",
                "sequence_off": "peltier_2 OFF → wait 45s → cooling_pump OFF → cooling_fan OFF",
            },
        },
    }
