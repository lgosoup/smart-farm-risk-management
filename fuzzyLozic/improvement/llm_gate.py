from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

# 프로젝트 루트(llm/ 패키지가 있는 폴더). fuzzyLozic/ 에서 실행해도 llm을 찾을 수 있게 한다.
_ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def _llm_runtime_available() -> bool:
    """transformers + CUDA GPU가 있을 때만 True. torch는 이 함수 안에서만 import한다."""
    try:
        if importlib.util.find_spec("transformers") is None or importlib.util.find_spec("torch") is None:
            return False
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_first_json_object(text: str) -> Dict[str, Any]:
    t = _strip_think_blocks(text).strip()
    t = re.sub(r"^\s*```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```\s*$", "", t)
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    raw = t[start : end + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raw2 = re.sub(r",\s*([}\]])", r"\1", raw)
        return json.loads(raw2)


def heuristic_gate(
    *,
    current_packet: Dict[str, Any],
    recent_packets: List[Dict[str, Any]],
    action: Dict[str, Any],
) -> Dict[str, Any]:
    """Fallback gate when LLM is unavailable.

    Conservative rules:
    - If integrity spike/out_of_range exists in current, HOLD.
    - If action is membership shift and current is already unstable (many TS flags), HOLD.
    - Otherwise, APPROVE.
    """
    integ = current_packet.get("sensor_integrity") if isinstance(current_packet.get("sensor_integrity"), dict) else {}
    spk = integ.get("spike") if isinstance(integ.get("spike"), dict) else {}
    oor = integ.get("out_of_range") if isinstance(integ.get("out_of_range"), dict) else {}

    ts = current_packet.get("timeseries") if isinstance(current_packet.get("timeseries"), dict) else {}
    flags = ts.get("flags") if isinstance(ts.get("flags"), list) else []

    bias_tags: List[str] = []
    notes: List[str] = []

    if spk or oor:
        bias_tags.append("NOISE_SUSPECT")
        notes.append("현재 패킷에 spike/out_of_range가 있어 개선 적용을 보류")
        return {"decision": "HOLD", "reason": "노이즈 의심", "bias_tags": bias_tags, "notes": notes}

    # if many warnings/dangers recently, prefer stability
    recent_labels: List[str] = []
    for p in recent_packets[-10:]:
        ak = p.get("action_key")
        if isinstance(ak, str):
            recent_labels.append(ak)
    danger_like = sum(1 for x in recent_labels if x in {"경고", "위험", "긴급"})
    if danger_like >= 7:
        bias_tags.append("STABILITY_NEED")
        notes.append("최근 경고/위험 비중이 높음")

    if isinstance(action, dict) and action.get("action_type") == "membership_points" and len(flags) >= 3:
        bias_tags.append("STABILITY_NEED")
        notes.append("시계열 플래그가 많아 멤버십 변경은 보수적으로")
        return {"decision": "HOLD", "reason": "상태 불안정", "bias_tags": bias_tags, "notes": notes}

    return {"decision": "APPROVE", "reason": "안전 범위 내 미세 조정", "bias_tags": bias_tags, "notes": notes}


def llm_gate(
    *,
    current_packet: Dict[str, Any],
    recent_packets: List[Dict[str, Any]],
    action: Dict[str, Any],
) -> Dict[str, Any]:
    """LLM gate. If model load fails, fall back to heuristic_gate."""
    # GPU/transformers가 없으면 모델을 받거나 올리려 하지 말고 바로 휴리스틱으로 간다.
    # (llm.loader는 GPU 전용(device_map={"": 0})이라, 없는 환경에서는 수 GB 모델을 내려받은 뒤에야 실패한다)
    if not _llm_runtime_available():
        return heuristic_gate(current_packet=current_packet, recent_packets=recent_packets, action=action)
    try:
        root = str(_ROOT_DIR)
        if root not in sys.path:
            sys.path.insert(0, root)
        from llm.loader import get_llm
        from llm.improvement_prompts import SYSTEM_PROMPT_IMPROVEMENT_GATE

        tokenizer, model = get_llm()

        payload = {
            "current_packet": current_packet,
            "recent_packets": recent_packets[-8:],
            "action": action,
        }
        payload_str = json.dumps(payload, ensure_ascii=False, indent=2)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_IMPROVEMENT_GATE},
            {
                "role": "user",
                "content": "/no_think\n아래 JSON만 근거로 스키마대로 JSON object 1개만 출력.\n\n" + payload_str,
            },
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")

        out = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
        )

        raw = tokenizer.decode(
            out[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        ).strip()

        obj = _extract_first_json_object(raw)
        if not isinstance(obj, dict):
            raise ValueError("gate output not a dict")

        # Hard clamp: allow only APPROVE/HOLD
        dec = obj.get("decision")
        if dec not in {"APPROVE", "HOLD"}:
            obj["decision"] = "HOLD"
            obj["reason"] = "형식 불일치로 보류"
        return obj
    except Exception:
        return heuristic_gate(current_packet=current_packet, recent_packets=recent_packets, action=action)
