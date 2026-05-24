import json
from typing import Dict

def load_inputs(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("inputs.json은 JSON object여야 합니다.")

    for k, v in data.items():
        if not isinstance(v, (int, float)):
            raise ValueError(f"입력값 오류: {k}={v} (숫자가 아님)")

    return data
