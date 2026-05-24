from __future__ import annotations

"""
Run:
  cd fuzzyLozic
  python run_control_optimizer.py --input inputs.json --ec-mode water
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control import (
    build_monitor_only_output,
    get_default_optimizer_config,
    optimize_control,
    save_control_output,
    should_optimize,
)
from main import run_assessment


REQUIRED_INPUT_KEYS = ["T", "DO", "pH", "FR", "EC", "Turbidity"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fuzzy assessment and control optimization without polluting fuzzy history.")
    parser.add_argument("--input", default=str(DATA_DIR / "inputs.json"), help="Path to the current sensor input JSON.")
    parser.add_argument("--ec-mode", default="water", choices=["water", "hydro"])
    parser.add_argument("--output", default=str(DATA_DIR / "outputs.json"), help="Path to the refreshed fuzzy output packet.")
    parser.add_argument("--control-output", default=str(DATA_DIR / "device_control_outputs.json"), help="Path to the latest control result JSON.")
    parser.add_argument("--control-history", default=str(DATA_DIR / "device_control_history.jsonl"), help="Path to the append-only control history JSONL.")
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (BASE_DIR / path).resolve()


def load_inputs_file(path: Path) -> Dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in input file: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError("Input JSON must be an object.")

    parsed: Dict[str, float] = {}
    for key in REQUIRED_INPUT_KEYS:
        value = data.get(key)
        if not isinstance(value, (int, float)):
            raise ValueError(f"Missing or non-numeric sensor value: {key}")
        parsed[key] = float(value)
    return parsed


def main() -> int:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    control_output_path = resolve_path(args.control_output)
    control_history_path = resolve_path(args.control_history)
    history_inputs_path = DATA_DIR / "history_inputs.jsonl"
    history_packets_path = DATA_DIR / "history_packets.jsonl"

    try:
        current_inputs = load_inputs_file(input_path)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    current_packet = run_assessment(
        current_inputs,
        ec_mode=args.ec_mode,
        history_inputs_path=str(history_inputs_path),
        history_packets_path=str(history_packets_path),
        save_history=False,
    )

    output_path.write_text(json.dumps(current_packet, ensure_ascii=False, indent=2), encoding="utf-8")

    config = get_default_optimizer_config()
    config.update(
        {
            "history_inputs_path": str(history_inputs_path),
            "history_packets_path": str(history_packets_path),
            "control_history_path": str(control_history_path),
            "device_control_output_path": str(control_output_path),
            "device_control_history_path": str(control_history_path),
        }
    )

    if should_optimize(current_packet):
        result = optimize_control(current_inputs, current_packet, ec_mode=args.ec_mode, config=config)
    else:
        result = build_monitor_only_output(current_inputs, current_packet, config)

    save_control_output(
        result,
        output_path=control_output_path,
        history_path=control_history_path,
        max_records=int(config["history_max_records"]),
    )

    predicted = result.get("prediction") if isinstance(result.get("prediction"), dict) else None
    predicted_risk = predicted.get("predicted_risk") if isinstance(predicted, dict) else None
    predicted_score = predicted_risk.get("risk_score") if isinstance(predicted_risk, dict) else None
    predicted_label = predicted_risk.get("risk_label") if isinstance(predicted_risk, dict) else None

    print(f"current action_key: {current_packet.get('action_key')}")
    print(f"control_mode: {result.get('control_mode')}")
    print("selected_control: " + json.dumps(result.get("selected_control", {}), ensure_ascii=False))
    print(f"estimated_total_wh: {result.get('estimated_total_wh')}")
    print(f"predicted risk_score: {predicted_score}")
    print(f"predicted risk_label: {predicted_label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
