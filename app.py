# app.py
import json
import sys
from pathlib import Path
import subprocess

_ROOT = Path(__file__).resolve().parent
BASE_DIR = _ROOT / "fuzzyLozic"
DATA_DIR = _ROOT / "data"
HISTORY_PACKETS_PATH = DATA_DIR / "history_packets.jsonl"


def _read_jsonl_tail(path: Path, n: int) -> list[dict]:
    if n <= 0 or not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows[-n:]


def run_fuzzy_main():
    script_path = BASE_DIR / "main.py"
    if not script_path.exists():
        raise FileNotFoundError(f"fuzzyLozic/main.py를 찾을 수 없습니다: {script_path}")

    result = subprocess.run(
        [sys.executable, "main.py"],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("❌ fuzzyLozic/main.py 실행 실패")
        print(result.stderr)
        raise RuntimeError("퍼지 로직 실행 중 오류 발생")

    if result.stdout.strip():
        print(result.stdout)


def _print_human_readable(obj: dict) -> None:
    # 사람이 보기 좋게 출력(괄호/콤마 제거)
    status = obj.get("상태", "알수없음")
    print(f"\n[상태] {status}")

    # 조치 모드
    if isinstance(obj.get("지금 할 일"), list):
        todo = obj.get("지금 할 일", [])
        if todo:
            print("\n지금 할 일")
            for x in todo:
                if isinstance(x, str) and x.strip():
                    print(f"- {x.strip()}")
        return

    # 설명 모드
    summary = obj.get("요약")
    if isinstance(summary, str) and summary.strip():
        print(f"\n요약\n- {summary.strip()}")

    causes = obj.get("주요 원인")
    if isinstance(causes, list) and causes:
        print("\n주요 원인")
        for x in causes:
            if isinstance(x, str) and x.strip():
                print(f"- {x.strip()}")

    actions = obj.get("권장 조치")
    if isinstance(actions, list) and actions:
        print("\n권장 조치")
        for x in actions:
            if isinstance(x, str) and x.strip():
                print(f"- {x.strip()}")

    basis = obj.get("판정 근거")
    if isinstance(basis, dict) and basis:
        print("\n판정 근거")
        score = basis.get("위험 점수")
        if isinstance(score, (int, float)):
            print(f"- 위험 점수: {score}")

        reasons = basis.get("판단 이유")
        if isinstance(reasons, list) and reasons:
            print("- 판단 이유:")
            for r in reasons:
                if isinstance(r, str) and r.strip():
                    print(f"  - {r.strip()}")


def main():
    # 1) 퍼지 로직 실행 → history_packets.jsonl 갱신
    run_fuzzy_main()

    # 2) user_query: 인자 없으면 "설명" 중심(조치 단어 넣으면 조치 모드로 너무 짧아질 수 있음)
    user_query = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else "현재 상태를 근거와 함께 자세히 설명"

    # 3) history 로드
    history_packets = _read_jsonl_tail(HISTORY_PACKETS_PATH, n=60)
    if not history_packets:
        raise FileNotFoundError(f"history_packets가 비어있거나 파일을 찾을 수 없습니다: {HISTORY_PACKETS_PATH}")

    current_packet = history_packets[-1]
    past_packets = history_packets[:-1]

    # 4) LLM 실행 (선택 의존성 — transformers 없으면 여기서만 실패)
    from llm.generate import run_smartfarm_llm_timeseries
    result = run_smartfarm_llm_timeseries(
        current_packet=current_packet,
        history_packets=past_packets,
        current_input=None,
        user_query=user_query,
    )

    # 5) 터미널: 사람용 출력(괄호 제거)
    try:
        obj = json.loads(result)
        if isinstance(obj, dict):
            _print_human_readable(obj)
        else:
            print(result)
    except Exception:
        print(result)


if __name__ == "__main__":
    main()
