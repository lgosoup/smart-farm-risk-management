import sys
from pathlib import Path

# fuzzyLozic/ 를 sys.path에 추가 → 루트에서 pytest 실행 가능
_FUZZY_DIR = Path(__file__).resolve().parent.parent
if str(_FUZZY_DIR) not in sys.path:
    sys.path.insert(0, str(_FUZZY_DIR))
