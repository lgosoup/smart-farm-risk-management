import sys
from pathlib import Path

import pytest

# fuzzyLozic/ 를 sys.path에 추가 → 루트에서 pytest 실행 가능
_FUZZY_DIR = Path(__file__).resolve().parent.parent
if str(_FUZZY_DIR) not in sys.path:
    sys.path.insert(0, str(_FUZZY_DIR))


@pytest.fixture(autouse=True)
def _isolate_runtime_config(tmp_path, monkeypatch):
    """테스트가 data/crop_config.json, data/improvement_state.json(현재 작물·밴딧 상태)에
    영향을 받지 않도록 기본 설정만 사용한다."""
    import config

    monkeypatch.setattr(config, "CROP_CONFIG_PATH", str(tmp_path / "crop_config.json"))
    monkeypatch.setattr(config, "DEFAULT_STATE_PATH", str(tmp_path / "improvement_state.json"))
