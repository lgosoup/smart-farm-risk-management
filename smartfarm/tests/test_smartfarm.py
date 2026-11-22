"""Regression tests for smartfarm/ and research/ (offline, no API keys, no network).

Run from repo root:  python -m pytest -q smartfarm/tests
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FUZZY_DIR = REPO_ROOT / "fuzzyLozic"
PROFILES_DIR = REPO_ROOT / "smartfarm" / "crop_profile" / "profiles"
CROPS = ["basil", "lettuce", "strawberry", "tomato", "wasabi"]

for _p in (str(REPO_ROOT), str(FUZZY_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class _NoNetwork(Exception):
    pass


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Any real HTTP call fails the test; also make sure no API keys leak in."""
    import requests

    def _boom(*a, **k):
        raise _NoNetwork(f"network call attempted: {a} {k}")

    monkeypatch.setattr(requests.Session, "request", _boom)
    monkeypatch.setattr(requests, "get", _boom)
    monkeypatch.delenv("NONGSARO_API_KEY", raising=False)
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)


def _load_profile(crop: str) -> dict:
    return json.loads((PROFILES_DIR / f"{crop}.json").read_text(encoding="utf-8"))


def _fuzzy_shapes(var_key: str):
    import config as fcfg

    variables = fcfg.default_config()["variables"]
    if var_key.startswith("EC_"):
        return variables["EC"][var_key[3:]].shapes
    return variables[var_key].shapes


# ---------------------------------------------------------------------------
# Crop profiles  <->  fuzzyLozic/config.py
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("crop", CROPS)
def test_profile_schema_keys(crop):
    prof = _load_profile(crop)
    for key in ("crop", "crop_ko", "fuzzy_params", "control_targets",
                "cultivation_mode", "rule_weights", "improvement_limits"):
        assert key in prof, key
    assert prof["crop"] == crop


@pytest.mark.parametrize("crop", CROPS)
def test_profile_labels_match_config_and_sets_are_well_formed(crop):
    prof = _load_profile(crop)
    for var_key, labels in prof["fuzzy_params"].items():
        shapes = _fuzzy_shapes(var_key)
        by_label = {s.label: s for s in shapes}
        for label, pts in labels.items():
            # exact label match (incl. en-dash in '약산성(경고–주의)'), else override is silently ignored
            assert label in by_label, f"{crop}:{var_key}: unknown label {label!r}"
            assert len(pts) == len(by_label[label].points), (crop, var_key, label)
            assert all(a <= b for a, b in zip(pts, pts[1:])), f"non-monotonic {crop}:{var_key}:{label} {pts}"
        # adjacent sets (config order) must overlap: no gaps and no zero-membership points
        ordered = [labels.get(s.label, list(s.points)) for s in shapes]
        for left, right in zip(ordered, ordered[1:]):
            assert right[0] < left[-1], f"{crop}:{var_key}: gap/touch between {left} and {right}"


@pytest.mark.parametrize("crop", CROPS)
def test_profile_boundary_values_have_nonzero_membership(crop):
    from membership import evaluate

    prof = _load_profile(crop)
    for var_key, labels in prof["fuzzy_params"].items():
        shapes = _fuzzy_shapes(var_key)
        merged = [(s.shape, tuple(labels.get(s.label, s.points))) for s in shapes]
        lo = min(p[0] for _, p in merged)
        hi = max(p[-1] for _, p in merged)
        # every set breakpoint strictly inside the universe must be covered by some set
        for _, pts in merged:
            for x in pts:
                if lo < x < hi:
                    total = sum(evaluate(sh, p, x) for sh, p in merged)
                    assert total > 0.0, f"{crop}:{var_key}: x={x} has zero membership everywhere"


@pytest.mark.parametrize("crop", CROPS)
def test_rule_weight_keys_map_to_real_rules(crop):
    import config as fcfg
    from rules import build_rules
    from smartfarm.fuzzy_config.fuzzy_config_manager import FuzzyConfigManager

    cfg = fcfg.default_config()
    rule_ids = set(build_rules(cfg).keys())
    mapped = FuzzyConfigManager()._rule_weights_to_config(_load_profile(crop)["rule_weights"])
    assert set(mapped) == {"R6", "R7", "R8", "R9"}
    assert set(mapped) <= set(cfg["rule_weights"])
    assert set(mapped) <= rule_ids


@pytest.mark.parametrize("crop", CROPS)
def test_activate_writes_temp_config_and_load_config_applies_it(crop, tmp_path):
    import config as fcfg
    from smartfarm.fuzzy_config.fuzzy_config_manager import FuzzyConfigManager

    cc = tmp_path / "crop_config.json"
    mgr = FuzzyConfigManager(crop_config_path=cc)
    assert mgr.activate(crop) is True
    assert cc.exists()

    prof = _load_profile(crop)
    cfg = fcfg.load_config(state_path=str(tmp_path / "none.json"), crop_config_path=str(cc))
    assert cfg["_crop_loaded"] == crop
    assert cfg["ec_mode_active"] == prof["control_targets"]["ec_mode"]
    for var_key, labels in prof["fuzzy_params"].items():
        if var_key.startswith("EC_"):
            var = cfg["variables"]["EC"][var_key[3:]]
        else:
            var = cfg["variables"][var_key]
        applied = {s.label: list(s.points) for s in var.shapes}
        for label, pts in labels.items():
            assert applied[label] == [float(x) for x in pts], (crop, var_key, label)
    rw = prof["rule_weights"]
    assert cfg["rule_weights"]["R6"] == rw["R6_high_temp_low_flow"]
    assert cfg["rule_weights"]["R9"] == rw["R9_high_ec_flow_border"]


def test_activate_by_korean_name_and_unknown(tmp_path):
    from smartfarm.fuzzy_config.fuzzy_config_manager import FuzzyConfigManager

    mgr = FuzzyConfigManager(crop_config_path=tmp_path / "cc.json")
    assert mgr.activate("바질") is True
    assert mgr.get_current_crop() == "basil"
    assert mgr.activate("없는작물") is False


def test_repository_activate_uses_injected_path_not_hardcoded_drive(tmp_path):
    from smartfarm.crop_profile import crop_profile_repository as repo_mod
    from smartfarm.fuzzy_config.fuzzy_config_manager import FuzzyConfigManager

    assert "E:" not in str(repo_mod._FUZZY_CONFIG_PATH)
    assert repo_mod._FUZZY_CONFIG_PATH == REPO_ROOT / "data" / "crop_config.json"

    pdir = tmp_path / "profiles"
    shutil.copytree(PROFILES_DIR, pdir)
    cc = tmp_path / "crop_config.json"
    repo = repo_mod.CropProfileRepository(profiles_dir=str(pdir), crop_config_path=cc)
    assert repo.activate("basil") is True
    written = json.loads(cc.read_text(encoding="utf-8"))
    expected = FuzzyConfigManager()._profile_to_crop_config(_load_profile("basil"))
    assert written == expected
    assert (tmp_path / "current_crop.json").exists()


# ---------------------------------------------------------------------------
# FuzzyParamConverter
# ---------------------------------------------------------------------------

def test_converter_reproduces_committed_profiles():
    from smartfarm.crop_profile.fuzzy_param_converter import FuzzyParamConverter

    for crop in CROPS:
        prof = _load_profile(crop)
        new = FuzzyParamConverter().convert(prof["raw_growth_params"])
        for var_key, labels in prof["fuzzy_params"].items():
            for label, pts in labels.items():
                assert new[var_key][label] == [float(x) for x in pts], (crop, var_key, label)


def test_converter_narrow_ec_range_is_monotonic_and_boundaries_covered():
    from membership import evaluate
    from smartfarm.crop_profile.fuzzy_param_converter import FuzzyParamConverter

    out = FuzzyParamConverter().convert({"ec_vegetative": [1.0, 1.1], "water_temp_optimal": [18, 22],
                                         "ph_optimal": [5.5, 6.5]})
    for pts in out["EC_hydro"].values():
        assert all(a <= b for a, b in zip(pts, pts[1:])), pts
    ec = out["EC_hydro"]

    def total(x):
        return sum(evaluate("tri" if len(p) == 3 else "trap", tuple(p), x) for p in ec.values())

    assert total(1.0) > 0 and total(1.1) > 0
    wt = out["water_temperature"]
    x = 17.0  # opt_min - 1
    assert sum(evaluate("tri" if len(p) == 3 else "trap", tuple(p), x) for p in wt.values()) > 0


# ---------------------------------------------------------------------------
# Document NLP + unit normalisation
# ---------------------------------------------------------------------------

def _extract(text: str) -> list[dict]:
    from smartfarm.document_nlp.document_nlp_extractor import DocumentNLPExtractor

    return DocumentNLPExtractor().extract({"raw_text": text, "source": "fixture", "source_type": "test"}, {})


@pytest.mark.parametrize("text,field,lo,hi,unit", [
    ("바질 수경재배 시 적정 수온은 18~22℃로 관리한다.", "T_water", 18.0, 22.0, "°C"),
    ("수경재배 시 적정 수온은 18 ~ 22 °C 이다.", "T_water", 18.0, 22.0, "°C"),
    ("수경재배 수온은 18～22℃ 범위가 적당하다.", "T_water", 18.0, 22.0, "°C"),  # fullwidth tilde
    ("양액의 EC 1.2~1.8 dS/m 범위가 적합하다.", "EC", 1.2, 1.8, "dS/m"),
    ("양액 EC는 1.2–1.8 mS/cm 로 관리한다.", "EC", 1.2, 1.8, "mS/cm"),
    ("배양액 pH 5.5-6.5 로 유지한다.", "pH", 5.5, 6.5, "pH"),
    ("배양액 PH 5.5-6.5 로 유지한다.", "pH", 5.5, 6.5, "pH"),  # upper-case unit
])
def test_nlp_range_extraction(text, field, lo, hi, unit):
    cands = [c for c in _extract(text) if c["field"] == field]
    assert cands, _extract(text)
    c = cands[0]
    assert (c["min"], c["max"], c["unit"]) == (lo, hi, unit)
    assert c["confidence"] >= 0.65


def test_nlp_single_value_is_kept():
    from smartfarm.generate_crop_profile import _map_nlp_candidates

    cands = _extract("배양액 pH는 6.0 으로 맞춘다.")
    assert cands and cands[0]["value"] == 6.0
    mapped = _map_nlp_candidates(cands)
    assert mapped and mapped[0]["param"] == "ph_optimal" and mapped[0]["value"] == 6.0


@pytest.mark.parametrize("unit,value,expected", [
    ("mS/cm", [1.2, 1.8], [1.2, 1.8]),
    ("dS/m", [1.2, 1.8], [1.2, 1.8]),
    ("\u00b5S/cm", [1200, 1800], [1.2, 1.8]),   # micro sign
    ("\u03bcS/cm", [1200, 1800], [1.2, 1.8]),   # Greek mu
    ("uS/cm", 1500, 1.5),
])
def test_unit_normalizer_ec(unit, value, expected):
    from smartfarm.crop_profile.unit_normalizer import UnitNormalizer

    out = UnitNormalizer().normalize([{"param": "ec_vegetative", "unit": unit, "value": value}])[0]
    assert out["unit_conversion"]["converted"] is True
    assert out["value_normalized"] == expected
    assert out["unit_normalized"] == "mS/cm"


@pytest.mark.parametrize("unit,value,expected", [
    ("℃", [18, 22], [18, 22]),
    ("°C", [18, 22], [18, 22]),
    ("°F", [68, 77], [20.0, 25.0]),
    ("℉", 68, 20.0),
])
def test_unit_normalizer_temperature(unit, value, expected):
    from smartfarm.crop_profile.unit_normalizer import UnitNormalizer

    out = UnitNormalizer().normalize([{"param": "water_temp_optimal", "unit": unit, "value": value}])[0]
    assert out["unit_conversion"]["converted"] is True
    assert out["value_normalized"] == expected


def test_unit_none_does_not_crash_normalizer_or_builder():
    from smartfarm.crop_profile.crop_profile_builder import CropProfileBuilder
    from smartfarm.crop_profile.unit_normalizer import UnitNormalizer
    from smartfarm.generate_crop_profile import _map_nlp_candidates

    mapped = _map_nlp_candidates(_extract("양액 EC는 1.2~1.8 범위로 관리한다."))
    assert mapped and mapped[0]["unit"] is None
    out = UnitNormalizer().normalize(mapped)
    assert out[0]["unit_conversion"]["converted"] is False  # -> pending, not crash
    prof = CropProfileBuilder().build({"crop": "x"}, mapped)
    assert prof["validation"]["pending_count"] == 1


def test_nlp_to_profile_end_to_end():
    from smartfarm.crop_profile.crop_profile_builder import CropProfileBuilder
    from smartfarm.generate_crop_profile import _map_nlp_candidates

    text = ("바질 수경재배 시 적정 수온은 18~22℃로 관리한다.\n"
            "양액의 EC 1200~1800 µS/cm 범위가 적합하다.\n"
            "배양액 pH 5.5-6.5 로 유지한다.")
    mapped = _map_nlp_candidates(_extract(text))
    prof = CropProfileBuilder().build({"crop": "basil", "crop_ko": "바질"}, mapped)
    assert prof["raw_growth_params"]["water_temp_optimal"] == [18.0, 22.0]
    assert prof["raw_growth_params"]["ec_vegetative"] == [1.2, 1.8]
    assert prof["raw_growth_params"]["ph_optimal"] == [5.5, 6.5]
    assert prof["control_targets"]["target_ec"] == 1.5
    assert prof["validation"]["passed_count"] == 3


# ---------------------------------------------------------------------------
# generate_crop_profile pipeline (no API key)
# ---------------------------------------------------------------------------

def test_semantic_scholar_integration_is_importable():
    import smartfarm.generate_crop_profile as gcp

    assert gcp.HAS_RAG is True
    info = gcp.scholar_normalize("basil")
    assert info["crop_common_name_ko"] == "바질"


def test_generate_without_key_does_not_overwrite_curated_profile(tmp_path, monkeypatch):
    from smartfarm.crop_profile import crop_profile_repository as repo_mod
    import smartfarm.generate_crop_profile as gcp

    pdir = tmp_path / "profiles"
    shutil.copytree(PROFILES_DIR, pdir)
    before = (pdir / "basil.json").read_text(encoding="utf-8")
    monkeypatch.setattr(repo_mod, "_DEFAULT_PROFILES_DIR", pdir)
    nongsaro_cache = REPO_ROOT / "smartfarm" / "cache"
    existed = nongsaro_cache.exists()

    prof = gcp.generate_crop_profile("바질", activate=False, api_key=None)
    assert prof["crop"] == "basil"
    assert (pdir / "basil.json").read_text(encoding="utf-8") == before
    assert nongsaro_cache.exists() == existed  # client without key leaves nothing on disk


def test_generate_unknown_korean_crop_gets_a_name(tmp_path, monkeypatch):
    from smartfarm.crop_profile import crop_profile_repository as repo_mod
    import smartfarm.generate_crop_profile as gcp

    monkeypatch.setattr(repo_mod, "_DEFAULT_PROFILES_DIR", tmp_path)
    prof = gcp.generate_crop_profile("고구마", activate=False, api_key=None)
    assert prof["crop"]  # was None before -> crash on save / activate
    assert (tmp_path / f"{prof['crop'].lower()}.json").exists()


# ---------------------------------------------------------------------------
# Nongsaro client (mocked HTTP)
# ---------------------------------------------------------------------------

_XML_OK = """<?xml version="1.0" encoding="UTF-8"?>
<response><header><resultCode>00</resultCode><resultMsg>OK</resultMsg></header>
<body><items>
<item><cntntsNo>1234</cntntsNo><cntntsNm>바질 수경재배</cntntsNm>
<cntnts>&lt;p&gt;적정 수온은 18~22℃ 이다.&lt;/p&gt; 양액 EC 1.2~1.8 dS/m</cntnts></item>
</items></body></response>""".encode("utf-8")

_XML_ERR = b"""<?xml version="1.0" encoding="UTF-8"?>
<response><header><resultCode>11</resultCode><resultMsg>INVALID KEY</resultMsg></header></response>"""


class _FakeResp:
    def __init__(self, content: bytes):
        self.content = content
        self.text = content.decode("latin-1")  # what requests does without a charset header
        self.status_code = 200

    def raise_for_status(self):
        pass


def test_nongsaro_without_key_returns_empty_and_writes_nothing(tmp_path):
    from smartfarm.public_api.nongsaro_crop_tech_api import NongsaroCropTechApiClient

    cache = tmp_path / "cache"
    client = NongsaroCropTechApiClient(api_key=None, cache_dir=str(cache))
    res = client.fetch_crop_documents({"crop_common_name_ko": "바질"})
    assert res["documents"] == [] and res["fallback"] is True and "error" in res
    assert not cache.exists()


def test_nongsaro_xml_parsing_with_mocked_response(tmp_path, monkeypatch):
    from smartfarm.public_api import nongsaro_crop_tech_api as mod

    monkeypatch.setattr(mod, "_INTER_REQUEST_DELAY", 0.0)
    cache = tmp_path / "cache"
    client = mod.NongsaroCropTechApiClient(api_key="DUMMY-KEY", cache_dir=str(cache))
    monkeypatch.setattr(client._session, "get", lambda *a, **k: _FakeResp(_XML_OK))
    res = client.fetch_crop_documents({"crop_common_name_ko": "바질"})
    docs = res["documents"]
    assert len(docs) == 1
    assert docs[0]["title"] == "바질 수경재배"           # Korean not garbled
    assert "18~22℃" in docs[0]["raw_text"] and "<p>" not in docs[0]["raw_text"]
    assert "DUMMY-KEY" not in docs[0]["url"]
    cached = list(cache.glob("*.json"))
    assert len(cached) == 1 and "DUMMY-KEY" not in cached[0].read_text(encoding="utf-8")

    from smartfarm.document_nlp.document_nlp_extractor import DocumentNLPExtractor
    fields = {c["field"] for c in DocumentNLPExtractor().extract(docs[0], {})}
    assert {"T_water", "EC"} <= fields


def test_nongsaro_error_response_is_not_cached(tmp_path, monkeypatch):
    from smartfarm.public_api import nongsaro_crop_tech_api as mod

    monkeypatch.setattr(mod, "_INTER_REQUEST_DELAY", 0.0)
    cache = tmp_path / "cache"
    client = mod.NongsaroCropTechApiClient(api_key="DUMMY-KEY", cache_dir=str(cache))
    monkeypatch.setattr(client._session, "get", lambda *a, **k: _FakeResp(_XML_ERR))
    res = client.fetch_crop_documents({"crop_common_name_ko": "바질"})
    assert res["documents"] == [] and res["fallback"] is True
    assert not list(cache.glob("*.json")) if cache.exists() else True


# ---------------------------------------------------------------------------
# research/ semantic scholar (offline cache)
# ---------------------------------------------------------------------------

def test_semantic_scholar_paths_are_package_relative():
    from research.research_support import semantic_scholar_search as ss

    assert ss.CACHE_DIR == REPO_ROOT / "research" / "cache" / "semantic_scholar"
    assert ss.OUTPUT_DIR == REPO_ROOT / "research" / "outputs"


@pytest.mark.parametrize("crop,crop_ko", [("basil", "바질"), ("strawberry", "딸기")])
def test_semantic_scholar_bundle_offline_from_cache(crop, crop_ko, monkeypatch):
    from research.research_support import semantic_scholar_search as ss

    monkeypatch.setattr(ss.time, "sleep", lambda s: None)
    before = sorted(p.name for p in ss.CACHE_DIR.iterdir())
    bundle = ss.collect_crop_source_bundle(crop=crop, crop_ko=crop_ko, limit_per_query=2,
                                           max_queries_per_category=2, use_cache=True)
    statuses = {q["status"] for q in bundle["query_results"]}
    assert statuses == {"ok_cached"}
    assert bundle["search_summary"]["total_unique_documents"] > 0
    assert sorted(p.name for p in ss.CACHE_DIR.iterdir()) == before


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------

def _write_img(path: Path, green: int) -> None:
    import cv2
    import numpy as np

    img = np.zeros((32, 32, 3), np.uint8)
    img[:, :, 1] = green  # BGR -> G channel
    ok, buf = cv2.imencode(path.suffix, img)
    assert ok
    buf.tofile(str(path))


def test_vision_reward_with_synthetic_images_in_korean_path(tmp_path):
    pytest.importorskip("cv2")
    from smartfarm.vision.plant_health_analyzer import compute_vision_reward, _find_image_for_date

    cap = tmp_path / "비전_캡처"  # non-ASCII path like C:\Users\...\공부\스마트팜\data
    cap.mkdir()
    today = date.today()
    _write_img(cap / f"{today:%Y%m%d}.png", 220)
    _write_img(cap / f"{today - timedelta(days=1):%Y%m%d}.jpg", 120)
    assert _find_image_for_date(cap, today - timedelta(days=1)).suffix == ".jpg"
    assert _find_image_for_date(cap, today - timedelta(days=7)) is None

    res = compute_vision_reward(cap)
    assert res is not None, "image in non-ASCII path could not be read"
    reward, details = res
    assert details["source"] == "vision"
    assert -0.1 <= reward <= 1.1
    assert details["image_date"] == today.isoformat()
    # greener today than yesterday -> temporal score above neutral-ish color score
    assert details["temporal_score"] > 0.5


def test_vision_reward_none_without_image_and_reward_falls_back(tmp_path):
    pytest.importorskip("cv2")
    from smartfarm.vision.plant_health_analyzer import compute_vision_reward
    from improvement.reward import compute_reward

    assert compute_vision_reward(tmp_path) is None
    reward, details = compute_reward(recent_packets=[{"action_key": "정상", "risk": {"risk_score": 20.0}}],
                                     vision_capture_dir=str(tmp_path))
    assert details["source"] == "heuristic"
    assert isinstance(reward, float)
