"""Crop-specific fuzzy evaluation built on fuzzyLozic.

The crop profile is applied in memory (no read/write of data/crop_config.json),
so simulations never touch the live system state.
"""
from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
_FUZZY_DIR = _ROOT / "fuzzyLozic"
if str(_FUZZY_DIR) not in sys.path:
    sys.path.insert(0, str(_FUZZY_DIR))

import config as _fz_config  # noqa: E402  (fuzzyLozic/config.py)
from fuzzify import fuzzify_inputs  # noqa: E402
from inference import infer  # noqa: E402

PROFILES_DIR = _ROOT / "smartfarm" / "crop_profile" / "profiles"

FUZZY_VARS = ("water_temperature", "pH", "EC", "flow_ratio", "turbidity")
ALERT_LABELS = {"경고", "위험", "긴급"}


@dataclass(frozen=True)
class Evaluation:
    risk_score: float
    risk_label: str
    memberships: Dict[str, Dict[str, float]]

    @property
    def is_alert(self) -> bool:
        return self.risk_label in ALERT_LABELS


class CropModel:
    """Fuzzy risk model for one crop: R_t = F(x_t; phi)."""

    def __init__(self, crop: str = "basil", profiles_dir: Path = PROFILES_DIR) -> None:
        profile_path = Path(profiles_dir) / f"{crop}.json"
        if not profile_path.exists():
            raise FileNotFoundError(f"crop profile not found: {profile_path}")
        self.profile = json.loads(profile_path.read_text(encoding="utf-8"))
        self.crop = str(self.profile.get("crop", crop))
        self.crop_ko = str(self.profile.get("crop_ko", ""))

        cfg = copy.deepcopy(_fz_config.default_config())
        overrides = {
            "memberships": {
                k: {lbl: pts for lbl, pts in v.items() if isinstance(pts, list)}
                for k, v in (self.profile.get("fuzzy_params") or {}).items()
                if isinstance(v, dict)
            },
            "rule_weights": {},
        }
        _fz_config._apply_membership_overrides(cfg, overrides)
        self.cfg = cfg

        targets = dict(self.profile.get("control_targets") or {})
        self.ec_mode = str(targets.get("ec_mode") or self.profile.get("cultivation_mode") or "water")
        self.targets: Dict[str, float] = {
            "water_temperature": float(targets.get("target_water_temp", 20.0)),
            "pH": float(targets.get("target_ph", 6.3)),
            "EC": float(targets.get("target_ec", 1.5)),
            "flow_ratio": float(targets.get("target_flow_ratio", 1.0)),
            "turbidity": 0.5,
        }
        self.optimal_ranges = self._optimal_ranges()
        self._cache: Dict[Tuple[float, ...], Evaluation] = {}

    # ------------------------------------------------------------------
    def _var_cfg(self, var: str):
        v = self.cfg["variables"][var]
        return v[self.ec_mode] if var == "EC" else v

    def _optimal_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Core (membership == 1) of each variable's stable set."""
        out: Dict[str, Tuple[float, float]] = {}
        for var in FUZZY_VARS:
            for shape in self._var_cfg(var).shapes:
                if "안정" in shape.label:
                    pts = shape.points
                    lo, hi = (pts[1], pts[2]) if len(pts) == 4 else (pts[1], pts[1])
                    out[var] = (float(lo), float(hi))
                    break
        return out

    def stable_label(self, var: str) -> Optional[str]:
        for shape in self._var_cfg(var).shapes:
            if "안정" in shape.label:
                return shape.label
        return None

    # ------------------------------------------------------------------
    def evaluate(self, state: Mapping[str, float]) -> Evaluation:
        key = (
            round(float(state.get("water_temperature", 0.0)), 2),
            round(float(state.get("pH", 0.0)), 2),
            round(float(state.get("EC", 0.0)), 3),
            round(float(state.get("flow_ratio", 0.0)), 3),
            round(float(state.get("turbidity", 0.0)), 2),
        )
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        inputs = dict(zip(FUZZY_VARS, key))
        memberships = fuzzify_inputs(inputs, self.cfg, ec_mode=self.ec_mode)
        score, label, *_ = infer(memberships, self.cfg)
        ev = Evaluation(float(score), str(label), memberships)
        if len(self._cache) > 200_000:
            self._cache.clear()
        self._cache[key] = ev
        return ev

    def in_optimal_range(self, ev: Evaluation) -> bool:
        """All fuzzy variables at least half inside their stable set."""
        for var in FUZZY_VARS:
            lbl = self.stable_label(var)
            if lbl is None:
                continue
            if ev.memberships.get(var, {}).get(lbl, 0.0) < 0.5:
                return False
        return True

    def danger_vars(self, ev: Evaluation) -> list[str]:
        """Variables whose danger set ('위험' in label) membership is >= 0.5."""
        out = []
        for var, mem in ev.memberships.items():
            for lbl, mu in mem.items():
                if "위험" in lbl and mu >= 0.5:
                    out.append(var)
                    break
        return out


@lru_cache(maxsize=16)
def get_crop_model(crop: str) -> CropModel:
    return CropModel(crop)


def available_crops() -> list[str]:
    return sorted(p.stem for p in PROFILES_DIR.glob("*.json"))
