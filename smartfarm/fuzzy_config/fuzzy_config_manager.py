"""
FuzzyConfigManager: Loads a Crop Profile and injects settings into the fuzzy logic system.

Integration architecture:
  CropProfile -> FuzzyConfigManager -> crop_config.json -> fuzzyLozic/config.py load_config()
                                    |
                                    FuzzyEngine (membership functions)
                                    RuleEngine (rule weights)
                                    Controller (control targets)
                                    ImprovementModule (delta caps)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


FUZZY_LOGIC_DIR = Path(__file__).parent.parent.parent / "fuzzyLozic"
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CROP_CONFIG_PATH = _DATA_DIR / "crop_config.json"
PROFILES_DIR = Path(__file__).parent.parent / "crop_profile" / "profiles"


class FuzzyConfigManager:
    def __init__(
        self,
        profiles_dir: Path | None = None,
        crop_config_path: Path | None = None,
    ):
        self.profiles_dir = profiles_dir or PROFILES_DIR
        self.crop_config_path = crop_config_path or CROP_CONFIG_PATH
        self._current_profile: dict | None = None

    # ------------------------------------------------------------------
    # Profile loading
    # ------------------------------------------------------------------

    def load_profile(self, crop_name: str) -> dict:
        """
        Load a crop profile by name (English or Korean).
        Searches profiles_dir for {crop_name}.json or matches crop_ko field.
        Raises FileNotFoundError if not found.
        """
        profiles_dir = Path(self.profiles_dir)

        # 1. Try direct filename match (e.g. "basil.json" or "바질.json")
        for candidate in [crop_name, crop_name.lower()]:
            direct = profiles_dir / f"{candidate}.json"
            if direct.exists():
                try:
                    data = json.loads(direct.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        self._current_profile = data
                        return data
                except Exception as exc:
                    raise FileNotFoundError(
                        f"Profile file {direct} exists but could not be parsed: {exc}"
                    ) from exc

        # 2. Scan all JSON files in profiles_dir for crop/crop_ko match
        if profiles_dir.exists():
            for json_file in profiles_dir.glob("*.json"):
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(data, dict):
                    continue
                crop_en = str(data.get("crop", "")).lower()
                crop_ko = str(data.get("crop_ko", ""))
                if crop_en == crop_name.lower() or crop_ko == crop_name:
                    self._current_profile = data
                    return data

        raise FileNotFoundError(
            f"No crop profile found for '{crop_name}' in {profiles_dir}"
        )

    # ------------------------------------------------------------------
    # Activation / deactivation
    # ------------------------------------------------------------------

    def activate(self, crop_name: str) -> bool:
        """
        Load crop profile and write crop_config.json to fuzzyLozic/.
        Returns True on success.

        This is the main entry point to switch crops.
        After calling this, load_config() in fuzzyLozic will pick up new settings.
        """
        try:
            profile = self.load_profile(crop_name)
            crop_config = self._profile_to_crop_config(profile)
            target = Path(self.crop_config_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(crop_config, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return True
        except Exception:
            return False

    def deactivate(self) -> None:
        """Remove crop_config.json (revert to system defaults)."""
        target = Path(self.crop_config_path)
        if target.exists():
            target.unlink()
        self._current_profile = None

    # ------------------------------------------------------------------
    # Status accessors
    # ------------------------------------------------------------------

    def get_current_crop(self) -> str | None:
        """Return name of currently active crop, or None."""
        if self._current_profile is None:
            # Try reading from the existing crop_config.json
            target = Path(self.crop_config_path)
            if target.exists():
                try:
                    data = json.loads(target.read_text(encoding="utf-8"))
                    return data.get("crop") or None
                except Exception:
                    pass
            return None
        return self._current_profile.get("crop") or None

    def get_ec_mode(self) -> str:
        """Return ec_mode for current profile."""
        if self._current_profile is None:
            return "water"
        mode = (
            self._current_profile.get("ec_mode")
            or (self._current_profile.get("control_targets") or {}).get("ec_mode")
            or self._current_profile.get("cultivation_mode")
            or "water"
        )
        return str(mode)

    def get_control_targets(self) -> dict:
        """Return control targets for current profile."""
        if self._current_profile is None:
            return {}
        return dict(self._current_profile.get("control_targets", {}))

    def get_rule_weights(self) -> dict:
        """
        Return flat rule weights dict for inference engine.
        Maps profile rule_weights keys to config rule IDs.
        Example: {"R6_high_temp_low_do": 1.2} -> {"R6": 1.2}
        """
        if self._current_profile is None:
            return {}
        raw = self._current_profile.get("rule_weights", {}) or {}
        return self._rule_weights_to_config(raw)

    def get_improvement_limits(self) -> dict:
        """Return delta caps for improvement module."""
        if self._current_profile is None:
            return {}
        return dict(self._current_profile.get("improvement_limits", {}))

    def status(self) -> dict:
        """Return current manager status: active crop, profile summary, config path."""
        active = self.get_current_crop()
        result: dict = {
            "active_crop": active,
            "crop_config_path": str(self.crop_config_path),
            "profiles_dir": str(self.profiles_dir),
            "config_file_exists": Path(self.crop_config_path).exists(),
        }
        if self._current_profile is not None:
            result["ec_mode"] = self.get_ec_mode()
            result["control_targets"] = self.get_control_targets()
            result["rule_weights"] = self.get_rule_weights()
            result["improvement_limits"] = self.get_improvement_limits()
        return result

    def list_available_crops(self) -> list[str]:
        """List all crops available in profiles_dir."""
        profiles_dir = Path(self.profiles_dir)
        if not profiles_dir.exists():
            return []
        crops: list[str] = []
        for json_file in sorted(profiles_dir.glob("*.json")):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            name = data.get("crop") or json_file.stem
            crops.append(str(name))
        return crops

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def _profile_to_crop_config(self, profile: dict) -> dict:
        """Convert Crop Profile to crop_config.json format.

        Produces:
        {
          "crop": str, "crop_ko": str, "ec_mode": str,
          "control_targets": dict,
          "improvement_limits": dict,
          "overrides": {
            "memberships": dict,
            "rule_weights": dict,
          }
        }
        """
        fuzzy_params = profile.get("fuzzy_params", {}) or {}
        rule_weights_raw = profile.get("rule_weights", {}) or {}
        improvement_limits = profile.get("improvement_limits", {}) or {}
        control_targets = profile.get("control_targets", {}) or {}
        ec_mode = str(
            profile.get("ec_mode")
            or (profile.get("control_targets") or {}).get("ec_mode")
            or profile.get("cultivation_mode")
            or "water"
        )

        memberships_block = self._fuzzy_params_to_memberships(fuzzy_params)
        rule_weights_block = self._rule_weights_to_config(rule_weights_raw)

        crop_config: dict = {
            "crop": profile.get("crop", "unknown"),
            "crop_ko": profile.get("crop_ko", ""),
            "ec_mode": ec_mode,
            "control_targets": control_targets,
            "improvement_limits": improvement_limits,
            "overrides": {
                "memberships": memberships_block,
                "rule_weights": rule_weights_block,
            },
        }
        return crop_config

    def _fuzzy_params_to_memberships(self, fuzzy_params: dict) -> dict:
        """
        Convert profile fuzzy_params format to config override format.

        Input:  {"T": {"안정(최적)": [19,20,25,26]}, "EC_hydro": {...}}
        Output: {"T": {"안정(최적)": [19,20,25,26]}, "EC_hydro": {...}}

        (memberships dict is used directly inside the "memberships" key of overrides)
        """
        # Pass through all keys as-is; the config layer handles EC_water / EC_hydro split
        result: dict = {}
        for var_key, labels in fuzzy_params.items():
            if isinstance(labels, dict):
                result[var_key] = {
                    lbl: pts for lbl, pts in labels.items()
                    if isinstance(pts, list)
                }
        return result

    def _rule_weights_to_config(self, rule_weights: dict) -> dict:
        """
        Convert profile rule_weights to config format.

        Input:  {"R6_high_temp_low_do": 1.2, "R7_low_flow_low_do": 1.0}
        Output: {"R6": 1.2, "R7": 1.0}

        Also accepts bare rule IDs (e.g. "R6": 1.2) which are passed through.
        """
        # Known named aliases -> rule IDs
        mapping: dict[str, str] = {
            "R6_high_temp_low_do": "R6",
            "R7_low_flow_low_do": "R7",
            "R8_low_flow_turbidity": "R8",
            "R9_high_ec_low_do": "R9",
        }

        result: dict = {}
        for key, weight in rule_weights.items():
            if not isinstance(weight, (int, float)):
                continue
            if key in mapping:
                result[mapping[key]] = float(weight)
            elif key.startswith("R") and key[1:].isdigit():
                # Bare rule ID like "R6"
                result[key] = float(weight)
            # else: unknown key, skip
        return result
