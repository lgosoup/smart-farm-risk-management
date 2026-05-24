"""Persistence layer for Crop Profile JSON files."""
from __future__ import annotations

import json
from pathlib import Path


# Absolute path for fuzzyLozic integration
_FUZZY_CONFIG_PATH = Path(
    r"E:\Smart Farm Risk Management System\Smart Farm Risk Management System"
    r"\fuzzyLozic\crop_config.json"
)

# Default profiles directory: sibling of this file's directory
_DEFAULT_PROFILES_DIR = Path(__file__).parent / "profiles"


class CropProfileRepository:
    """Load, save, and activate crop profiles stored as JSON files.

    Parameters
    ----------
    profiles_dir:
        Absolute or relative path to the directory that contains profile JSON
        files.  Defaults to ``smartfarm/crop_profile/profiles/`` relative to
        this module's location.
    """

    def __init__(self, profiles_dir: str | None = None) -> None:
        self._dir = Path(profiles_dir) if profiles_dir else _DEFAULT_PROFILES_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _profile_path(self, crop_name: str) -> Path:
        return self._dir / f"{crop_name.lower()}.json"

    def _read_json(self, path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_json(self, path: Path, data: dict) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def save(self, profile: dict) -> Path:
        """Save profile to ``profiles/{crop}.json`` and return the path."""
        crop_name = profile.get("crop", "unknown")
        path = self._profile_path(crop_name)
        self._write_json(path, profile)
        return path

    def load(self, crop_name: str) -> dict | None:
        """Load a profile by English or Korean crop name.

        Searches English name first, then scans all profiles for ``crop_ko``.
        Returns ``None`` if not found.
        """
        # Try direct English name
        path = self._profile_path(crop_name)
        data = self._read_json(path)
        if data is not None:
            return data

        # Scan all profiles for Korean name match
        for p in self._dir.glob("*.json"):
            candidate = self._read_json(p)
            if candidate and candidate.get("crop_ko") == crop_name:
                return candidate

        return None

    def list_crops(self) -> list[str]:
        """Return sorted list of available crop names (English)."""
        names = []
        for p in sorted(self._dir.glob("*.json")):
            data = self._read_json(p)
            if data and "crop" in data:
                names.append(data["crop"])
        return names

    def activate(self, crop_name: str) -> bool:
        """Mark a crop as the active crop.

        Writes ``current_crop.json`` to the parent of ``profiles_dir`` (i.e.
        ``smartfarm/crop_profile/current_crop.json``) and also writes the
        fuzzyLozic integration config.

        Returns ``True`` if successful, ``False`` if the profile is not found.
        """
        profile = self.load(crop_name)
        if profile is None:
            return False

        # Write current_crop.json alongside the profiles/ directory
        current_path = self._dir.parent / "current_crop.json"
        self._write_json(current_path, {"active_crop": profile.get("crop"), "profile": profile})

        # Write fuzzyLozic integration config
        self._write_fuzzy_config(profile)
        return True

    def get_active(self) -> dict | None:
        """Load the currently active crop profile."""
        current_path = self._dir.parent / "current_crop.json"
        data = self._read_json(current_path)
        if data is None:
            return None
        return data.get("profile")

    # ------------------------------------------------------------------ #
    # FuzzyLozic integration                                               #
    # ------------------------------------------------------------------ #

    def _to_fuzzy_overrides(self, profile: dict) -> dict:
        """Convert profile fuzzy_params to improvement_state override format.

        Returns a dict suitable for the ``overrides`` key in improvement_state.json:
        ``{"memberships": {...}, "rule_weights": {...}}``
        """
        fuzzy_params: dict = profile.get("fuzzy_params", {})
        rule_weights: dict = profile.get("rule_weights", {})

        # Map profile rule_weight keys to fuzzy rule IDs
        _rw_map = {
            "R6_high_temp_low_flow": "R6",
            "R7_high_ec_low_flow":   "R7",
            "R8_low_flow_turbidity": "R8",
            "R9_high_ec_flow_border": "R9",
        }
        fuzzy_rule_weights: dict = {}
        for profile_key, rule_id in _rw_map.items():
            if profile_key in rule_weights:
                fuzzy_rule_weights[rule_id] = rule_weights[profile_key]

        # Memberships: EC_hydro maps to the EC_hydro sub-key in config
        memberships: dict = {}
        for var_key, labels in fuzzy_params.items():
            memberships[var_key] = labels

        return {"memberships": memberships, "rule_weights": fuzzy_rule_weights}

    def _write_fuzzy_config(self, profile: dict) -> None:
        """Write ``fuzzyLozic/crop_config.json`` from the current profile."""
        overrides = self._to_fuzzy_overrides(profile)
        rule_weights_raw: dict = profile.get("rule_weights", {})

        _rw_map = {
            "R6_high_temp_low_flow": "R6",
            "R7_high_ec_low_flow":   "R7",
            "R8_low_flow_turbidity": "R8",
            "R9_high_ec_flow_border": "R9",
        }
        rule_weights_short: dict = {}
        for pk, rid in _rw_map.items():
            rule_weights_short[rid] = rule_weights_raw.get(pk, 1.0)

        crop_config: dict = {
            "crop":    profile.get("crop"),
            "crop_ko": profile.get("crop_ko"),
            "ec_mode": profile.get("control_targets", {}).get("ec_mode", "hydro"),
            "control_targets":   profile.get("control_targets", {}),
            "improvement_limits": profile.get("improvement_limits", {}),
            "overrides": {
                "memberships": overrides.get("memberships", {}),
                "rule_weights": rule_weights_short,
            },
        }

        _FUZZY_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._write_json(_FUZZY_CONFIG_PATH, crop_config)
