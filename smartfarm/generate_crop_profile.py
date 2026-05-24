#!/usr/bin/env python3
"""
Generate a Crop Profile from 농촌진흥청 API documents and activate it in the fuzzy logic system.

Usage:
    python -m smartfarm.generate_crop_profile --crop 바질
    python -m smartfarm.generate_crop_profile --crop basil --activate
    python -m smartfarm.generate_crop_profile --list
    python -m smartfarm.generate_crop_profile --status
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Optional: rag-research semantic scholar integration
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "rag-research"))
    from research_support.semantic_scholar_search import normalize_crop as scholar_normalize
    HAS_RAG = True
except ImportError:
    HAS_RAG = False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a Crop Profile and optionally activate it in the fuzzy logic system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--crop", "-c",
        metavar="CROP_NAME",
        help="Crop name in Korean or English (e.g. 바질 or basil)",
    )
    parser.add_argument(
        "--activate", "-a",
        action="store_true",
        default=False,
        help="After generating the profile, activate it (write crop_config.json)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        default=False,
        help="List all available crop profiles",
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        default=False,
        help="Show current fuzzy config manager status",
    )
    parser.add_argument(
        "--api-key", "-k",
        metavar="KEY",
        default=None,
        help="Nongsaro API key (overrides NONGSARO_API_KEY env var)",
    )
    return parser


def _list_profiles() -> None:
    """Print all available crop profiles."""
    from smartfarm.fuzzy_config.fuzzy_config_manager import FuzzyConfigManager
    mgr = FuzzyConfigManager()
    crops = mgr.list_available_crops()
    if not crops:
        print("No crop profiles found.")
    else:
        print(f"Available crop profiles ({len(crops)}):")
        for c in crops:
            print(f"  - {c}")


def _show_status() -> None:
    """Print current FuzzyConfigManager status."""
    from smartfarm.fuzzy_config.fuzzy_config_manager import FuzzyConfigManager
    mgr = FuzzyConfigManager()
    status = mgr.status()
    print("FuzzyConfigManager status:")
    print(f"  Active crop    : {status.get('active_crop', 'none')}")
    print(f"  Config file    : {status.get('crop_config_path')}")
    print(f"  Config exists  : {status.get('config_file_exists')}")
    print(f"  Profiles dir   : {status.get('profiles_dir')}")
    if status.get("ec_mode"):
        print(f"  EC mode        : {status.get('ec_mode')}")
    if status.get("control_targets"):
        targets = status["control_targets"]
        print(f"  Control targets: pH={targets.get('target_ph')}, "
              f"EC={targets.get('target_ec')}, water_temp={targets.get('target_water_temp')}")


def _fetch_documents(crop_name_ko: str, api_key: str | None) -> list[dict]:
    """Fetch documents from Nongsaro API. Returns empty list on failure."""
    try:
        import os
        from smartfarm.public_api.nongsaro_crop_tech_api import NongsaroCropTechApiClient
        key = api_key or os.environ.get("NONGSARO_API_KEY", "")
        client = NongsaroCropTechApiClient(api_key=key)
        crop_info = {"crop_common_name_ko": crop_name_ko, "crop_common_name_en": ""}
        result = client.fetch_crop_documents(crop_info)
        docs = result.get("documents", []) if isinstance(result, dict) else []
        return docs if isinstance(docs, list) else []
    except Exception as exc:
        print(f"  [WARNING] API fetch failed: {exc}")
        print("  Proceeding with fallback defaults only.")
        return []


_FIELD_TO_PARAM: dict[str, str] = {
    "pH":        "ph_optimal",
    "EC":        "ec_vegetative",
    "T_water":   "water_temp_optimal",
    "T_root":    "water_temp_optimal",
    "T_air":     "air_temp_optimal",
    "FR":        "flow_ratio",
    "Turbidity": "turbidity",
}


def _map_nlp_candidates(candidates: list[dict]) -> list[dict]:
    """Convert DocumentNLPExtractor output (field/min/max) to builder input (param/value)."""
    result = []
    for c in candidates:
        field = c.get("field", "")
        param = _FIELD_TO_PARAM.get(field)
        if not param:
            continue
        lo = c.get("min")
        hi = c.get("max")
        if lo is not None and hi is not None:
            value = [lo, hi]
        elif lo is not None:
            value = lo
        elif hi is not None:
            value = hi
        else:
            continue
        mapped = dict(c)
        mapped["param"] = param
        mapped["value"] = value
        result.append(mapped)
    return result


def _extract_candidates(documents: list[dict], crop_info: dict) -> list[dict]:
    """Run NLP extraction on documents. Returns empty list on failure."""
    if not documents:
        return []
    try:
        from smartfarm.document_nlp.document_nlp_extractor import DocumentNLPExtractor
        extractor = DocumentNLPExtractor()
        candidates: list[dict] = []
        for doc in documents:
            try:
                found = extractor.extract(doc, crop_info)
                if isinstance(found, list):
                    candidates.extend(found)
            except Exception as exc:
                print(f"  [WARNING] NLP extraction failed for doc '{doc.get('title', '')}': {exc}")
        return candidates
    except Exception as exc:
        print(f"  [WARNING] DocumentNLPExtractor unavailable: {exc}")
        return []


def _normalize_units(candidates: list[dict]) -> list[dict]:
    """Normalize candidate units. Returns candidates unchanged on failure."""
    if not candidates:
        return candidates
    try:
        from smartfarm.crop_profile.unit_normalizer import UnitNormalizer
        normalizer = UnitNormalizer()
        return normalizer.normalize(candidates)
    except Exception as exc:
        print(f"  [WARNING] Unit normalization failed: {exc}")
        return candidates


def _validate_candidates(candidates: list[dict], crop_info: dict) -> dict:
    """Validate candidates. Returns dict with passed/pending/rejected."""
    try:
        from smartfarm.crop_profile.candidate_validator import CandidateValidator
        validator = CandidateValidator()
        return validator.validate(candidates, crop_info)
    except Exception as exc:
        print(f"  [WARNING] Candidate validation failed: {exc}")
        return {"passed": [], "pending": [], "rejected": []}


def _build_profile(crop_info: dict, candidates: list[dict]) -> dict:
    """Build the crop profile using CropProfileBuilder."""
    from smartfarm.crop_profile.crop_profile_builder import CropProfileBuilder
    builder = CropProfileBuilder()
    return builder.build(crop_info=crop_info, candidates=candidates)


def _save_profile(profile: dict) -> Path:
    """Save profile via CropProfileRepository."""
    from smartfarm.crop_profile.crop_profile_repository import CropProfileRepository
    repo = CropProfileRepository()
    return repo.save(profile)


def _print_summary(
    crop_name: str,
    profile: dict,
    save_path: Path,
    activated: bool,
) -> None:
    """Print a summary of the generated profile."""
    print()
    print("=" * 60)
    print(f"Crop Profile Generated: {crop_name}")
    print("=" * 60)
    print(f"  Saved to      : {save_path}")
    print(f"  Crop (EN)     : {profile.get('crop', 'unknown')}")
    print(f"  Crop (KO)     : {profile.get('crop_ko', '')}")
    print(f"  Scientific    : {profile.get('scientific_name', '')}")
    print(f"  Growth type   : {profile.get('growth_type', '')}")

    validation = profile.get("validation", {})
    print(f"  Candidates    : {validation.get('passed_count', 0)} passed, "
          f"{validation.get('pending_count', 0)} pending, "
          f"{validation.get('rejected_count', 0)} rejected")
    print(f"  Confidence    : {validation.get('overall_confidence', 0.0):.2f}")

    fallbacks = profile.get("fallbacks_used", [])
    if fallbacks:
        print(f"  Fallbacks used: {', '.join(fallbacks)}")

    ctrl = profile.get("control_targets", {})
    if ctrl:
        print(f"  Control targets:")
        print(f"    pH           = {ctrl.get('target_ph')}")
        print(f"    EC           = {ctrl.get('target_ec')} mS/cm  (mode: {ctrl.get('ec_mode', 'hydro')})")
        print(f"    water_temp   = {ctrl.get('target_water_temp')} C")
        print(f"    flow_ratio   = {ctrl.get('target_flow_ratio')}")

    fuzzy = profile.get("fuzzy_params", {})
    if fuzzy:
        print(f"  Fuzzy params  : {', '.join(sorted(fuzzy.keys()))}")

    if activated:
        print()
        print("  [OK] Profile activated -> fuzzyLozic/crop_config.json written")
    print("=" * 60)


def generate_crop_profile(
    crop_name: str,
    activate: bool = False,
    api_key: str | None = None,
) -> dict:
    """
    Full pipeline: normalize -> fetch -> extract -> validate -> build -> save -> (activate).

    Parameters
    ----------
    crop_name:
        Crop name in Korean or English.
    activate:
        If True, write crop_config.json to fuzzyLozic/ after saving.
    api_key:
        Nongsaro API key (or None to use env var / skip).

    Returns
    -------
    dict
        The generated crop profile.
    """
    print(f"\n[1] Normalizing crop name: '{crop_name}'")
    from smartfarm.crop_input.crop_input_normalizer import CropInputNormalizer
    normalizer = CropInputNormalizer()
    crop_info = normalizer.normalize(crop_name)
    print(f"    -> {crop_info.get('crop_common_name_en', crop_name)} "
          f"/ {crop_info.get('crop_common_name_ko', '')} "
          f"(confidence={crop_info.get('confidence', 0.0):.2f})")

    # Optional: supplement crop_info using RAG semantic scholar module
    if HAS_RAG:
        try:
            print("    (rag-research available: supplementing crop_info via scholar_normalize)")
            rag_info = scholar_normalize(
                crop_info.get("crop_common_name_en") or crop_name
            )
            if isinstance(rag_info, dict):
                # Merge non-null fields from rag_info into crop_info without overwriting
                for k, v in rag_info.items():
                    if v is not None and crop_info.get(k) is None:
                        crop_info[k] = v
        except Exception as exc:
            print(f"    [WARNING] scholar_normalize failed: {exc}")

    # Build a unified crop_info dict for downstream use
    crop_info.setdefault("crop", crop_info.get("crop_common_name_en", crop_name))
    crop_info.setdefault("crop_ko", crop_info.get("crop_common_name_ko", ""))

    print(f"\n[2] Fetching documents from Nongsaro API...")
    crop_name_ko = crop_info.get("crop_ko") or crop_name
    documents = _fetch_documents(crop_name_ko, api_key)
    print(f"    -> {len(documents)} document(s) fetched")

    print(f"\n[3] Running NLP extraction...")
    candidates = _extract_candidates(documents, crop_info)
    print(f"    -> {len(candidates)} candidate(s) extracted")

    print(f"\n[3.5] Mapping NLP fields to profile params...")
    candidates = _map_nlp_candidates(candidates)
    print(f"    -> {len(candidates)} candidate(s) after field mapping")

    print(f"\n[4] Normalizing units...")
    normalized = _normalize_units(candidates)

    print(f"\n[5] Validating candidates...")
    validation_result = _validate_candidates(normalized, crop_info)
    n_passed = len(validation_result.get("passed", []))
    n_pending = len(validation_result.get("pending", []))
    n_rejected = len(validation_result.get("rejected", []))
    print(f"    -> passed={n_passed}, pending={n_pending}, rejected={n_rejected}")

    print(f"\n[6] Building crop profile...")
    # Pass normalized candidates to builder (it will re-validate internally but
    # we already have results; pass all normalized so builder applies its own logic)
    profile = _build_profile(crop_info, normalized)
    print(f"    -> profile built (fallbacks: {profile.get('fallbacks_used', [])})")

    print(f"\n[7] Saving profile...")
    save_path = _save_profile(profile)
    print(f"    -> saved to {save_path}")

    activated = False
    if activate:
        print(f"\n[8] Activating profile in fuzzy logic system...")
        from smartfarm.fuzzy_config.fuzzy_config_manager import FuzzyConfigManager
        mgr = FuzzyConfigManager()
        success = mgr.activate(crop_info.get("crop", crop_name))
        if success:
            activated = True
            print(f"    -> crop_config.json written to {mgr.crop_config_path}")
        else:
            print(f"    [WARNING] Activation failed. Profile saved but not activated.")

    _print_summary(crop_name, profile, save_path, activated)
    return profile


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    # Handle --list
    if args.list:
        _list_profiles()
        return 0

    # Handle --status
    if args.status:
        _show_status()
        return 0

    # Require --crop for profile generation
    if not args.crop:
        parser.print_help()
        return 1

    try:
        generate_crop_profile(
            crop_name=args.crop,
            activate=args.activate,
            api_key=args.api_key,
        )
        return 0
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
