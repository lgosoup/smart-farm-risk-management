#!/usr/bin/env python3
"""Quick crop activation script. Usage: python -m smartfarm.activate_crop basil"""
import sys
from smartfarm.fuzzy_config.fuzzy_config_manager import FuzzyConfigManager


def main():
    if len(sys.argv) < 2:
        mgr = FuzzyConfigManager()
        status = mgr.status()
        print(f"Active crop: {status.get('active_crop', 'none')}")
        print(f"Available: {', '.join(mgr.list_available_crops())}")
        return

    crop = sys.argv[1]
    mgr = FuzzyConfigManager()
    if mgr.activate(crop):
        targets = mgr.get_control_targets()
        print(f"Activated: {crop}")
        print(f"  EC mode: {mgr.get_ec_mode()}")
        print(f"  Targets: pH={targets.get('target_ph')}, EC={targets.get('target_ec')}, water_temp={targets.get('target_water_temp')}")
    else:
        print(f"Failed to activate: {crop}")
        sys.exit(1)


if __name__ == "__main__":
    main()
