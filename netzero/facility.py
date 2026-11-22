"""Function-based device abstraction.

A facility is described only by data (profiles/facilities/*.json). Each device
declares which functions it performs, how strongly it is expected to move each
environment variable (a prior that the learned model refines), its control
levels and its power draw. Control candidates and hardware commands are derived
from this data, so the optimizer never refers to a concrete device by name.
"""
from __future__ import annotations

import copy
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

FACILITIES_DIR = Path(__file__).resolve().parent / "profiles" / "facilities"

# function name -> (environment variable, direction)
FUNCTIONS: Dict[str, Tuple[Optional[str], int]] = {
    "water_temp_down": ("water_temperature", -1),
    "water_temp_up": ("water_temperature", +1),
    "flow_up": ("flow_ratio", +1),
    "ec_up": ("EC", +1),
    "ec_down": ("EC", -1),
    "turbidity_down": ("turbidity", -1),
    "oxygen_supply": (None, 0),
    "air_exchange": (None, 0),
}

Control = Dict[str, float]


@dataclass(frozen=True)
class Effect:
    var: str
    model: str  # "direct": dx = k*u  |  "exchange": dx = k*u*(outdoor - x)  |  "dilution": dx = k*u*x
    prior: float


@dataclass
class Device:
    id: str
    name: str
    type: str
    functions: List[str]
    levels: List[float]
    unit: str = "min"
    power_w: float = 0.0
    energy_wh_per_unit: float = 0.0
    resource_per_unit: Dict[str, float] = field(default_factory=dict)
    min_level: float = 0.0
    effects: List[Effect] = field(default_factory=list)
    hardware: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "Device":
        effects = []
        for var, spec in (d.get("effects") or {}).items():
            if isinstance(spec, (int, float)):
                spec = {"model": "direct", "prior": float(spec)}
            effects.append(Effect(var=str(var), model=str(spec.get("model", "direct")), prior=float(spec.get("prior", 0.0))))
        unknown = [f for f in d.get("functions", []) if f not in FUNCTIONS]
        if unknown:
            raise ValueError(f"device {d.get('id')}: unknown functions {unknown}")
        min_level = float(d.get("min_level", 0.0))
        levels = sorted({float(x) for x in d.get("levels", [0.0]) if float(x) >= min_level} | {min_level})
        return cls(
            id=str(d["id"]),
            name=str(d.get("name", d["id"])),
            type=str(d.get("type", "generic")),
            functions=list(d.get("functions", [])),
            levels=levels,
            unit=str(d.get("unit", "min")),
            power_w=float(d.get("power_w", 0.0)),
            energy_wh_per_unit=float(d.get("energy_wh_per_unit", 0.0)),
            resource_per_unit={str(k): float(v) for k, v in (d.get("resource_per_unit") or {}).items()},
            min_level=min_level,
            effects=effects,
            hardware=dict(d.get("hardware") or {}),
        )

    def energy_wh(self, level: float) -> float:
        if level <= 0:
            return 0.0
        if self.unit == "min":
            return self.power_w * level / 60.0
        return self.energy_wh_per_unit * level

    def resources(self, level: float) -> Dict[str, float]:
        return {k: v * level for k, v in self.resource_per_unit.items()}

    def directions(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for f in self.functions:
            var, sign = FUNCTIONS[f]
            if var is not None:
                out[var] = sign
        return out


@dataclass
class Facility:
    facility_id: str
    name: str
    description: str
    devices: List[Device]
    step_minutes: int = 60
    prior_thermal_coupling: float = 0.1
    raw: Dict[str, object] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    def device(self, device_id: str) -> Device:
        for d in self.devices:
            if d.id == device_id:
                return d
        raise KeyError(device_id)

    def without(self, device_id: str) -> "Facility":
        """Copy of this facility with one device removed (hardware change scenario)."""
        f = copy.deepcopy(self)
        f.devices = [d for d in f.devices if d.id != device_id]
        return f

    def idle_control(self) -> Control:
        return {d.id: d.min_level for d in self.devices}

    def providers(self, function: str) -> List[Device]:
        return [d for d in self.devices if function in d.functions]

    # ------------------------------------------------------------------
    def candidates(self, needed_functions: Optional[Iterable[str]] = None) -> List[Control]:
        """Enumerate control candidates.

        Devices that perform none of the needed functions stay at their minimum
        level, which keeps the search space small (function-driven pruning).
        Passing None enumerates every device.
        """
        needed = None if needed_functions is None else set(needed_functions)
        axes: List[Sequence[float]] = []
        for d in self.devices:
            if needed is None or any(f in needed for f in d.functions):
                axes.append(d.levels)
            else:
                axes.append([d.min_level])
        ids = [d.id for d in self.devices]
        return [dict(zip(ids, combo)) for combo in itertools.product(*axes)]

    def energy_wh(self, control: Mapping[str, float]) -> float:
        return sum(d.energy_wh(float(control.get(d.id, 0.0))) for d in self.devices)

    def resources(self, control: Mapping[str, float]) -> Dict[str, float]:
        total: Dict[str, float] = {}
        for d in self.devices:
            for k, v in d.resources(float(control.get(d.id, 0.0))).items():
                total[k] = total.get(k, 0.0) + v
        return total

    # ------------------------------------------------------------------
    def build_commands(self, control: Mapping[str, float]) -> Dict[str, Dict[str, object]]:
        """Translate abstract device levels into hardware commands.

        Pin numbers, drivers and protection sequences live in each device's
        `hardware` block, so swapping hardware only means editing the profile.
        """
        commands: Dict[str, Dict[str, object]] = {}
        for d in self.devices:
            level = float(control.get(d.id, 0.0))
            hw = d.hardware or {}
            cmd: Dict[str, object] = {
                "type": d.type,
                "enabled": level > 0,
                "level": level,
                "unit": d.unit,
                "estimated_wh": round(d.energy_wh(level), 4),
                "functions": list(d.functions),
            }
            if d.unit == "min":
                cmd["duration_sec"] = int(round(level * 60))
            else:
                cmd["target_amount"] = level
            if hw.get("pins"):
                cmd["pins"] = hw["pins"]
            if hw.get("driver"):
                cmd["driver"] = hw["driver"]
            prot = hw.get("protection")
            if isinstance(prot, dict) and level > 0:
                support = hw.get("support_pins") or {}
                delay = int(prot.get("startup_delay_sec", 0))
                cooldown = int(prot.get("cooldown_sec", 0))
                main_pins = list((hw.get("pins") or {}).values())
                support_pins = list(support.values())
                cmd["sequence"] = {
                    "start": [{"pins": support_pins, "action": "ON"}, {"wait_sec": delay}, {"pins": main_pins, "action": "ON"}],
                    "stop": [{"pins": main_pins, "action": "OFF"}, {"wait_sec": cooldown}, {"pins": support_pins, "action": "OFF"}],
                }
                cmd["support_duration_sec"] = int(round(level * 60)) + delay + cooldown
            commands[d.id] = cmd
        return commands


def load_facility(name_or_path: str | Path) -> Facility:
    p = Path(name_or_path)
    if not p.suffix:
        p = FACILITIES_DIR / f"{name_or_path}.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    return Facility(
        facility_id=str(data.get("facility_id", p.stem)),
        name=str(data.get("name", p.stem)),
        description=str(data.get("description", "")),
        devices=[Device.from_dict(d) for d in data.get("devices", [])],
        step_minutes=int(data.get("step_minutes", 60)),
        prior_thermal_coupling=float(data.get("prior_thermal_coupling", 0.1)),
        raw=data,
    )


def simulator_truth(name_or_path: str | Path) -> Dict[str, object]:
    """Hidden physical parameters of a facility. Only the simulator reads this."""
    p = Path(name_or_path)
    if not p.suffix:
        p = FACILITIES_DIR / f"{name_or_path}.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    return dict(data.get("simulator_truth") or {})


def available_facilities() -> List[str]:
    return sorted(p.stem for p in FACILITIES_DIR.glob("*.json"))
