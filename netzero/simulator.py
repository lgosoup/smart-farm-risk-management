"""Virtual smart farm.

The simulator owns hidden physical parameters (theta_sim) taken from the
facility's `simulator_truth` block. Controllers only see noisy sensor values,
the device profile (with priors) and the outdoor forecast, so they have to
learn how this particular facility actually behaves.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from .facility import Facility

CLIMATE_DIR = Path(__file__).resolve().parent / "profiles" / "climate"

STATE_BOUNDS = {
    "water_temperature": (0.0, 45.0),
    "flow_ratio": (0.0, 1.5),
    "EC": (0.0, 6.0),
    "turbidity": (0.0, 30.0),
    "pH": (3.0, 9.5),
}


# ----------------------------------------------------------------------------
# Climate
# ----------------------------------------------------------------------------
@dataclass
class Climate:
    climate_id: str
    name: str
    mean: float
    amplitude: float
    peak_hour: float
    day_sd: float
    humidity_mean: float
    humidity_amplitude: float
    sunrise: float
    sunset: float
    events: List[Dict[str, float]] = field(default_factory=list)
    seed: int = 0

    def __post_init__(self) -> None:
        rng = random.Random(self.seed * 7919 + 17)
        self._day_offsets = [rng.gauss(0.0, self.day_sd) for _ in range(400)]

    def _heatwave(self, day: int) -> tuple[float, float]:
        dm = da = 0.0
        for ev in self.events:
            if ev.get("type") == "heatwave" and ev.get("start_day", 0) <= day < ev.get("end_day", 0):
                dm += float(ev.get("delta_mean_c", 0.0))
                da += float(ev.get("delta_amplitude_c", 0.0))
        return dm, da

    def outdoor_temp(self, t: int) -> float:
        """Outdoor temperature at hour t (t=0 is midnight of day 0)."""
        day, hour = divmod(int(t), 24)
        dm, da = self._heatwave(day)
        base = self.mean + dm + self._day_offsets[day % len(self._day_offsets)]
        amp = self.amplitude + da
        return base + amp * math.cos(2 * math.pi * (hour - self.peak_hour) / 24.0)

    def humidity(self, t: int) -> float:
        hour = int(t) % 24
        return self.humidity_mean - self.humidity_amplitude * math.cos(2 * math.pi * (hour - self.peak_hour) / 24.0)

    def sun(self, t: int) -> float:
        """Relative solar intensity 0..1 (half-sine between sunrise and sunset)."""
        h = (int(t) % 24) + 0.5
        if h <= self.sunrise or h >= self.sunset:
            return 0.0
        return math.sin(math.pi * (h - self.sunrise) / (self.sunset - self.sunrise))


def load_climate(name_or_path: str | Path = "summer_south", seed: int = 0) -> Climate:
    p = Path(name_or_path)
    if not p.suffix:
        p = CLIMATE_DIR / f"{name_or_path}.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    return Climate(
        climate_id=str(d.get("climate_id", p.stem)),
        name=str(d.get("name", p.stem)),
        mean=float(d["outdoor_mean_c"]),
        amplitude=float(d["outdoor_amplitude_c"]),
        peak_hour=float(d.get("peak_hour", 15)),
        day_sd=float(d.get("day_to_day_sd_c", 0.0)),
        humidity_mean=float(d.get("humidity_mean", 65.0)),
        humidity_amplitude=float(d.get("humidity_amplitude", 10.0)),
        sunrise=float(d.get("sunrise_hour", 6.0)),
        sunset=float(d.get("sunset_hour", 19.0)),
        events=list(d.get("events", [])),
        seed=seed,
    )


def available_climates() -> List[str]:
    return sorted(p.stem for p in CLIMATE_DIR.glob("*.json"))


# ----------------------------------------------------------------------------
# Virtual farm
# ----------------------------------------------------------------------------
class VirtualFarm:
    def __init__(
        self,
        facility: Facility,
        truth: Mapping[str, object],
        climate: Climate,
        initial_state: Optional[Mapping[str, float]] = None,
        seed: int = 0,
    ) -> None:
        self.facility = facility
        self.truth = json.loads(json.dumps(truth))  # deep copy, may be mutated by events
        self.climate = climate
        self.rng = random.Random(seed)
        self.t = 0
        self.state: Dict[str, float] = {
            "water_temperature": 22.0,
            "pH": 6.15,
            "EC": 1.5,
            "flow_ratio": 1.0,
            "turbidity": 0.5,
        }
        if initial_state:
            self.state.update({k: float(v) for k, v in initial_state.items()})
        self._refresh_air()

    # ------------------------------------------------------------------
    def _eff(self, device_id: str, key: str, default: float = 0.0) -> float:
        return float(((self.truth.get("device_effects") or {}).get(device_id) or {}).get(key, default))

    def _refresh_air(self) -> None:
        out = self.climate.outdoor_temp(self.t)
        offset = float(self.truth.get("greenhouse_air_offset_c", 2.0)) * self.climate.sun(self.t)
        self.state["outdoor_temperature"] = out
        self.state["air_temperature"] = out + offset
        self.state["humidity"] = self.climate.humidity(self.t)

    def degrade_device(self, device_id: str, factor: float) -> None:
        eff = (self.truth.setdefault("device_effects", {})).setdefault(device_id, {})
        for k in list(eff.keys()):
            eff[k] = float(eff[k]) * float(factor)

    def remove_device(self, device_id: str) -> None:
        self.facility = self.facility.without(device_id)

    # ------------------------------------------------------------------
    def observe(self) -> Dict[str, float]:
        """Noisy sensor reading (what controllers see)."""
        noise = self.truth.get("sensor_noise") or {}
        obs = {}
        for k, v in self.state.items():
            sd = float(noise.get(k, 0.0))
            obs[k] = v + (self.rng.gauss(0.0, sd) if sd > 0 else 0.0)
        if self.rng.random() < float(self.truth.get("spike_prob", 0.0)):
            k = self.rng.choice(["water_temperature", "EC", "flow_ratio", "turbidity"])
            obs[k] = obs[k] * self.rng.choice([0.6, 1.4])
        for k, (lo, hi) in STATE_BOUNDS.items():
            if k in obs:
                obs[k] = min(max(obs[k], lo), hi)
        return obs

    def step(self, control: Mapping[str, float]) -> Dict[str, float]:
        """Advance one step (step_minutes) with the given device levels.

        Returns true energy (Wh) and resource use of this step.
        """
        s = self.state
        tr = self.truth
        u = {d.id: float(control.get(d.id, d.min_level)) for d in self.facility.devices}
        out = s["outdoor_temperature"]
        air = s["air_temperature"]
        sun = self.climate.sun(self.t)
        pn = tr.get("process_noise") or {}

        # water temperature
        wt = s["water_temperature"]
        dwt = float(tr.get("thermal_coupling", 0.1)) * (air - wt) + float(tr.get("solar_gain_c_per_h", 0.5)) * sun
        for d in self.facility.devices:
            dwt += self._eff(d.id, "water_temperature") * u[d.id]
            dwt += self._eff(d.id, "water_temperature_exchange") * u[d.id] * (out - wt)
        wt_next = wt + dwt + self.rng.gauss(0.0, float(pn.get("water_temperature", 0.0)))

        # flow ratio (saturating pump response, decay toward floor)
        fl = tr.get("flow") or {}
        fr = s["flow_ratio"]
        dfr = -float(fl.get("decay_per_h", 0.05)) * (fr - float(fl.get("floor", 0.4)))
        for d in self.facility.devices:
            dfr += self._eff(d.id, "flow_gain") * u[d.id] * (float(fl.get("max", 1.06)) - fr)
        fr_next = fr + dfr + self.rng.gauss(0.0, float(pn.get("flow_ratio", 0.0)))

        # EC (plant uptake in daylight, evaporation concentrates)
        ecp = tr.get("ec") or {}
        ec = s["EC"]
        dec = -float(ecp.get("uptake_per_h", 0.01)) * (0.4 + sun) + float(ecp.get("evaporation_per_c", 0.0)) * max(0.0, air - 20.0) * ec
        for d in self.facility.devices:
            dec += self._eff(d.id, "EC") * u[d.id]
        ec_next = ec + dec + self.rng.gauss(0.0, float(pn.get("EC", 0.0)))

        # turbidity (slow rise, filtered by circulation)
        tb = s["turbidity"]
        dtb = float((tr.get("turbidity") or {}).get("rise_per_h", 0.04)) * (1.0 + 0.5 * sun)
        for d in self.facility.devices:
            dtb -= self._eff(d.id, "turbidity_gain") * u[d.id] * tb
        tb_next = tb + dtb + self.rng.gauss(0.0, float(pn.get("turbidity", 0.0)))

        # pH (mean reverting random walk, dosing lowers slightly)
        php = tr.get("ph") or {}
        ph = s["pH"]
        dose = sum(u[d.id] for d in self.facility.devices if "ec_up" in d.functions)
        dph = float(php.get("reversion", 0.05)) * (float(php.get("mean", 6.15)) - ph) - 0.0005 * dose
        ph_next = ph + dph + self.rng.gauss(0.0, float(php.get("drift_sd", 0.02)))

        new = {
            "water_temperature": wt_next,
            "flow_ratio": fr_next,
            "EC": ec_next,
            "turbidity": tb_next,
            "pH": ph_next,
        }
        for k, (lo, hi) in STATE_BOUNDS.items():
            new[k] = min(max(new[k], lo), hi)
        s.update(new)

        # true energy (profile power x hidden power factor)
        pf = tr.get("power_factor") or {}
        energy_wh = sum(d.energy_wh(u[d.id]) * float(pf.get(d.id, 1.0)) for d in self.facility.devices)
        resources = self.facility.resources(u)

        self.t += 1
        self._refresh_air()
        return {"energy_wh": energy_wh, **resources}
