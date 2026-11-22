"""Learned environment dynamics:  x_{t+1} = x_t + A(d_t - x_t) + B u_t + s*sun_t + c.

One recursive-least-squares (RLS) estimator per modeled variable. Device
effects start from the priors in the facility profile and are refined from
observed transitions, so the same code adapts to each facility.

Features per variable
  bias                  constant drift (uptake, clogging, ...)
  ambient               (outdoor - x)          [water_temperature only]
  sun                   solar intensity 0..1   [water_temperature, EC, turbidity]
  <device>              u_i                    (effect model "direct")
  <device>:exchange     u_i * (outdoor - x)    (effect model "exchange")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from .facility import Facility

MODELED_VARS = ("water_temperature", "flow_ratio", "EC", "turbidity")
BOUNDS = {
    "water_temperature": (0.0, 45.0),
    "flow_ratio": (0.0, 1.5),
    "EC": (0.0, 6.0),
    "turbidity": (0.0, 30.0),
    "pH": (3.0, 9.5),
}
# measurement noise scale used to normalise the RLS covariance
NOISE_SD = {"water_temperature": 0.15, "flow_ratio": 0.015, "EC": 0.03, "turbidity": 0.06}
# 1-sigma prior uncertainty for non-device terms
BASE_PRIOR_SD = {
    "bias": {"water_temperature": 0.3, "flow_ratio": 0.05, "EC": 0.02, "turbidity": 0.05},
    "ambient": {"water_temperature": 0.1},
    "sun": {"water_temperature": 0.6, "EC": 0.02, "turbidity": 0.05},
}


@dataclass
class _VarModel:
    var: str
    features: List[Tuple[str, Optional[str]]]  # (kind, device_id)
    theta: np.ndarray
    P: np.ndarray
    names: List[str] = field(default_factory=list)


class LearnedDynamics:
    def __init__(self, facility: Facility, learn: bool = True, forgetting: float = 0.995) -> None:
        self.learn = learn
        self.lam = float(forgetting)
        self.models: Dict[str, _VarModel] = {}
        self.n_updates = 0
        self.rebuild(facility)

    # ------------------------------------------------------------------
    def rebuild(self, facility: Facility) -> None:
        """(Re)build feature sets from the facility, keeping learned values of
        devices that still exist (hardware change scenario)."""
        old = self.models
        self.facility = facility
        self.models = {}
        for var in MODELED_VARS:
            feats: List[Tuple[str, Optional[str]]] = [("bias", None)]
            prior: List[float] = [0.0]
            sd: List[float] = [BASE_PRIOR_SD["bias"].get(var, 0.05)]
            if var == "water_temperature":
                feats.append(("ambient", None))
                prior.append(float(facility.prior_thermal_coupling))
                sd.append(BASE_PRIOR_SD["ambient"]["water_temperature"])
            if var in BASE_PRIOR_SD["sun"]:
                feats.append(("sun", None))
                prior.append(0.0)
                sd.append(BASE_PRIOR_SD["sun"][var])
            for d in facility.devices:
                for e in d.effects:
                    if e.var != var:
                        continue
                    kind = "exchange" if e.model == "exchange" else "direct"
                    feats.append((kind, d.id))
                    prior.append(e.prior)
                    sd.append(max(abs(e.prior) * 0.8, 1e-4))
            theta = np.array(prior, dtype=float)
            P = np.diag((np.array(sd) / NOISE_SD[var]) ** 2)
            names = [k if dev is None else (dev if k == "direct" else f"{dev}:exchange") for k, dev in feats]
            # carry over learned parameters
            if var in old:
                om = old[var]
                for i, nm in enumerate(names):
                    if nm in om.names:
                        j = om.names.index(nm)
                        theta[i] = om.theta[j]
                        P[i, i] = om.P[j, j]
            self.models[var] = _VarModel(var, feats, theta, P, names)

    # ------------------------------------------------------------------
    @staticmethod
    def _phi(vm: _VarModel, x: Mapping[str, float], u: Mapping[str, float], outdoor: float, sun: float) -> np.ndarray:
        xv = float(x.get(vm.var, 0.0))
        out = []
        for kind, dev in vm.features:
            if kind == "bias":
                out.append(1.0)
            elif kind == "ambient":
                out.append(outdoor - xv)
            elif kind == "sun":
                out.append(sun)
            elif kind == "direct":
                out.append(float(u.get(dev, 0.0)))
            elif kind == "exchange":
                out.append(float(u.get(dev, 0.0)) * (outdoor - xv))
        return np.array(out, dtype=float)

    def predict(self, x: Mapping[str, float], u: Mapping[str, float], outdoor: float, sun: float) -> Dict[str, float]:
        nxt = dict(x)
        for var, vm in self.models.items():
            v = float(x.get(var, 0.0)) + float(vm.theta @ self._phi(vm, x, u, outdoor, sun))
            lo, hi = BOUNDS[var]
            nxt[var] = min(max(v, lo), hi)
        return nxt

    def update(self, x: Mapping[str, float], u: Mapping[str, float], outdoor: float, sun: float, x_next: Mapping[str, float]) -> None:
        if not self.learn:
            return
        for var, vm in self.models.items():
            phi = self._phi(vm, x, u, outdoor, sun)
            y = float(x_next.get(var, 0.0)) - float(x.get(var, 0.0))
            err = y - float(vm.theta @ phi)
            # robust: ignore implausible jumps (sensor spikes)
            if abs(err) > 6.0 * NOISE_SD[var] + 0.5 * abs(y):
                continue
            Pphi = vm.P @ phi
            denom = self.lam + float(phi @ Pphi)
            k = Pphi / denom
            vm.theta = vm.theta + k * err
            vm.P = (vm.P - np.outer(k, Pphi)) / self.lam
        self.n_updates += 1

    # ------------------------------------------------------------------
    def coefficients(self) -> Dict[str, Dict[str, float]]:
        return {var: {nm: round(float(th), 5) for nm, th in zip(vm.names, vm.theta)} for var, vm in self.models.items()}

    def device_effect(self, var: str, device_id: str) -> Optional[float]:
        vm = self.models.get(var)
        if vm is None:
            return None
        for nm, th in zip(vm.names, vm.theta):
            if nm in (device_id, f"{device_id}:exchange"):
                return float(th)
        return None
