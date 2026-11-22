"""Controllers compared in the experiments.

A  ThresholdController      fixed thresholds + timers (common practice baseline)
B  FuzzyOptimizer(legacy)   original behaviour: optimise only on 경고+, fixed priors, flat Wh cost
C  FuzzyOptimizer(learned)  optimise every step, learned dynamics, 3-step lookahead
D  FuzzyOptimizer(carbon)   C + hourly carbon intensity and price in the energy term
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

from .facility import Control, Facility, FUNCTIONS
from .fuzzy import CropModel
from .grid import GridProfile
from .model import LearnedDynamics
from .simulator import Climate


@dataclass
class Context:
    t: int
    climate: Climate
    grid: GridProfile


class Controller:
    label = "?"
    name = "controller"
    model: Optional[LearnedDynamics] = None

    def decide(self, obs: Mapping[str, float], ctx: Context) -> Tuple[Control, Dict[str, object]]:
        raise NotImplementedError

    def observe(self, obs: Mapping[str, float], control: Control, ctx: Context, obs_next: Mapping[str, float]) -> None:
        if self.model is not None:
            self.model.update(obs, control, ctx.climate.outdoor_temp(ctx.t), ctx.climate.sun(ctx.t), obs_next)

    def set_facility(self, facility: Facility) -> None:
        self.facility = facility
        if self.model is not None:
            self.model.rebuild(facility)


# ----------------------------------------------------------------------------
class ThresholdController(Controller):
    """Deadband thresholds around crop targets plus fixed circulation timer."""

    label = "A"
    name = "고정 임계값"

    def __init__(self, facility: Facility, crop: CropModel, pump_timer_fraction: float = 0.67) -> None:
        self.facility = facility
        self.crop = crop
        self.pump_timer_fraction = pump_timer_fraction
        self.model = None

    def decide(self, obs: Mapping[str, float], ctx: Context) -> Tuple[Control, Dict[str, object]]:
        tg = self.crop.targets
        wt, fr, ec = obs["water_temperature"], obs["flow_ratio"], obs["EC"]
        outdoor = ctx.climate.outdoor_temp(ctx.t)
        u = self.facility.idle_control()
        for d in self.facility.devices:
            top = max(d.levels)
            mid = d.levels[len(d.levels) // 2]
            if "water_temp_down" in d.functions and d.type != "ventilation_fan":
                if wt > tg["water_temperature"] + 1.0:
                    u[d.id] = top
                elif wt > tg["water_temperature"] + 0.3:
                    u[d.id] = mid
            elif d.type == "ventilation_fan":
                if outdoor < wt - 1.0 and wt > tg["water_temperature"] - 0.5:
                    u[d.id] = top
            if "flow_up" in d.functions:
                timer = min(d.levels, key=lambda x: abs(x - top * self.pump_timer_fraction))
                u[d.id] = top if (fr < 0.95 or obs["turbidity"] > 0.7) else timer
            if "ec_up" in d.functions and ec < tg["EC"] - 0.1:
                u[d.id] = mid
        return u, {"mode": "threshold"}


# ----------------------------------------------------------------------------
@dataclass
class OptimizerWeights:
    risk: float = 1.0
    energy: float = 6.0
    resource: float = 2.0
    change: float = 0.5
    over_control: float = 6.0
    comfort: float = 4.0
    range_margin: float = 0.75
    range_tolerance: float = 0.1
    energy_normalizer_kwh: float = 0.25
    resource_normalizer_ml: float = 40.0
    w_carbon: float = 0.7
    w_price: float = 0.3
    infeasible_penalty: float = 1000.0


class FuzzyOptimizer(Controller):
    def __init__(
        self,
        facility: Facility,
        crop: CropModel,
        *,
        label: str,
        name: str,
        always: bool,
        learn: bool,
        carbon: bool,
        horizon: int,
        weights: Optional[OptimizerWeights] = None,
    ) -> None:
        self.facility = facility
        self.crop = crop
        self.label = label
        self.name = name
        self.always = always
        self.carbon = carbon
        self.horizon = max(1, int(horizon))
        self.w = weights or OptimizerWeights()
        self.model = LearnedDynamics(facility, learn=learn)
        self.prev: Control = facility.idle_control()

    # ------------------------------------------------------------------
    def needed_functions(self, obs: Mapping[str, float]) -> List[str]:
        """Which functions could help now. Devices outside these stay at minimum."""
        rng = self.crop.optimal_ranges
        tg = self.crop.targets
        need = []
        wt_lo, wt_hi = rng.get("water_temperature", (tg["water_temperature"] - 2, tg["water_temperature"] + 2))
        if obs["water_temperature"] > wt_lo:
            need.append("water_temp_down")
        if obs["water_temperature"] < wt_hi:
            need.append("water_temp_up")
        if obs["flow_ratio"] < rng.get("flow_ratio", (0.9, 1.05))[1]:
            need.append("flow_up")
        if obs["EC"] < tg["EC"]:
            need.append("ec_up")
        if obs["EC"] > tg["EC"]:
            need.append("ec_down")
        if obs["turbidity"] > 0.3:
            need.append("turbidity_down")
        return need

    def _energy_term(self, kwh: float, hour: int, grid: GridProfile) -> float:
        if self.carbon:
            kwh = kwh * grid.weighted_cost_factor(hour, self.w.w_carbon, self.w.w_price)
        return kwh / self.w.energy_normalizer_kwh

    def _over_control(self, control: Control, pred: Mapping[str, float]) -> float:
        pen = 0.0
        for d in self.facility.devices:
            if control.get(d.id, 0.0) <= d.min_level:
                continue
            for var, sign in d.directions().items():
                lo, hi = self.crop.optimal_ranges.get(var, (None, None))
                if lo is None:
                    continue
                width = max(hi - lo, 1e-3)
                x = float(pred.get(var, 0.0))
                if sign < 0 and x < lo:
                    pen += (lo - x) / width
                elif sign > 0 and x > hi:
                    pen += (x - hi) / width
        return pen

    def _discomfort(self, ev) -> float:
        """Continuous distance from the stable sets: sum of (1 - mu_stable).

        Mamdani centroid output is flat inside each label region, so this term
        gives the optimizer a gradient toward the optimal range."""
        tot = 0.0
        for var in ("water_temperature", "flow_ratio", "EC", "turbidity"):
            lbl = self.crop.stable_label(var)
            if lbl is not None:
                tot += 1.0 - float(ev.memberships.get(var, {}).get(lbl, 0.0))
        return tot

    def _violation(self, ev) -> float:
        """Predicted range violation used as the first-priority criterion.

        A variable counts as satisfied when its stable-set membership is at
        least `range_margin` (above the 0.5 used for scoring, to absorb
        prediction error)."""
        v = 0.0
        for var in ("water_temperature", "flow_ratio", "EC", "turbidity"):
            lbl = self.crop.stable_label(var)
            if lbl is None:
                continue
            mu = float(ev.memberships.get(var, {}).get(lbl, 0.0))
            v += max(0.0, self.w.range_margin - mu)
        return round(v, 3)

    def _change(self, control: Control) -> float:
        tot = 0.0
        for d in self.facility.devices:
            span = max(max(d.levels) - d.min_level, 1e-6)
            tot += abs(control.get(d.id, 0.0) - self.prev.get(d.id, d.min_level)) / span
        return tot / max(len(self.facility.devices), 1)

    # ------------------------------------------------------------------
    def decide(self, obs: Mapping[str, float], ctx: Context) -> Tuple[Control, Dict[str, object]]:
        ev_now = self.crop.evaluate(obs)
        if not self.always and not ev_now.is_alert:
            u = self.facility.idle_control()
            self.prev = u
            return u, {"mode": "monitor", "risk_now": ev_now.risk_score, "pred_next": self.model.predict(obs, u, ctx.climate.outdoor_temp(ctx.t), ctx.climate.sun(ctx.t))}

        t, clim, grid = ctx.t, ctx.climate, ctx.grid
        idle = self.facility.idle_control()
        candidates = self.facility.candidates(self.needed_functions(obs))
        scored = []
        for c in candidates:
            x1 = self.model.predict(obs, c, clim.outdoor_temp(t), clim.sun(t))
            ev1 = self.crop.evaluate(x1)
            risks = [ev1.risk_score]
            infeasible = bool(self.crop.danger_vars(ev1))
            violation = self._violation(ev1)
            # lookahead: assume this control is held for the next steps
            # (constant-input rollout), so pre-cooling before a hot hour pays off
            x = x1
            for k in range(1, self.horizon):
                x = self.model.predict(x, c, clim.outdoor_temp(t + k), clim.sun(t + k))
                evk = self.crop.evaluate(x)
                risks.append(evk.risk_score)
                violation += 0.5 * self._violation(evk)
            weights = [1.0] + [0.5] * (len(risks) - 1)
            risk = sum(w * r for w, r in zip(weights, risks)) / sum(weights)
            comfort = self._discomfort(ev1)
            kwh = self.facility.energy_wh(c) / 1000.0
            nutrient = self.facility.resources(c).get("nutrient_ml", 0.0)
            J = (
                self.w.risk * risk
                + self.w.energy * self._energy_term(kwh, t, grid)
                + self.w.resource * nutrient / self.w.resource_normalizer_ml
                + self.w.change * self._change(c)
                + self.w.over_control * self._over_control(c, x1)
                + self.w.comfort * comfort
                + (self.w.infeasible_penalty if infeasible else 0.0)
            )
            scored.append((round(violation, 3), J, c, x1, ev1))
        # 1) keep the crop in its optimal range first (lexicographic),
        # 2) then minimise energy/carbon, risk and wear among those candidates
        v_min = min(v for v, *_ in scored)
        feasible = [row for row in scored if row[0] <= v_min + self.w.range_tolerance + 1e-9]
        _, J, u, x1, ev1 = min(feasible, key=lambda row: row[1])
        self.prev = dict(u)
        return dict(u), {
            "mode": "optimize",
            "risk_now": ev_now.risk_score,
            "pred_next": x1,
            "pred_risk": ev1.risk_score,
            "objective": J,
            "n_candidates": len(candidates),
        }


# ----------------------------------------------------------------------------
CONTROLLER_LABELS = {
    "A": "고정 임계값 제어",
    "B": "기존 시스템 (경고 시에만 최적화, 고정 계수)",
    "C": "상시 최적화 + 학습형 예측",
    "D": "C + 탄소·요금 반영",
}


def make_controller(label: str, facility: Facility, crop: CropModel, weights: Optional[OptimizerWeights] = None) -> Controller:
    label = label.upper()
    if label == "A":
        return ThresholdController(facility, crop)
    if label == "B":
        return FuzzyOptimizer(facility, crop, label="B", name=CONTROLLER_LABELS["B"], always=False, learn=False, carbon=False, horizon=1, weights=weights)
    if label == "C":
        return FuzzyOptimizer(facility, crop, label="C", name=CONTROLLER_LABELS["C"], always=True, learn=True, carbon=False, horizon=3, weights=weights)
    if label == "D":
        return FuzzyOptimizer(facility, crop, label="D", name=CONTROLLER_LABELS["D"], always=True, learn=True, carbon=True, horizon=3, weights=weights)
    raise ValueError(f"unknown controller {label}")
