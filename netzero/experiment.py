"""Scenario runner and metrics.

Example
-------
python -m netzero.experiment --facility small_rack mid_greenhouse leaky_greenhouse \
    --crop basil --climate summer_south --days 7
python -m netzero.experiment --facility mid_greenhouse --event degrade:chiller_a:0.5@3
python -m netzero.experiment --facility small_rack --event remove:vent_fan@2
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from .controllers import CONTROLLER_DESCRIPTIONS, CONTROLLER_LABELS, DEFAULT_CONTROLLERS, Context, OptimizerWeights, make_controller, resolve_controller
from .facility import available_facilities, load_facility, simulator_truth
from .fuzzy import get_crop_model
from .grid import load_grid
from .simulator import VirtualFarm, available_climates, load_climate

RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class Event:
    day: int
    type: str  # "degrade" | "remove"
    device: str
    factor: float = 1.0

    @classmethod
    def parse(cls, s: str) -> "Event":
        """'degrade:chiller_a:0.5@3' or 'remove:vent_fan@2'"""
        body, _, day = s.partition("@")
        parts = body.split(":")
        if parts[0] == "degrade":
            return cls(int(day or 0), "degrade", parts[1], float(parts[2]))
        if parts[0] == "remove":
            return cls(int(day or 0), "remove", parts[1])
        raise ValueError(f"unknown event: {s}")


@dataclass
class EpisodeResult:
    facility: str
    controller: str
    controller_name: str
    crop: str
    climate: str
    days: int
    seed: int
    metrics: Dict[str, Optional[float]]
    series: Dict[str, List[float]] = field(default_factory=dict)
    learned_coefficients: Optional[Dict[str, Dict[str, float]]] = None
    runtime_sec: float = 0.0

    def to_dict(self, with_series: bool = True) -> Dict[str, object]:
        d = {
            "facility": self.facility,
            "controller": self.controller,
            "controller_name": self.controller_name,
            "crop": self.crop,
            "climate": self.climate,
            "days": self.days,
            "seed": self.seed,
            "metrics": self.metrics,
            "learned_coefficients": self.learned_coefficients,
            "runtime_sec": round(self.runtime_sec, 2),
        }
        if with_series:
            d["series"] = self.series
        return d


def _adaptation_hours(errors: Sequence[float], threshold: float, window: int = 24) -> Optional[int]:
    """First hour after which the rolling mean error stays below threshold for a full window."""
    n = len(errors)
    if n < 2 * window:
        return None
    roll = [sum(errors[i - window:i]) / window for i in range(window, n + 1)]
    for i, r in enumerate(roll):
        if r <= threshold and all(x <= threshold for x in roll[i:i + window]):
            return i + window
    return None


def run_episode(
    facility_name: str,
    controller_label: str,
    *,
    crop: str = "basil",
    climate: str = "summer_south",
    grid: str = "kr_stylized",
    days: int = 7,
    seed: int = 0,
    events: Sequence[Event] = (),
    weights: Optional[OptimizerWeights] = None,
    initial_state: Optional[Mapping[str, float]] = None,
) -> EpisodeResult:
    t0 = time.time()
    facility = load_facility(facility_name)
    truth = simulator_truth(facility_name)
    crop_model = get_crop_model(crop)
    clim = load_climate(climate, seed=seed)
    grid_p = load_grid(grid)
    init = {
        "water_temperature": crop_model.targets["water_temperature"],
        "EC": crop_model.targets["EC"],
        "pH": crop_model.targets["pH"],
        "flow_ratio": 1.0,
        "turbidity": 0.5,
    }
    if initial_state:
        init.update(initial_state)
    farm = VirtualFarm(facility, truth, clim, initial_state=init, seed=seed)
    ctrl = make_controller(controller_label, facility, crop_model, weights)

    steps = days * 24
    s: Dict[str, List[float]] = {k: [] for k in [
        "hour", "outdoor", "water_temperature", "EC", "flow_ratio", "turbidity", "pH",
        "risk", "energy_wh", "carbon_kg", "cost_krw", "ef", "pred_error_wt",
    ]}
    s["controls"] = []  # type: ignore[assignment]
    in_range = alert = danger = 0
    energy_wh = carbon = cost = nutrient = 0.0
    errors: List[float] = []
    pending = sorted(events, key=lambda e: e.day)

    for t in range(steps):
        while pending and t >= pending[0].day * 24:
            ev = pending.pop(0)
            if ev.type == "degrade":
                farm.degrade_device(ev.device, ev.factor)
            elif ev.type == "remove":
                farm.remove_device(ev.device)
                ctrl.set_facility(farm.facility)

        ctx = Context(t=t, climate=clim, grid=grid_p)
        obs = farm.observe()
        u, info = ctrl.decide(obs, ctx)
        used = farm.step(u)
        true_next = farm.state
        obs_next = farm.observe()
        ctrl.observe(obs, u, ctx, obs_next)

        pred = info.get("pred_next") if isinstance(info, dict) else None
        if isinstance(pred, dict):
            err = abs(float(pred["water_temperature"]) - true_next["water_temperature"])
            errors.append(err)
            s["pred_error_wt"].append(round(err, 4))

        ev_true = crop_model.evaluate(true_next)
        in_range += int(crop_model.in_optimal_range(ev_true))
        alert += int(ev_true.is_alert)
        danger += int(bool(crop_model.danger_vars(ev_true)))

        hour = t % 24
        e_wh = float(used["energy_wh"])
        energy_wh += e_wh
        c_kg = e_wh / 1000.0 * grid_p.ef(hour)
        carbon += c_kg
        k_krw = e_wh / 1000.0 * grid_p.price_at(hour)
        cost += k_krw
        nutrient += float(used.get("nutrient_ml", 0.0))

        s["hour"].append(t)
        s["outdoor"].append(round(clim.outdoor_temp(t), 3))
        for k in ["water_temperature", "EC", "flow_ratio", "turbidity", "pH"]:
            s[k].append(round(true_next[k], 4))
        s["risk"].append(round(ev_true.risk_score, 3))
        s["energy_wh"].append(round(e_wh, 3))
        s["carbon_kg"].append(round(c_kg, 5))
        s["cost_krw"].append(round(k_krw, 3))
        s["ef"].append(grid_p.ef(hour))
        s["controls"].append({k: float(v) for k, v in u.items()})  # type: ignore[union-attr]

    n = float(steps)
    metrics: Dict[str, Optional[float]] = {
        "in_range_rate": round(in_range / n, 4),
        "alert_rate": round(alert / n, 4),
        "danger_hours": float(danger),
        "mean_risk": round(sum(s["risk"]) / n, 3),
        "energy_kwh": round(energy_wh / 1000.0, 4),
        "carbon_kg": round(carbon, 4),
        "cost_krw": round(cost, 1),
        "carbon_intensity_kg_per_kwh": round(carbon / (energy_wh / 1000.0), 4) if energy_wh > 0 else None,
        "nutrient_ml": round(nutrient, 1),
        "pred_mae_wt_first_day": round(sum(errors[:24]) / max(len(errors[:24]), 1), 4) if errors else None,
        "pred_mae_wt_last_day": round(sum(errors[-24:]) / max(len(errors[-24:]), 1), 4) if errors else None,
        "adaptation_hours": (lambda h: float(h) if h is not None else None)(_adaptation_hours(errors, 0.25)) if errors else None,
    }
    return EpisodeResult(
        facility=facility_name,
        controller=ctrl.label,
        controller_name=CONTROLLER_LABELS.get(ctrl.label, ctrl.name),
        crop=crop,
        climate=climate,
        days=days,
        seed=seed,
        metrics=metrics,
        series=s,  # type: ignore[arg-type]
        learned_coefficients=ctrl.model.coefficients() if ctrl.model is not None and ctrl.model.learn else None,
        runtime_sec=time.time() - t0,
    )


def compare(
    facilities: Sequence[str],
    controllers: Sequence[str] = DEFAULT_CONTROLLERS,
    **kwargs,
) -> List[EpisodeResult]:
    out = []
    for f in facilities:
        for c in controllers:
            out.append(run_episode(f, c, **kwargs))
    return out


def summary_rows(results: Sequence[EpisodeResult]) -> List[Dict[str, object]]:
    base = {r.facility: r for r in results if r.controller == "A"}
    rows = []
    for r in results:
        m = r.metrics
        b = base.get(r.facility)
        saving = None
        carbon_red = None
        if b is not None and b.metrics["energy_kwh"]:
            saving = round(100.0 * (b.metrics["energy_kwh"] - m["energy_kwh"]) / b.metrics["energy_kwh"], 1)
            carbon_red = round(100.0 * (b.metrics["carbon_kg"] - m["carbon_kg"]) / b.metrics["carbon_kg"], 1)
        rows.append({
            "facility": r.facility,
            "controller": CONTROLLER_LABELS.get(r.controller, r.controller),
            "in_range_%": round(100 * m["in_range_rate"], 1),
            "alert_%": round(100 * m["alert_rate"], 1),
            "danger_h": m["danger_hours"],
            "energy_kWh": m["energy_kwh"],
            "energy_saving_%": saving,
            "carbon_kg": m["carbon_kg"],
            "carbon_reduction_%": carbon_red,
            "kgCO2/kWh": m["carbon_intensity_kg_per_kwh"],
            "cost_KRW": m["cost_krw"],
            "pred_mae_first→last": (
                f"{m['pred_mae_wt_first_day']:.2f}→{m['pred_mae_wt_last_day']:.2f}" if m["pred_mae_wt_first_day"] is not None else "-"
            ),
            "adapt_h": m["adaptation_hours"],
        })
    return rows


def _print_table(rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="가상 스마트팜 운전 방식 비교 (흔한 방식 / 학습형 / 탄소 반영)")
    ap.add_argument("--facility", nargs="+", default=available_facilities(), choices=available_facilities())
    ap.add_argument("--controllers", nargs="+", default=list(DEFAULT_CONTROLLERS),
                    help="흔한방식 학습형 탄소반영 (또는 A C D). 참고용: 경고시에만계산 (B)")
    ap.add_argument("--crop", default="basil")
    ap.add_argument("--climate", default="summer_south", choices=available_climates())
    ap.add_argument("--grid", default="kr_stylized")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--event", action="append", default=[], help="degrade:<device>:<factor>@<day> | remove:<device>@<day>")
    ap.add_argument("--out", default=None, help="result JSON path (default: netzero/results/<timestamp>.json)")
    ap.add_argument("--no-series", action="store_true", help="omit hourly series from the JSON output")
    args = ap.parse_args(argv)

    events = [Event.parse(e) for e in args.event]
    try:
        codes = [resolve_controller(c) for c in args.controllers]
    except ValueError as e:
        ap.error(str(e))
    if "A" not in codes:
        codes.insert(0, "A")  # savings are measured against the baseline
    results = compare(
        args.facility,
        codes,
        crop=args.crop,
        climate=args.climate,
        grid=args.grid,
        days=args.days,
        seed=args.seed,
        events=events,
    )
    rows = summary_rows(results)
    print(f"crop={args.crop} climate={args.climate} days={args.days} seed={args.seed} events={args.event or '-'}")
    for k in codes:
        print(f"  {CONTROLLER_LABELS[k]}: {CONTROLLER_DESCRIPTIONS[k]}")
    print("  절감·감축률은 흔한 방식 대비")
    print()
    _print_table(rows)

    out = Path(args.out) if args.out else RESULTS_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": vars(args),
        "summary": rows,
        "episodes": [r.to_dict(with_series=not args.no_series) for r in results],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
