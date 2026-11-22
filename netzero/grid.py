"""Hourly grid carbon intensity and electricity price."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

GRID_DIR = Path(__file__).resolve().parent / "profiles" / "grid"


@dataclass
class GridProfile:
    grid_id: str
    name: str
    emission_factor: List[float]  # kgCO2e per kWh, 24 hourly values
    price: List[float]  # KRW per kWh, 24 hourly values

    def ef(self, hour: int) -> float:
        return float(self.emission_factor[int(hour) % 24])

    def price_at(self, hour: int) -> float:
        return float(self.price[int(hour) % 24])

    @property
    def ef_mean(self) -> float:
        return sum(self.emission_factor) / len(self.emission_factor)

    @property
    def price_mean(self) -> float:
        return sum(self.price) / len(self.price)

    def weighted_cost_factor(self, hour: int, w_carbon: float = 0.5, w_price: float = 0.5) -> float:
        """Relative cost of 1 kWh at this hour (1.0 = daily average)."""
        return w_carbon * self.ef(hour) / self.ef_mean + w_price * self.price_at(hour) / self.price_mean


def load_grid(name_or_path: str | Path = "kr_stylized") -> GridProfile:
    p = Path(name_or_path)
    if not p.suffix:
        p = GRID_DIR / f"{name_or_path}.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    ef = [float(x) for x in d["emission_factor_kg_per_kwh"]]
    price = [float(x) for x in d["price_krw_per_kwh"]]
    if len(ef) != 24 or len(price) != 24:
        raise ValueError("grid profile needs 24 hourly values")
    return GridProfile(str(d.get("grid_id", p.stem)), str(d.get("name", p.stem)), ef, price)
