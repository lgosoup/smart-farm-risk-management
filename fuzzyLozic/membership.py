"""
Membership function utilities.

All shapes are configured in ``config.py``; this module only performs
evaluation using tri/ trap definitions.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


def tri(a: float, b: float, c: float, x: float) -> float:
    """Triangular membership."""
    if x is None:
        return 0.0
    if a == b and b == c:
        return 1.0 if x == a else 0.0
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def trap(a: float, b: float, c: float, d: float, x: float) -> float:
    """Trapezoidal membership."""
    if x is None:
        return 0.0
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def evaluate(shape: str, points: Tuple[float, ...], x: float) -> float:
    """Dispatch membership evaluation based on shape name."""
    if shape == "tri":
        if len(points) != 3:
            raise ValueError("Triangular membership requires 3 points")
        return tri(points[0], points[1], points[2], x)
    if shape == "trap":
        if len(points) != 4:
            raise ValueError("Trapezoidal membership requires 4 points")
        return trap(points[0], points[1], points[2], points[3], x)
    raise ValueError(f"Unsupported shape: {shape}")


def select_top_memberships(memberships: Dict[str, float], top_n: int = 3) -> Dict[str, float]:
    """Return the top membership labels sorted by degree (descending)."""
    sorted_items = sorted(memberships.items(), key=lambda kv: kv[1], reverse=True)
    return dict(sorted_items[:top_n])


def sample_universe(start: float, end: float, step: float) -> List[float]:
    """Generate a list of points across the output universe."""
    points: List[float] = []
    current = start
    while current <= end + 1e-9:
        points.append(round(current, 10))
        current += step
    return points
