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
    # 꼭짓점을 경계 검사보다 먼저 본다: a == b(왼쪽 숄더)나 b == c(오른쪽 숄더)일 때
    # x == b 에서 0이 나오는 문제 방지
    if x == b:
        return 1.0
    if x <= a or x >= c:
        return 0.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def trap(a: float, b: float, c: float, d: float, x: float) -> float:
    """Trapezoidal membership."""
    if x is None:
        return 0.0
    # 평탄 구간을 경계 검사보다 먼저 본다: a == b(예: pH 강산성 (0,0,4.5,5.5),
    # 유량 정체 (0,0,0.35,0.45))나 c == d(습도 (88,95,100,100))일 때 끝점 값
    # (유량 0.0, pH 0.0, 습도 100)이 0이 되어 아무 규칙도 발화하지 않던 버그 수정
    if b <= x <= c:
        return 1.0
    if x <= a or x >= d:
        return 0.0
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
