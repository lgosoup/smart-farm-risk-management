from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def _load_image(path: str | Path) -> np.ndarray:
    """BGR 이미지 로드 후 RGB로 반환."""
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"이미지 로드 실패: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def compute_color_health(img: np.ndarray) -> float:
    """
    식물 색상 건강도를 [0, 1]로 반환.

    ExG(Excess Green Index) = 2G - R - B, 픽셀별 정규화([-2,2] → [0,1])
    HSV 채도(Saturation) 평균 (0~255 → [0,1])
    최종: 0.6 * exg_score + 0.4 * sat_score
    """
    img_f = img.astype(np.float32)
    r, g, b = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]

    exg = 2.0 * g - r - b  # range [-510, 510] per channel 255
    # normalize to [0, 1]: shift by +510 then divide by 1020
    exg_norm = np.clip((exg + 510.0) / 1020.0, 0.0, 1.0)
    exg_score = float(exg_norm.mean())

    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sat_score = float(img_hsv[:, :, 1].mean()) / 255.0

    return float(np.clip(0.6 * exg_score + 0.4 * sat_score, 0.0, 1.0))


def _date_from_path(path: Path) -> Optional[date]:
    """파일명에서 YYYYMMDD 추출 → date. 없으면 None."""
    m = re.search(r"(\d{8})", path.stem)
    if not m:
        return None
    try:
        return date(int(m.group(1)[:4]), int(m.group(1)[4:6]), int(m.group(1)[6:8]))
    except ValueError:
        return None


def _find_image_for_date(capture_dir: Path, target_date: date) -> Optional[Path]:
    """capture_dir에서 target_date(YYYYMMDD) 이름의 이미지 파일을 찾아 반환."""
    date_str = target_date.strftime("%Y%m%d")
    for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG"):
        p = capture_dir / f"{date_str}.{ext}"
        if p.exists():
            return p
    return None


def compute_temporal_score(
    today_health: float,
    capture_dir: Path,
    today_date: date,
) -> float:
    """
    전날(1d), 7일 전, 30일 전 이미지와 색상 점수 비교 후 가중 평균 반환 [0, 1].

    비교 이미지가 없는 구간은 today_health로 채움(중립적 처리).
    가중치: 1d=0.5, 7d=0.3, 30d=0.2
    """
    offsets = [(1, 0.5), (7, 0.3), (30, 0.2)]
    weighted_sum = 0.0
    weight_total = 0.0

    for days_ago, weight in offsets:
        past_date = today_date - timedelta(days=days_ago)
        past_path = _find_image_for_date(capture_dir, past_date)
        if past_path is not None:
            try:
                past_img = _load_image(past_path)
                past_health = compute_color_health(past_img)
                # 오늘 - 과거: 개선이면 양수, 악화면 음수; 0.5 기준으로 정규화
                delta = today_health - past_health
                score = float(np.clip(0.5 + delta, 0.0, 1.0))
            except Exception:
                score = today_health
        else:
            score = today_health

        weighted_sum += weight * score
        weight_total += weight

    if weight_total == 0.0:
        return today_health
    return float(np.clip(weighted_sum / weight_total, 0.0, 1.0))


def compute_plant_health_score(
    today_path: str | Path,
    capture_dir: str | Path,
) -> Tuple[float, Dict]:
    """
    오늘 이미지로부터 식물 건강 복합 점수를 계산.

    composite = 0.5 * color_health + 0.5 * temporal_score
    반환: (composite [0, 1], 상세 dict)
    """
    today_path = Path(today_path)
    capture_dir = Path(capture_dir)

    img = _load_image(today_path)
    color_health = compute_color_health(img)

    today_date = _date_from_path(today_path) or date.today()
    temporal = compute_temporal_score(color_health, capture_dir, today_date)

    composite = float(np.clip(0.5 * color_health + 0.5 * temporal, 0.0, 1.0))

    return composite, {
        "color_health": round(color_health, 4),
        "temporal_score": round(temporal, 4),
        "composite": round(composite, 4),
        "image_date": today_date.isoformat(),
        "image_path": str(today_path),
    }


def compute_vision_reward(
    capture_dir: str | Path,
) -> Optional[Tuple[float, Dict]]:
    """
    오늘 날짜(YYYYMMDD.jpg/png) 이미지를 찾아 보상값 계산.

    이미지가 없으면 None 반환 → 호출자가 휴리스틱으로 fallback.
    reward = composite * 1.2 - 0.1   → [0,1] → [-0.1, 1.1]
    (기존 휴리스틱 범위와 동일)
    """
    capture_dir = Path(capture_dir)
    today = date.today()
    today_path = _find_image_for_date(capture_dir, today)

    if today_path is None:
        return None

    try:
        composite, details = compute_plant_health_score(today_path, capture_dir)
    except Exception:
        return None

    reward = float(np.clip(composite * 1.2 - 0.1, -0.1, 1.1))
    details["reward"] = round(reward, 4)
    return reward, {"source": "vision", **details}
