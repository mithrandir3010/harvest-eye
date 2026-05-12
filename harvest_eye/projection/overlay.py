"""Görsel overlay — mantarlar üzerine nokta çizimi."""
from __future__ import annotations
import cv2
import numpy as np

from harvest_eye.config import ProjectionConfig
from harvest_eye.maturity.shape_classifier import ClassificationResult, HarvestDecision


def draw_harvest_overlay(
    frame: np.ndarray,
    results: list[ClassificationResult],
    cfg: ProjectionConfig | None = None,
    frame_idx: int = 0,
) -> np.ndarray:
    """
    Tespit sonuçlarını görüntü üzerine çiz.

    Hasat edilecek → yeşil daire
    Bekleyecek     → kırmızı daire

    Returns:
        Başlık şeridi eklenmiş BGR görüntü
    """
    cfg = cfg or ProjectionConfig()
    out = frame.copy()
    h_img, w_img = out.shape[:2]

    harvest_count = sum(1 for r in results if r.decision == HarvestDecision.HARVEST)
    wait_count    = sum(1 for r in results if r.decision == HarvestDecision.WAIT)

    for res in results:
        cx = res.detection.cx_px
        cy = res.detection.cy_px
        color = cfg.green_color if res.decision == HarvestDecision.HARVEST else cfg.red_color
        r = cfg.dot_radius
        cv2.circle(out, (cx, cy), r, color, -1)
        cv2.circle(out, (cx, cy), r, (255, 255, 255), 1)

    # Başlık şeridi
    bar = np.zeros((48, w_img, 3), dtype=np.uint8)
    prefix = f"Kare: {frame_idx} | " if frame_idx > 0 else ""
    title = (
        f"{prefix}Toplam: {len(results)} | "
        f"Hasat: {harvest_count} (yesil) | "
        f"Bekle: {wait_count} (kirmizi)"
    )
    cv2.putText(bar, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return np.vstack([bar, out])
