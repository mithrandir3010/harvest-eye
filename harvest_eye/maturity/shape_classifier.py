"""Şekil tabanlı hasat kararı modülü."""
from __future__ import annotations
import math
import logging
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from harvest_eye.config import HarvestConfig
from harvest_eye.detection.detector import Detection

logger = logging.getLogger(__name__)


class HarvestDecision(Enum):
    HARVEST = "harvest"   # yeşil — topla
    WAIT    = "wait"      # kırmızı — bekle
    UNKNOWN = "unknown"   # gri — belirsiz


@dataclass
class ClassificationResult:
    detection: Detection
    decision: HarvestDecision
    circularity: float
    aspect_ratio: float
    confidence: float  # karar güveni (0-1)


class ShapeClassifier:
    """
    Bounding box şekil analizine dayalı hasat sınıflandırıcı.

    Dairesellik ve en-boy oranına göre karar verir.
    Derin öğrenme modeli hazır olmadan önce kullanılır.
    """

    def __init__(self, cfg: HarvestConfig | None = None):
        self.cfg = cfg or HarvestConfig()

    def classify(
        self,
        frame: np.ndarray,
        detection: Detection,
    ) -> ClassificationResult:
        """Tek bir tespiti sınıflandır."""
        circularity = self._compute_circularity(frame, detection)
        aspect_ratio = (
            detection.w_px / detection.h_px
            if detection.h_px > 0
            else 1.0
        )

        is_harvest_circ = circularity < self.cfg.circularity_thresh
        is_harvest_ar   = aspect_ratio > self.cfg.aspect_thresh

        if is_harvest_circ or is_harvest_ar:
            decision = HarvestDecision.HARVEST
            conf = max(
                1.0 - circularity / self.cfg.circularity_thresh if is_harvest_circ else 0.0,
                (aspect_ratio - 1.0) / self.cfg.aspect_thresh if is_harvest_ar else 0.0,
            )
        else:
            decision = HarvestDecision.WAIT
            conf = min(
                circularity / self.cfg.circularity_thresh,
                1.0,
            )

        return ClassificationResult(
            detection=detection,
            decision=decision,
            circularity=circularity,
            aspect_ratio=aspect_ratio,
            confidence=round(conf, 3),
        )

    def classify_batch(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> list[ClassificationResult]:
        """Tüm tespitler için toplu sınıflandırma."""
        return [self.classify(frame, d) for d in detections]

    def _compute_circularity(
        self,
        frame: np.ndarray,
        det: Detection,
    ) -> float:
        """
        Bounding box kırpımından dairesellik hesapla.
        circularity = 4π × alan / çevre²
        1.0 = mükemmel daire, 0.0 = düz çizgi
        """
        h, w = frame.shape[:2]
        x1 = max(0, det.cx_px - det.w_px // 2)
        y1 = max(0, det.cy_px - det.h_px // 2)
        x2 = min(w, det.cx_px + det.w_px // 2)
        y2 = min(h, det.cy_px + det.h_px // 2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return 1.0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 1.0

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if perimeter < 1e-6:
            return 1.0

        return round(4.0 * math.pi * area / (perimeter ** 2), 4)
