"""Tek kare işleme pipeline'ı."""
from __future__ import annotations
import numpy as np

from harvest_eye.config import HarvestEyeConfig
from harvest_eye.detection.detector import MushroomDetector
from harvest_eye.maturity.shape_classifier import ShapeClassifier, ClassificationResult
from harvest_eye.projection.overlay import draw_harvest_overlay


class InferencePipeline:
    """
    Tek kare: tespit → sınıflandırma → görselleştirme.

    Video stream ve tek görüntü işleme için ortak temel.
    """

    def __init__(self, cfg: HarvestEyeConfig | None = None):
        self.cfg = cfg or HarvestEyeConfig()
        self.detector  = MushroomDetector(self.cfg.detection)
        self.classifier = ShapeClassifier(self.cfg.harvest)

    def setup(self) -> None:
        """Modeli yükle."""
        self.detector.load()

    def process(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
    ) -> tuple[list[ClassificationResult], np.ndarray]:
        """
        Bir kareyi işle.

        Returns:
            (results, overlay_frame) — sonuçlar ve çizimli kare
        """
        detections = self.detector.detect(frame)
        results    = self.classifier.classify_batch(frame, detections)
        overlay    = draw_harvest_overlay(
            frame, results, self.cfg.projection, frame_idx
        )
        return results, overlay
