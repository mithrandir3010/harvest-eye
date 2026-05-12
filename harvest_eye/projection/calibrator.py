"""Kamera ↔ projektör homografi kalibrasyon modülü."""
from __future__ import annotations
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CALIBRATION_DIR = Path(__file__).parent.parent.parent / "data/calibration"


class ProjectorCalibrator:
    """
    Kamera koordinatını projektör koordinatına dönüştürür.

    Homografi matrisi bir kez hesaplanır ve .npy dosyasına kaydedilir.
    Sonraki çalıştırmalarda diskten yüklenir.
    """

    def __init__(self, homography_path: Path | None = None):
        self.homography_path = homography_path or (
            CALIBRATION_DIR / "homography.npy"
        )
        self._H: np.ndarray | None = None

    @property
    def is_calibrated(self) -> bool:
        return self._H is not None

    def load(self) -> bool:
        """Kayıtlı homografi matrisini yükle."""
        if not self.homography_path.exists():
            logger.warning(f"Homografi dosyasi bulunamadi: {self.homography_path}")
            return False
        self._H = np.load(str(self.homography_path))
        logger.info(f"Homografi yuklendi: {self.homography_path}")
        return True

    def save(self) -> None:
        """Homografi matrisini kaydet."""
        if self._H is None:
            raise RuntimeError("Kalibrasyon yapilmamis, kayit edilecek matris yok.")
        self.homography_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(self.homography_path), self._H)
        logger.info(f"Homografi kaydedildi: {self.homography_path}")

    def calibrate_from_points(
        self,
        camera_points: np.ndarray,
        projector_points: np.ndarray,
    ) -> None:
        """
        Nokta çiftlerinden homografi hesapla.

        Args:
            camera_points: Kamera koordinatları, shape (N, 2)
            projector_points: Projektör koordinatları, shape (N, 2)
        """
        if len(camera_points) < 4:
            raise ValueError("En az 4 nokta çifti gerekli.")

        self._H, mask = cv2.findHomography(
            camera_points.astype(np.float32),
            projector_points.astype(np.float32),
            cv2.RANSAC,
            5.0,
        )
        inliers = int(mask.sum()) if mask is not None else 0
        logger.info(f"Homografi hesaplandi. Inlier: {inliers}/{len(camera_points)}")

    def transform(self, point: tuple[float, float]) -> tuple[int, int]:
        """
        Kamera koordinatını projektör koordinatına dönüştür.

        Args:
            point: (x, y) kamera koordinatı

        Returns:
            (x, y) projektör koordinatı
        """
        if self._H is None:
            raise RuntimeError("Kalibrasyon yapilmamis. load() veya calibrate_from_points() cagir.")

        pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self._H)
        x, y = transformed[0][0]
        return int(x), int(y)

    def transform_batch(
        self, points: list[tuple[float, float]]
    ) -> list[tuple[int, int]]:
        """Birden fazla noktayı dönüştür."""
        return [self.transform(p) for p in points]
