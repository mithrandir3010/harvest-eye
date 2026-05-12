"""YOLOv8 mantar tespit modülü."""
from __future__ import annotations
import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from harvest_eye.config import DetectionConfig

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Tek bir tespit sonucu."""
    cx: float        # normalize merkez x (0-1)
    cy: float        # normalize merkez y (0-1)
    w: float         # normalize genişlik (0-1)
    h: float         # normalize yükseklik (0-1)
    conf: float      # güven skoru
    class_id: int    # sınıf ID
    class_name: str  # sınıf adı
    cx_px: int       # piksel merkez x
    cy_px: int       # piksel merkez y
    w_px: int        # piksel genişlik
    h_px: int        # piksel yükseklik


class MushroomDetector:
    """YOLOv8 tabanlı mantar tespit sınıfı."""

    def __init__(self, cfg: DetectionConfig | None = None):
        self.cfg = cfg or DetectionConfig()
        self._model: YOLO | None = None
        self._class_names: list[str] = ["mushroom"]

    def load(self) -> None:
        """Modeli belleğe yükle."""
        weights = Path(self.cfg.weights)
        if not weights.exists():
            raise FileNotFoundError(f"Model bulunamadı: {weights}")
        self._model = YOLO(str(weights))
        logger.info(f"Model yüklendi: {weights}")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Bir kare üzerinde tespit yap.

        Args:
            frame: BGR numpy array (H, W, 3)

        Returns:
            Detection listesi
        """
        if not self.is_loaded:
            self.load()

        results = self._model.predict(
            source=frame,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device if self.cfg.device != "auto" else None,
            verbose=False,
        )

        detections: list[Detection] = []
        if not results:
            return detections

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections

        xywh = r.boxes.xywhn.cpu().numpy()   # normalize
        xywh_px = r.boxes.xywh.cpu().numpy() # piksel
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        for i in range(len(xywh)):
            cx, cy, w, h = xywh[i]
            cx_p, cy_p, w_p, h_p = xywh_px[i]
            cls_id = int(classes[i])
            cls_name = (
                self._class_names[cls_id]
                if cls_id < len(self._class_names)
                else f"class_{cls_id}"
            )
            detections.append(Detection(
                cx=float(cx), cy=float(cy),
                w=float(w), h=float(h),
                conf=float(confs[i]),
                class_id=cls_id,
                class_name=cls_name,
                cx_px=int(cx_p), cy_px=int(cy_p),
                w_px=int(w_p), h_px=int(h_p),
            ))

        return detections
