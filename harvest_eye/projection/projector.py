"""Projektör kontrol arayüzü ve implementasyonları."""
from __future__ import annotations
import abc
import logging
from pathlib import Path

import cv2
import numpy as np

from harvest_eye.config import ProjectionConfig

logger = logging.getLogger(__name__)


class Projector(abc.ABC):
    """
    Projektör soyut temel sınıfı.

    Tüm projektör implementasyonları bu arayüzü uygular.
    Donanım değişse bile pipeline kodu değişmez.
    """

    @abc.abstractmethod
    def connect(self) -> bool:
        """Projektöre bağlan. Başarılıysa True döner."""

    @abc.abstractmethod
    def project_frame(self, frame: np.ndarray) -> None:
        """İşlenmiş kareyi projektöre gönder."""

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Bağlantıyı kapat."""

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()


class SimulatedProjector(Projector):
    """
    Simüle projektör — OpenCV penceresi kullanır.

    Gerçek donanım olmadan geliştirme ve test için.
    Mevcut count_harvest.py görselleştirmesiyle aynı davranır.
    """

    def __init__(self, cfg: ProjectionConfig | None = None):
        self.cfg = cfg or ProjectionConfig()
        self._window_name = "Harvest Eye — Simulation"
        self._connected = False

    def connect(self) -> bool:
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self._window_name,
            self.cfg.display_width,
            self.cfg.display_height,
        )
        self._connected = True
        logger.info("SimulatedProjector baglandi (OpenCV penceresi)")
        return True

    def project_frame(self, frame: np.ndarray) -> None:
        if not self._connected:
            return
        cv2.imshow(self._window_name, frame)

    def wait_key(self, delay: int = 1) -> int:
        return cv2.waitKey(delay) & 0xFF

    def disconnect(self) -> None:
        cv2.destroyWindow(self._window_name)
        self._connected = False
        logger.info("SimulatedProjector baglanti kesildi")


class PicoProjector(Projector):
    """
    Pico projektör implementasyonu — HDMI/USB output.

    TODO: Gerçek donanım gelince implement edilecek.
    Şu an NotImplementedError fırlatır.
    """

    def __init__(self, cfg: ProjectionConfig | None = None, display_index: int = 1):
        self.cfg = cfg or ProjectionConfig()
        self.display_index = display_index  # ikinci ekran = projektör

    def connect(self) -> bool:
        raise NotImplementedError(
            "PicoProjector henüz implement edilmedi. "
            "SimulatedProjector kullanın."
        )

    def project_frame(self, frame: np.ndarray) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError


def get_projector(cfg: ProjectionConfig | None = None) -> Projector:
    """
    Konfigürasyona göre uygun projektör döner.

    Şu an her zaman SimulatedProjector döner.
    Gerçek donanım eklenince burada seçim yapılır.
    """
    cfg = cfg or ProjectionConfig()
    logger.info("SimulatedProjector secildi (PicoProjector henuz hazir degil)")
    return SimulatedProjector(cfg)
