"""
Kamera ↔ projektör kalibrasyon scripti.

Kullanım:
    python scripts/calibrate_projector.py
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from harvest_eye.projection.calibrator import ProjectorCalibrator

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Projektor kalibrasyonu")
    p.add_argument("--output", default=str(ROOT / "data/calibration/homography.npy"))
    args = p.parse_args()

    logger.info("=" * 50)
    logger.info("PROJEKTOR KALİBRASYONU")
    logger.info("=" * 50)
    logger.info("TODO: Bu script henuz tamamlanmamis.")
    logger.info("Adimlar:")
    logger.info("  1. Kamerayı ve projektoru monte et")
    logger.info("  2. 9x6 checkerboard pattern mantar yüzeyine koy")
    logger.info("  3. Bu scripti calistir")
    logger.info("  4. Ekrandaki yönergeleri takip et")
    logger.info("")
    logger.info("Simdilik: 4 köse noktasi ile temel kalibrasyon")

    # Yer tutucu — gerçek kalibrasyon ileride eklenecek
    # Örnek: 4 köşe noktası manuel girilir
    print("\n4 kamera koordinati girin (x y formatında, her satira bir tane):")
    camera_pts = []
    for i in range(4):
        line = input(f"  Nokta {i+1}: ")
        x, y = map(float, line.strip().split())
        camera_pts.append([x, y])

    print("\n4 projektor koordinati girin (kameradaki noktalara karsilik):")
    proj_pts = []
    for i in range(4):
        line = input(f"  Nokta {i+1}: ")
        x, y = map(float, line.strip().split())
        proj_pts.append([x, y])

    calibrator = ProjectorCalibrator(Path(args.output))
    calibrator.calibrate_from_points(
        np.array(camera_pts), np.array(proj_pts)
    )
    calibrator.save()
    logger.info(f"Kalibrasyon tamamlandi: {args.output}")


if __name__ == "__main__":
    main()
