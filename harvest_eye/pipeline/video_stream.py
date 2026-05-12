"""Gerçek zamanlı video / kamera stream pipeline."""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from harvest_eye.config import HarvestEyeConfig
from harvest_eye.pipeline.inference import InferencePipeline
from harvest_eye.projection.projector import get_projector

logger = logging.getLogger(__name__)
_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def _is_video(source: str) -> bool:
    return Path(source).suffix.lower() in _VIDEO_SUFFIXES


def run_stream(args: argparse.Namespace) -> None:
    cfg = HarvestEyeConfig()
    if args.config:
        cfg = HarvestEyeConfig.from_yaml(Path(args.config))

    cfg.detection.conf  = args.conf
    cfg.detection.iou   = args.iou
    cfg.harvest.circularity_thresh = args.circularity_thresh
    cfg.harvest.aspect_thresh      = args.aspect_thresh

    pipeline = InferencePipeline(cfg)
    pipeline.setup()

    projector = get_projector(cfg.projection)
    projector.connect()

    source = args.source
    is_video = _is_video(source)

    if is_video:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"Video acilamadi: {source}")
            projector.disconnect()
            sys.exit(1)

        fps        = cap.get(cv2.CAP_PROP_FPS) or 30
        n_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx  = 0
        last_frame: np.ndarray | None = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                results, overlay = pipeline.process(frame, frame_idx)
                last_frame = overlay
                projector.project_frame(overlay)

                if hasattr(projector, "wait_key"):
                    key = projector.wait_key(1)
                    if key == 27:   # ESC
                        break
                    elif key == 32: # Boşluk — pause
                        while True:
                            k2 = projector.wait_key(0)
                            if k2 == 32 or k2 == 27:
                                break
                    elif key == 83 or key == 3:  # Sağ ok
                        pos = min(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) + 30, n_frames - 1)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                    elif key == 81 or key == 2:  # Sol ok
                        pos = max(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 30, 0)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

            # Video bitti — son kare bekle
            if last_frame is not None and hasattr(projector, "wait_key"):
                projector.project_frame(last_frame)
                while True:
                    if projector.wait_key(0) == 27:
                        break
        finally:
            cap.release()
            projector.disconnect()

    else:
        # Tek görüntü
        frame = cv2.imread(source)
        if frame is None:
            logger.error(f"Goruntu acilamadi: {source}")
            projector.disconnect()
            sys.exit(1)

        results, overlay = pipeline.process(frame)
        projector.project_frame(overlay)

        harvest = sum(1 for r in results if r.decision.value == "harvest")
        wait    = sum(1 for r in results if r.decision.value == "wait")
        logger.info(f"Toplam: {len(results)} | Hasat: {harvest} | Bekle: {wait}")

        if hasattr(projector, "wait_key"):
            while True:
                if projector.wait_key(0) == 27:
                    break

        projector.disconnect()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    p = argparse.ArgumentParser(description="Harvest Eye — Mantar hasat tespiti")
    p.add_argument("--source",              required=True, help="Goruntu veya video yolu")
    p.add_argument("--config",              default=None,  help="YAML konfig dosyasi")
    p.add_argument("--conf",                type=float, default=0.25)
    p.add_argument("--iou",                 type=float, default=0.35)
    p.add_argument("--circularity-thresh",  type=float, default=0.75, dest="circularity_thresh")
    p.add_argument("--aspect-thresh",       type=float, default=1.15, dest="aspect_thresh")
    args = p.parse_args()
    run_stream(args)


if __name__ == "__main__":
    main()
