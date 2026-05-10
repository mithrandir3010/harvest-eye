"""Tek görüntü üzerinde YOLOv8 tespit tahmini."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ultralytics import YOLO

from utils import check_gpu, resolve_device, setup_dirs

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    """Proje kök dizini."""
    return Path(__file__).resolve().parent.parent


def main() -> None:
    """Bir görüntü yolundan kutu tahmini üretir ve loglar."""
    parser = argparse.ArgumentParser(description="Harvest-Eye tek görüntü çıkarımı")
    parser.add_argument("--source", type=Path, required=True, help="Görüntü dosyası")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Ağırlık (.pt); varsayılan: models/detection/best.pt",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    root = _project_root()
    setup_dirs(root)

    weights = args.weights or (root / "models" / "detection" / "best.pt")
    if not weights.is_file():
        logger.error("Ağırlık bulunamadı: %s — önce train_detection.py çalıştırın.", weights)
        sys.exit(1)
    if not args.source.is_file():
        logger.error("Görüntü yok: %s", args.source)
        sys.exit(1)

    device = resolve_device(args.device)
    if device == "cuda":
        check_gpu()

    model = YOLO(str(weights))
    results = model.predict(source=str(args.source), device=device, imgsz=args.imgsz, verbose=False)

    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        logger.info("Tespit sayısı: %d", len(boxes))
        if len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            names = model.names
            for i in range(len(boxes)):
                ci = int(cls[i])
                if isinstance(names, dict):
                    cname = names.get(ci, str(ci))
                else:
                    cname = names[ci]
                logger.info(
                    "  [%d] %s conf=%.3f box=%s",
                    i,
                    cname,
                    float(conf[i]),
                    [float(x) for x in xyxy[i]],
                )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
