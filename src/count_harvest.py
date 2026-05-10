"""YOLO tespiti + kutu genişliğine göre hasat / bekle sınıflandırması ve görselleştirme."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

from utils import check_gpu, resolve_device, setup_dirs

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    """Proje kök dizini."""
    return Path(__file__).resolve().parent.parent


def _latest_saved_image(run_dir: Path) -> Path | None:
    """``predict(save=True)`` çıktı klasöründeki en yeni görüntüyü döndürür."""
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    candidates: list[Path] = []
    for pat in patterns:
        candidates.extend(run_dir.glob(pat))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def count_by_cap_width(
    r: object,
    px_to_mm: float,
    harvest_mm: float,
) -> tuple[int, int, int]:
    """
    Kutu genişliğine (xywh ``w``) göre hasat ve bekle sayımlarını üretir.

    Args:
        ``r``: Ultralytics ``Results`` (tek görüntü).
        px_to_mm: Piksel başına milimetre (kalibrasyon).
        harvest_mm: Bu genişlik ve üzeri hasat kabul edilir.

    Returns:
        ``(toplam_kutu, hasat_sayısı, bekle_sayısı)``
    """
    boxes = getattr(r, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return 0, 0, 0
    xywh = boxes.xywh
    if hasattr(xywh, "cpu"):
        xywh = xywh.cpu().numpy()
    n = len(boxes)
    harvest = 0
    wait = 0
    for i in range(n):
        w_px = float(xywh[i, 2])
        cap_mm = w_px * px_to_mm
        if cap_mm >= harvest_mm:
            harvest += 1
        else:
            wait += 1
    return n, harvest, wait


def main() -> None:
    """CLI: tahmin, sayım ve (isteğe bağlı) matplotlib ile önizleme."""
    root = _project_root()
    parser = argparse.ArgumentParser(
        description="Mantar tespiti + kap genişliğine göre hasat/bekle sayımı",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Giriş görüntüsü",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Ağırlık (.pt); varsayılan: models/detection/best.pt",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.35)
    parser.add_argument("--px-to-mm", type=float, default=0.3, help="1 px ≈ kaç mm (kalibre)")
    parser.add_argument("--harvest-mm", type=float, default=40.0, help="Hasat eşiği (kap genişliği, mm)")
    parser.add_argument("--project", type=str, default="runs/test")
    parser.add_argument("--name", type=str, default="sayim")
    parser.add_argument("--line-width", type=int, default=1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Matplotlib penceresini açma (sunucu / headless)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    setup_dirs(root)
    weights = args.weights or (root / "models" / "detection" / "best.pt")
    weights = weights if weights.is_absolute() else (root / weights).resolve()
    if not weights.is_file():
        logger.error("Ağırlık yok: %s", weights)
        sys.exit(1)
    src = args.source if args.source.is_absolute() else (Path.cwd() / args.source).resolve()
    if not src.is_file():
        logger.error("Görüntü yok: %s", src)
        sys.exit(1)

    device = resolve_device(args.device)
    if device == "cuda":
        check_gpu()

    project_dir = (root / args.project).resolve()
    model = YOLO(str(weights))
    results = model.predict(
        source=str(src),
        conf=args.conf,
        iou=args.iou,
        save=True,
        project=str(project_dir),
        name=args.name,
        line_width=args.line_width,
        device=device,
        verbose=False,
    )

    r = results[0]
    total, harvest, wait = count_by_cap_width(r, args.px_to_mm, args.harvest_mm)

    logger.info("Toplam tespit edilen mantar: %d", total)
    logger.info("Hasat edilecek (≥%.1f mm): %d", args.harvest_mm, harvest)
    logger.info("Bekleyecek      (<%.1f mm): %d", args.harvest_mm, wait)

    run_out = project_dir / args.name
    pred_path = _latest_saved_image(run_out)
    if pred_path is None:
        logger.warning("Kaydedilen tahmin görüntüsü bulunamadı: %s", run_out)
        return

    img_bgr = cv2.imread(str(pred_path))
    if img_bgr is None:
        logger.error("Görüntü okunamadı: %s", pred_path)
        sys.exit(1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if args.no_show:
        out_png = run_out / "count_preview.png"
        plt.figure(figsize=(14, 10))
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.title(f"Toplam: {total} mantar | Hasat: {harvest} | Bekle: {wait}")
        plt.savefig(out_png, bbox_inches="tight", dpi=120)
        plt.close()
        logger.info("Önizleme kaydedildi: %s", out_png)
    else:
        plt.figure(figsize=(14, 10))
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.title(f"Toplam: {total} mantar | Hasat: {harvest} | Bekle: {wait}")
        plt.show()


if __name__ == "__main__":
    main()
