"""YOLO tespiti + kutu genişliğine göre hasat / bekle sınıflandırması ve görselleştirme."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from utils import check_gpu, resolve_device, setup_dirs

logger = logging.getLogger(__name__)

# BGR (OpenCV)
_COLOR_HARVEST_BGR = (0, 220, 0)
_COLOR_WAIT_BGR = (0, 0, 220)
_COLOR_RING_BGR = (255, 255, 255)
_CIRCLE_RADIUS_PX = 8
_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov"}


def _project_root() -> Path:
    """Proje kök dizini."""
    return Path(__file__).resolve().parent.parent


def _is_video_source(path: Path) -> bool:
    """``.mp4``, ``.avi``, ``.mov`` için video modu."""
    return path.suffix.lower() in _VIDEO_SUFFIXES


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


def render_harvest_preview(
    img_bgr: np.ndarray,
    r: object,
    px_to_mm: float,
    harvest_mm: float,
    total: int,
    harvest: int,
    wait: int,
    frame_idx: int | None = None,
) -> np.ndarray:
    """
    Orijinal BGR görüntü üzerine merkez daireleri ve üst başlığı çizer (bbox yok).

    Args:
        img_bgr: Kaynak görüntü (BGR).
        ``r``: Ultralytics sonuç (kutular).
        px_to_mm: Piksel → mm ölçeği.
        harvest_mm: Hasat eşiği (mm).
        total, harvest, wait: Başlıkta gösterilecek sayılar.
        frame_idx: Verilmişse başlığa ``Kare: X |`` eklenir (video modu).

    Returns:
        Başlık şeridi eklenmiş BGR görüntü.
    """
    out = img_bgr.copy()
    boxes = getattr(r, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        xywh = boxes.xywh
        if hasattr(xywh, "cpu"):
            xywh = xywh.cpu().numpy()
        for i in range(len(boxes)):
            cx = int(round(float(xywh[i, 0])))
            cy = int(round(float(xywh[i, 1])))
            w_px = float(xywh[i, 2])
            cap_mm = w_px * px_to_mm
            fill = _COLOR_HARVEST_BGR if cap_mm >= harvest_mm else _COLOR_WAIT_BGR
            center = (cx, cy)
            cv2.circle(out, center, _CIRCLE_RADIUS_PX, fill, thickness=-1)
            cv2.circle(out, center, _CIRCLE_RADIUS_PX, _COLOR_RING_BGR, thickness=1)

    if frame_idx is not None:
        title = f"Kare: {frame_idx} | Toplam: {total} | Hasat: {harvest} (yesil) | Bekle: {wait} (kirmizi)"
    else:
        title = f"Toplam: {total} | Hasat: {harvest} (yesil) | Bekle: {wait} (kirmizi)"
    header_h = 48
    h, w = out.shape[:2]
    header = np.full((header_h, w, 3), 32, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = min(0.55, (w * 0.85) / max(cv2.getTextSize(title, font, 0.65, 2)[0][0], 1))
    thickness = 2
    (tw, th), _ = cv2.getTextSize(title, font, scale, thickness)
    tx = max(8, (w - tw) // 2)
    ty = (header_h + th) // 2
    cv2.putText(header, title, (tx, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return np.vstack([header, out])


def main() -> None:
    """CLI: tahmin, sayım ve OpenCV ile önizleme."""
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
        help="Pencere açma; yalnızca PNG kaydet (headless)",
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
    run_out = project_dir / args.name
    run_out.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))

    if _is_video_source(src):
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            logger.error("Video açılamadı: %s", src)
            sys.exit(1)
        frame_idx = 0
        last_preview: np.ndarray | None = None
        if not args.no_show:
            cv2.namedWindow("Harvest Eye", cv2.WINDOW_NORMAL)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                stop_display = False
                for r in model.predict(
                    source=frame,
                    stream=True,
                    conf=args.conf,
                    iou=args.iou,
                    save=False,
                    device=device,
                    verbose=False,
                ):
                    total, harvest, wait = count_by_cap_width(r, args.px_to_mm, args.harvest_mm)
                    preview = render_harvest_preview(
                        frame,
                        r,
                        args.px_to_mm,
                        args.harvest_mm,
                        total,
                        harvest,
                        wait,
                        frame_idx=frame_idx,
                    )
                    last_preview = preview
                    if not args.no_show:
                        hp, wp = preview.shape[:2]
                        cv2.resizeWindow("Harvest Eye", wp, hp)
                        cv2.imshow("Harvest Eye", preview)
                        if (cv2.waitKey(1) & 0xFF) == 27:
                            stop_display = True
                            break
                if stop_display:
                    break
        finally:
            cap.release()
            if not args.no_show:
                cv2.destroyAllWindows()

        out_png = run_out / "count_preview.png"
        if last_preview is not None:
            cv2.imwrite(str(out_png), last_preview)
            logger.info("Son kare kaydedildi: %s (toplam %d kare işlendi)", out_png, frame_idx)
        else:
            logger.warning("Kaydedilecek kare yok (video boş olabilir).")
        return

    results = model.predict(
        source=str(src),
        conf=args.conf,
        iou=args.iou,
        save=False,
        device=device,
        verbose=False,
    )

    r = results[0]
    total, harvest, wait = count_by_cap_width(r, args.px_to_mm, args.harvest_mm)

    logger.info("Toplam tespit edilen mantar: %d", total)
    logger.info("Hasat edilecek (≥%.1f mm): %d", args.harvest_mm, harvest)
    logger.info("Bekleyecek      (<%.1f mm): %d", args.harvest_mm, wait)

    img_bgr = cv2.imread(str(src))
    if img_bgr is None:
        logger.error("Görüntü okunamadı: %s", src)
        sys.exit(1)

    preview = render_harvest_preview(
        img_bgr, r, args.px_to_mm, args.harvest_mm, total, harvest, wait
    )
    out_png = run_out / "count_preview.png"
    cv2.imwrite(str(out_png), preview)
    logger.info("Önizleme kaydedildi: %s", out_png)

    if not args.no_show:
        h_img, w_img = img_bgr.shape[:2]
        cv2.namedWindow("Harvest Eye", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Harvest Eye", w_img, h_img)
        cv2.imshow("Harvest Eye", preview)
        logger.info("Çıkmak için pencerede bir tuşa basın.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
