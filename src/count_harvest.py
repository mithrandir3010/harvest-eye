"""YOLO tespiti + şekil analizi ile hasat / bekle sınıflandırması ve görselleştirme."""

from __future__ import annotations

import argparse
import logging
import math
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


def _compute_circularity(crop_bgr: np.ndarray) -> float:
    """
    OTSU eşik + en büyük kontur üzerinden dairesellik (4π alan / çevre²) döndürür.

    Args:
        crop_bgr: BGR kırpım; boş veya çok küçükse 1.0 (HAM varsayımı).

    Returns:
        ``0..1`` aralığında dairesellik veya geçersiz durumda ``1.0``.
    """
    if crop_bgr.size == 0 or crop_bgr.shape[0] < 2 or crop_bgr.shape[1] < 2:
        return 1.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    peri = float(cv2.arcLength(cnt, True))
    if peri < 1e-6:
        return 1.0
    return (4.0 * math.pi * area) / (peri * peri)


def _box_is_harvest(
    img_bgr: np.ndarray,
    xyxy_row: np.ndarray,
    circularity_thresh: float,
    aspect_thresh: float,
) -> bool:
    """
    Kırpım + bbox en-boy oranına göre HASAT (yeşil) mu HAM (kırmızı) mu döner.

    HASAT: dairesellik ``< circularity_thresh`` **veya** en/boy ``> aspect_thresh``.
    HAM: her iki koşulun tersi.

    Args:
        img_bgr: Tam kare BGR.
        xyxy_row: ``[x1, y1, x2, y2]`` piksel.
        circularity_thresh: Bu değerin altı düzleşmiş sayılır (HASAT sinyali).
        aspect_thresh: Genişlik / yükseklik bu değeri aşarsa HASAT sinyali.

    Returns:
        ``True`` ise HASAT (yeşil), ``False`` ise HAM (kırmızı).
    """
    x1 = int(round(float(xyxy_row[0])))
    y1 = int(round(float(xyxy_row[1])))
    x2 = int(round(float(xyxy_row[2])))
    y2 = int(round(float(xyxy_row[3])))
    h_img, w_img = img_bgr.shape[:2]
    x1 = max(0, min(x1, w_img - 1))
    y1 = max(0, min(y1, h_img - 1))
    x2 = max(0, min(x2, w_img))
    y2 = max(0, min(y2, h_img))
    if x2 <= x1 or y2 <= y1:
        return False
    crop = img_bgr[y1:y2, x1:x2]
    w_box = float(x2 - x1)
    h_box = float(max(1, y2 - y1))
    aspect = w_box / h_box
    circ = _compute_circularity(crop)
    circ_harvest = circ < circularity_thresh
    aspect_harvest = aspect > aspect_thresh
    return circ_harvest or aspect_harvest


def count_by_shape(
    r: object,
    img_bgr: np.ndarray,
    circularity_thresh: float,
    aspect_thresh: float,
) -> tuple[int, int, int]:
    """
    Şekil analizine göre hasat / bekle sayımlarını üretir.

    Args:
        ``r``: Ultralytics ``Results`` (tek görüntü).
        img_bgr: Tespitin yapıldığı BGR görüntüsü (kırpım için).
        circularity_thresh: Düşük dairesellik HASAT eşiği.
        aspect_thresh: Büyük en/boy HASAT eşiği.

    Returns:
        ``(toplam_kutu, hasat_sayısı, bekle_sayısı)``
    """
    boxes = getattr(r, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return 0, 0, 0
    xyxy = boxes.xyxy
    if hasattr(xyxy, "cpu"):
        xyxy = xyxy.cpu().numpy()
    n = len(xyxy)
    harvest = 0
    wait = 0
    for i in range(n):
        if _box_is_harvest(img_bgr, xyxy[i], circularity_thresh, aspect_thresh):
            harvest += 1
        else:
            wait += 1
    return n, harvest, wait


def render_harvest_preview(
    img_bgr: np.ndarray,
    r: object,
    circularity_thresh: float,
    aspect_thresh: float,
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
        circularity_thresh, aspect_thresh: Şekil eşikleri.
        total, harvest, wait: Başlıkta gösterilecek sayılar.
        frame_idx: Verilmişse başlığa ``Kare: X |`` eklenir (video modu).

    Returns:
        Başlık şeridi eklenmiş BGR görüntü.
    """
    out = img_bgr.copy()
    boxes = getattr(r, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy
        if hasattr(xyxy, "cpu"):
            xyxy = xyxy.cpu().numpy()
        for i in range(len(xyxy)):
            row = xyxy[i]
            x1 = float(row[0])
            y1 = float(row[1])
            x2 = float(row[2])
            y2 = float(row[3])
            cx = int(round((x1 + x2) / 2.0))
            cy = int(round((y1 + y2) / 2.0))
            is_harv = _box_is_harvest(img_bgr, row, circularity_thresh, aspect_thresh)
            fill = _COLOR_HARVEST_BGR if is_harv else _COLOR_WAIT_BGR
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
    parser.add_argument(
        "--circularity-thresh",
        type=float,
        default=0.75,
        help="Bu dairesellik değerinin altı HASAT sinyali (düzleşmiş)",
    )
    parser.add_argument(
        "--aspect-thresh",
        type=float,
        default=1.15,
        help="En/boy bu değeri aşarsa HASAT sinyali (yatay yayılmış)",
    )
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
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n_frames <= 0:
            n_frames = 2**31 - 1
        last_preview: np.ndarray | None = None
        writer_mp4: cv2.VideoWriter | None = None
        writer_avi: cv2.VideoWriter | None = None
        vid_frames_written: int = 0
        _video_writers_init_done: bool = False

        def _ensure_video_writers(prv: np.ndarray) -> None:
            """İlk kare boyutuna göre MP4 (mp4v) ve AVI (XVID) yazıcılarını oluşturur."""
            nonlocal writer_mp4, writer_avi, _video_writers_init_done
            if _video_writers_init_done:
                return
            fh, fw = prv.shape[:2]
            fps_raw = cap.get(cv2.CAP_PROP_FPS)
            fps = float(fps_raw) if fps_raw and fps_raw > 0 else 25.0
            if fps > 120.0:
                fps = 25.0

            out_mp4 = (run_out / "output_video.mp4").resolve()
            out_avi = (run_out / "output_video.avi").resolve()
            logger.info(f"Video kaydediliyor: {out_mp4}")
            w_tmp = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))
            if not w_tmp.isOpened():
                logger.error("VideoWriter açılamadı!")
                writer_mp4 = None
            else:
                writer_mp4 = w_tmp

            logger.info(f"Video kaydediliyor: {out_avi}")
            w_tmp2 = cv2.VideoWriter(str(out_avi), cv2.VideoWriter_fourcc(*"XVID"), fps, (fw, fh))
            if not w_tmp2.isOpened():
                logger.error("VideoWriter açılamadı!")
                writer_avi = None
            else:
                writer_avi = w_tmp2
            _video_writers_init_done = True

        def _write_video_output(prv: np.ndarray) -> None:
            """İşlenmiş kareyi her iki yazıcıya yazar; her 100 karede günlük."""
            nonlocal vid_frames_written
            _ensure_video_writers(prv)
            if writer_mp4 is not None and writer_mp4.isOpened():
                writer_mp4.write(prv)
            if writer_avi is not None and writer_avi.isOpened():
                writer_avi.write(prv)
            vid_frames_written += 1
            if vid_frames_written % 100 == 0:
                logger.info(f"{vid_frames_written} kare yazıldı")

        predict_kw: dict[str, object] = {
            "conf": args.conf,
            "iou": args.iou,
            "save": False,
            "device": device,
            "verbose": False,
        }

        def _infer_frame(frame_bgr: np.ndarray) -> None:
            """Mevcut kareyi işler; ``last_preview`` ve pencereyi günceller (``--no-show`` değilse)."""
            nonlocal last_preview
            frame_idx_disp = max(1, int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            for r in model.predict(source=frame_bgr, stream=True, **predict_kw):
                total, harvest, wait = count_by_shape(
                    r, frame_bgr, args.circularity_thresh, args.aspect_thresh
                )
                preview = render_harvest_preview(
                    frame_bgr,
                    r,
                    args.circularity_thresh,
                    args.aspect_thresh,
                    total,
                    harvest,
                    wait,
                    frame_idx=frame_idx_disp,
                )
                last_preview = preview
                _write_video_output(preview)
                if not args.no_show:
                    cv2.imshow("Harvest Eye", preview)
                break

        try:
            if args.no_show:
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        _infer_frame(frame)
                finally:
                    cap.release()
            else:
                paused = False
                ended_on_eof = False
                try:
                    cv2.namedWindow("Harvest Eye", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Harvest Eye", 1280, 720)
                    while True:
                        if not paused:
                            ret, frame = cap.read()
                            if not ret:
                                ended_on_eof = True
                                break
                            _infer_frame(frame)
                        else:
                            if last_preview is not None:
                                cv2.imshow("Harvest Eye", last_preview)

                        raw_key = cv2.waitKey(0 if paused else 1)
                        print(f"key: {raw_key}")
                        key = raw_key & 0xFF
                        if key == 27:
                            ended_on_eof = False
                            break
                        if key == 32:
                            paused = not paused
                            continue
                        if key == 83 or key == 3:
                            next_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                            tgt = min(next_pos + 30, max(0, n_frames - 1))
                            cap.set(cv2.CAP_PROP_POS_FRAMES, tgt)
                            ret, frame = cap.read()
                            if ret:
                                _infer_frame(frame)
                            continue
                        if key == 81 or key == 2:
                            next_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                            tgt = max(next_pos - 30, 0)
                            cap.set(cv2.CAP_PROP_POS_FRAMES, tgt)
                            ret, frame = cap.read()
                            if ret:
                                _infer_frame(frame)
                            continue
                finally:
                    cap.release()

                if last_preview is not None and ended_on_eof:
                    cv2.imshow("Harvest Eye", last_preview)
                    while True:
                        raw_key = cv2.waitKey(0)
                        print(f"key: {raw_key}")
                        key = raw_key & 0xFF
                        if key == 27:
                            break

                cv2.destroyAllWindows()
        finally:
            if writer_mp4 is not None:
                writer_mp4.release()
            if writer_avi is not None:
                writer_avi.release()

        out_png = run_out / "count_preview.png"
        if last_preview is not None:
            cv2.imwrite(str(out_png), last_preview)
            logger.info("Son kare kaydedildi: %s", out_png)
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

    img_bgr = cv2.imread(str(src))
    if img_bgr is None:
        logger.error("Görüntü okunamadı: %s", src)
        sys.exit(1)

    total, harvest, wait = count_by_shape(
        r, img_bgr, args.circularity_thresh, args.aspect_thresh
    )

    logger.info("Toplam tespit edilen mantar: %d", total)
    logger.info("Hasat (yesil, sekil): %d", harvest)
    logger.info("Bekle (kirmizi, sekil): %d", wait)

    preview = render_harvest_preview(
        img_bgr,
        r,
        args.circularity_thresh,
        args.aspect_thresh,
        total,
        harvest,
        wait,
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
