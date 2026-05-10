"""YOLOv8 ile Model 1 (mantar tespiti) eğitimi, metrik raporu ve ONNX dışa aktarımı."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

from utils import check_gpu, resolve_device, setup_dirs

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    """Proje kök dizini."""
    return Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    """Argparse yapılandırması."""
    p = argparse.ArgumentParser(description="Harvest-Eye YOLOv8 tespit eğitimi")
    p.add_argument("--model", choices=("yolov8n", "yolov8s", "yolov8m"), default="yolov8n")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return p.parse_args()


def main() -> None:
    """Eğitim, en iyi ağırlığı kopyalama, doğrulama metrikleri ve ONNX export."""
    args = _parse_args()
    root = _project_root()
    setup_dirs(root)

    data_yaml = root / "data" / "datasets.yaml"
    if not data_yaml.is_file():
        logger.error("Veri yapılandırması yok: %s — önce prepare_data.py çalıştırın.", data_yaml)
        sys.exit(1)

    device = resolve_device(args.device)
    if device == "cuda":
        check_gpu()

    weights_name = f"{args.model}.pt"
    model = YOLO(weights_name)

    logger.info("Eğitim başlıyor: model=%s epochs=%s batch=%s imgsz=%s device=%s", args.model, args.epochs, args.batch, args.imgsz, device)

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        project=str(root / "runs" / "detection"),
        name="train",
        patience=20,
        optimizer="auto",
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
        exist_ok=True,
    )

    trainer = getattr(model, "trainer", None)
    best_src: Path | None = None
    if trainer is not None and getattr(trainer, "best", None):
        best_src = Path(trainer.best)
    if best_src is None or not best_src.is_file():
        logger.error("Eğitim sonrası best.pt bulunamadı.")
        sys.exit(1)

    dest_dir = root / "models" / "detection"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_pt = dest_dir / "best.pt"
    shutil.copy2(best_src, dest_pt)
    logger.info("En iyi ağırlık kopyalandı: %s", dest_pt)

    val_model = YOLO(str(dest_pt))
    metrics = val_model.val(data=str(data_yaml), device=device, imgsz=args.imgsz)
    box = getattr(metrics, "box", metrics)
    map50 = getattr(box, "map50", None)
    mp = getattr(box, "mp", None)
    mr = getattr(box, "mr", None)
    logger.info("Doğrulama — mAP50: %s, precision: %s, recall: %s", map50, mp, mr)

    export_model = YOLO(str(dest_pt))
    export_model.export(format="onnx", imgsz=args.imgsz, simplify=True)
    onnx_candidate = dest_pt.with_suffix(".onnx")
    dest_onnx = dest_dir / "best.onnx"
    if onnx_candidate.is_file():
        if onnx_candidate.resolve() != dest_onnx.resolve():
            shutil.move(str(onnx_candidate), str(dest_onnx))
        logger.info("ONNX dışa aktarıldı: %s", dest_onnx)
    else:
        found = sorted(dest_dir.glob("*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
        if found:
            if found[0].resolve() != dest_onnx.resolve():
                shutil.move(str(found[0]), str(dest_onnx))
            logger.info("ONNX dışa aktarıldı: %s", dest_onnx)
        else:
            logger.warning("ONNX dosyası bulunamadı; ``runs/`` veya proje kökünde .onnx arayın.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
