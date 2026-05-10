"""Ortak yardımcılar: GPU kontrolü, dizinler, indirme ve COCO→YOLO dönüşümleri."""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path
from typing import Any, Mapping

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def check_gpu() -> bool:
    """
    CUDA kullanılabilirliğini kontrol eder; varsa cihaz ve CUDA sürümünü loglar.

    Returns:
        ``True`` eğer CUDA mevcut ve kullanılabilir ise, aksi halde ``False``.
    """
    import torch

    if not torch.cuda.is_available():
        logger.warning("CUDA kullanılamıyor — eğitim/tahmin CPU üzerinde çalışacak.")
        return False
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    cv = torch.version.cuda or "bilinmiyor"
    logger.info("GPU: %s | compute capability: %s.%s | PyTorch CUDA: %s", name, major, minor, cv)
    return True


def setup_dirs(root: Path) -> None:
    """
    Proje için beklenen veri, model ve çıktı dizinlerini oluşturur.

    Args:
        root: Proje kök dizini (``harvest-eye``).
    """
    subdirs = [
        root / "data" / "raw",
        root / "data" / "processed" / "train" / "images",
        root / "data" / "processed" / "train" / "labels",
        root / "data" / "processed" / "val" / "images",
        root / "data" / "processed" / "val" / "labels",
        root / "data" / "processed" / "test" / "images",
        root / "data" / "processed" / "test" / "labels",
        root / "models" / "detection",
        root / "models" / "maturity",
        root / "runs",
        root / "notebooks",
    ]
    for d in subdirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.debug("Dizinler hazır: %d yol", len(subdirs))


def download_file(url: str, dest: Path, desc: str | None = None) -> None:
    """
    HTTP(S) üzerinden dosyayı akışla indirir ve ``tqdm`` ile ilerleme gösterir.

    Args:
        url: İndirilecek kaynak adresi.
        dest: Yazılacak dosya yolu (üst dizin oluşturulur).
        desc: İlerleme çubuğu etiketi.

    Raises:
        requests.RequestException: Ağ veya HTTP hata kodu durumunda.
    """
    dest = dest.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    label = desc or dest.name
    try:
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            chunk = 1024 * 256
            with open(dest, "wb") as out, tqdm(
                total=total if total > 0 else None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=label,
            ) as bar:
                for data in resp.iter_content(chunk_size=chunk):
                    if data:
                        out.write(data)
                        bar.update(len(data))
    except requests.RequestException as exc:
        if dest.is_file():
            dest.unlink(missing_ok=True)
        logger.error("İndirme başarısız (%s): %s", url, exc)
        raise


def coco_to_yolo(
    annotation: Mapping[str, Any],
    img_w: int,
    img_h: int,
) -> tuple[int, float, float, float, float]:
    """
    Tek bir COCO nesnesinden YOLO satırı üretir (normalize merkez + genişlik/yükseklik).

    ``annotation`` şu anahtarları içermelidir:

    - ``bbox``: ``[x_min, y_min, width, height]`` (piksel, COCO)
    - ``class_id``: YOLO sınıf indeksi (``0 .. nc-1``)

    Args:
        annotation: COCO bbox ve eşlenmiş YOLO ``class_id`` içeren sözlük.
        img_w: Görüntü genişliği.
        img_h: Görüntü yüksekliği.

    Returns:
        ``(class_id, x_center, y_center, w, h)`` — tüm koordinatlar 0–1 aralığında.

    Raises:
        KeyError: Gerekli anahtarlar eksikse.
        ValueError: Boyutlar geçersizse.
    """
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Geçersiz görüntü boyutu: {img_w}x{img_h}")
    bbox = annotation["bbox"]
    class_id = int(annotation["class_id"])
    x, y, w, h = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    x_c = (x + w / 2.0) / img_w
    y_c = (y + h / 2.0) / img_h
    w_n = w / img_w
    h_n = h / img_h
    return class_id, x_c, y_c, w_n, h_n


def seg_to_bbox(polygon_points: list[float] | list[list[float]]) -> tuple[float, float, float, float]:
    """
    COCO segmentation çokgeninden eksen hizalı sınırlayıcı kutu (piksel) üretir.

    Args:
        polygon_points: Düz liste ``[x1,y1,x2,y2,...]`` veya ``[[x,y], ...]``.

    Returns:
        ``(x_min, y_min, width, height)`` piksel cinsinden.
    """
    xs: list[float]
    ys: list[float]
    if polygon_points and isinstance(polygon_points[0], (list, tuple)):
        pairs = polygon_points  # type: ignore[assignment]
        xs = [float(p[0]) for p in pairs]
        ys = [float(p[1]) for p in pairs]
    else:
        flat = [float(x) for x in polygon_points]  # type: ignore[arg-type]
        xs = flat[0::2]
        ys = flat[1::2]
    if not xs or not ys:
        raise ValueError("Segmentation çokgeni boş.")
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return x_min, y_min, x_max - x_min, y_max - y_min


def extract_zip(archive: Path, dest_dir: Path) -> None:
    """
    ZIP arşivini hedef dizine güvenli şekilde açar.

    Args:
        archive: ZIP dosyası.
        dest_dir: Çıktı kökü.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(dest_dir)
    logger.info("ZIP çıkarıldı: %s -> %s", archive, dest_dir)


def try_extract_rar(archive: Path, dest_dir: Path) -> bool:
    """
    RAR arşivini açmayı dener (``rarfile`` veya sistemdeki ``7z`` / ``unrar``).

    Args:
        archive: RAR dosyası.
        dest_dir: Çıktı kökü.

    Returns:
        ``True`` eğer çıkarma başarılı olduysa.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        import rarfile  # type: ignore[import-untyped]

        with rarfile.RarFile(archive) as rf:
            rf.extractall(dest_dir)
        logger.info("RAR çıkarıldı (rarfile): %s", archive)
        return True
    except ImportError:
        logger.debug("rarfile yüklü değil; harici araç deneniyor.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("rarfile ile açılamadı: %s", exc)

    seven = shutil.which("7z") or shutil.which("7zz")
    if seven:
        import subprocess

        try:
            subprocess.run(
                [seven, "x", "-y", f"-o{dest_dir}", str(archive)],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("RAR/ZIP çıkarıldı (7-Zip): %s", archive)
            return True
        except subprocess.CalledProcessError as exc:
            logger.warning("7-Zip çıkarma hatası: %s", exc)

    unrar = shutil.which("unrar")
    if unrar:
        import subprocess

        try:
            subprocess.run([unrar, "x", "-y", str(archive), str(dest_dir)], check=True, capture_output=True, text=True)
            logger.info("RAR çıkarıldı (unrar): %s", archive)
            return True
        except subprocess.CalledProcessError as exc:
            logger.warning("unrar çıkarma hatası: %s", exc)

    logger.error(
        "RAR açılamadı. ``pip install rarfile`` ve WinRAR/UnRAR veya 7-Zip kurun; "
        "veya ``%s`` dosyasını elle ``%s`` içine çıkarın.",
        archive,
        dest_dir,
    )
    return False


def resolve_device(requested: str) -> str:
    """
    İstenen cihazı doğrular; ``cuda`` istenmiş ama yoksa ``cpu`` döner.

    Args:
        requested: ``cuda`` veya ``cpu``.

    Returns:
        Kullanılacak cihaz dizesi.
    """
    import torch

    r = requested.lower().strip()
    if r == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA yok — cihaz cpu olarak ayarlandı.")
        return "cpu"
    return r
