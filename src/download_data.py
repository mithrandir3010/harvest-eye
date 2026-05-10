"""M18K (Google Drive) ve Oyster Maturity (Mendeley) veri setlerini indirir ve arşivleri açar."""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import Path

import gdown
import requests

from utils import (
    download_file,
    extract_zip,
    setup_dirs,
    try_extract_rar,
)

logger = logging.getLogger(__name__)

M18K_DRIVE_ID = "1iPANwP1k6tbz1EZRdG2qbst1tqqYsZ_x"
MENDELEY_DATASET_ID = "hf55tkx489"
MENDELEY_API = f"https://data.mendeley.com/public-api/datasets/{MENDELEY_DATASET_ID}"
COMPLETE_NAME = ".harvest_eye_download_complete"


def _project_root() -> Path:
    """``src/download_data.py`` için üst dizin (proje kökü)."""
    return Path(__file__).resolve().parent.parent


def _write_complete(marker_dir: Path) -> None:
    """İndirmenin tamamlandığını işaretler."""
    (marker_dir / COMPLETE_NAME).write_text("ok\n", encoding="utf-8")


def _is_complete(marker_dir: Path) -> bool:
    """Önceki başarılı indirme var mı?"""
    return (marker_dir / COMPLETE_NAME).is_file()


def _mendeley_download_url() -> tuple[str, str]:
    """
    Mendeley public API ile birincil dosyanın indirme URL'sini ve dosya adını döndürür.

    Returns:
        ``(download_url, filename)``

    Raises:
        RuntimeError: API yanıtı beklenen formatta değilse.
    """
    try:
        resp = requests.get(MENDELEY_API, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Mendeley API okunamadı ({MENDELEY_API}): {exc}") from exc

    files = data.get("files")
    if not files:
        raise RuntimeError("Mendeley yanıtında dosya listesi yok.")
    main = files[0] if isinstance(files, list) else files
    details = main.get("content_details") or {}
    url = details.get("download_url")
    name = main.get("filename") or "download.bin"
    if not url:
        raise RuntimeError("Mendeley dosyasında download_url bulunamadı.")
    return str(url), str(name)


def download_m18k(raw_root: Path, force: bool = False) -> None:
    """
    M18K V1 segmentasyon arşivini Google Drive'dan indirir ve ZIP ise açar.

    Args:
        raw_root: ``data/raw`` kökü.
        force: ``True`` ise tamamlanmış olsa bile yeniden indirir.
    """
    dest_dir = raw_root / "m18k"
    dest_dir.mkdir(parents=True, exist_ok=True)
    if _is_complete(dest_dir) and not force:
        logger.info("M18K zaten indirilmiş; atlanıyor (%s). Yeniden indirmek için --force kullanın.", dest_dir)
        return

    out_zip = dest_dir / "m18k_drive.zip"
    try:
        gdown.download(id=M18K_DRIVE_ID, output=str(out_zip), quiet=False)
    except Exception as exc:  # noqa: BLE001
        logger.exception("M18K gdown indirmesi başarısız: %s", exc)
        raise SystemExit(1) from exc

    if not out_zip.is_file() or out_zip.stat().st_size < 1024:
        logger.error("M18K indirme dosyası geçersiz veya boş: %s", out_zip)
        raise SystemExit(1)

    extract_root = dest_dir / "extracted"
    extract_root.mkdir(parents=True, exist_ok=True)
    try:
        extract_zip(out_zip, extract_root)
    except zipfile.BadZipFile as exc:
        logger.error("M18K arşivi ZIP değil veya bozuk: %s — %s", out_zip, exc)
        raise SystemExit(1) from exc

    _write_complete(dest_dir)
    logger.info("M18K hazır: %s", extract_root)


def download_oyster(raw_root: Path, force: bool = False) -> None:
    """
    Oyster maturity veri setini Mendeley'den indirir; ZIP/RAR ise açmayı dener.

    Args:
        raw_root: ``data/raw`` kökü.
        force: ``True`` ise tamamlanmış olsa bile yeniden indirir.
    """
    dest_dir = raw_root / "oyster"
    dest_dir.mkdir(parents=True, exist_ok=True)
    if _is_complete(dest_dir) and not force:
        logger.info("Oyster zaten indirilmiş; atlanıyor (%s).", dest_dir)
        return

    try:
        url, fname = _mendeley_download_url()
    except RuntimeError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    archive = dest_dir / fname.replace("/", "_")
    try:
        download_file(url, archive, desc=fname)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Oyster indirme hatası: %s", exc)
        raise SystemExit(1) from exc

    lower = archive.suffix.lower()
    extract_root = dest_dir / "extracted"
    if lower == ".zip":
        try:
            extract_zip(archive, extract_root)
        except Exception as exc:  # noqa: BLE001
            logger.error("ZIP açılamadı: %s", exc)
            raise SystemExit(1) from exc
    elif lower == ".rar":
        if not try_extract_rar(archive, extract_root):
            logger.warning(
                "Oyster arşivi RAR; otomatik çıkarma başarısız. 7-Zip veya ``pip install rarfile`` "
                "ve UnRAR ile deneyin; elle çıkardıktan sonra ``%s`` altında ``extracted`` beklenir.",
                dest_dir,
            )
            raise SystemExit(1)
    else:
        logger.warning(
            "Bilinmeyen arşiv uzantısı (%s). Dosyayı elle ``%s`` içine çıkarın; "
            "ardından ``%s`` dosyasını silip ``--force`` ile yeniden deneyebilirsiniz.",
            archive.suffix,
            extract_root,
            dest_dir / COMPLETE_NAME,
        )
        raise SystemExit(1)

    _write_complete(dest_dir)
    logger.info("Oyster indirme adımı tamam: %s", archive)


def main() -> None:
    """CLI giriş noktası."""
    parser = argparse.ArgumentParser(description="Harvest-Eye veri seti indirme")
    parser.add_argument("--only", choices=("m18k", "oyster", "all"), default="all", help="Hangi veri seti")
    parser.add_argument("--force", action="store_true", help="Tamamlanmış indirmeyi yine de yenile")
    args = parser.parse_args()

    root = _project_root()
    setup_dirs(root)
    raw_root = root / "data" / "raw"

    if args.only in ("m18k", "all"):
        download_m18k(raw_root, force=args.force)
    if args.only in ("oyster", "all"):
        download_oyster(raw_root, force=args.force)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
