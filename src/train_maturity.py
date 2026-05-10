"""Model 2: EfficientNet tabanlı olgunluk sınıflandırması (yer tutucu / sonraki adım)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from utils import check_gpu, setup_dirs

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    """Proje kök dizini."""
    return Path(__file__).resolve().parent.parent


def main() -> None:
    """
    Olgunluk modeli eğitimi henüz uygulanmadı.

    Oyster (Mendeley) veri düzeni netleştikten sonra burada timm + EfficientNet
    ile sınıflandırıcı eklenecek.
    """
    parser = argparse.ArgumentParser(description="Model 2 — olgunluk (yakında)")
    parser.add_argument("--epochs", type=int, default=50, help="Gelecek: epoch sayısı")
    _ = parser.parse_args()

    root = _project_root()
    setup_dirs(root)
    check_gpu()

    oyster = root / "data" / "raw" / "oyster" / "extracted"
    if not oyster.is_dir():
        logger.warning("Oyster ham veri yok: %s — önce download_data.py çalıştırın.", oyster)

    logger.info(
        "train_maturity.py yer tutucu: EfficientNet + timm eğitimi bir sonraki sprintte; "
        "çıktı ağırlıklar ``models/maturity/`` altında saklanacak."
    )
    sys.exit(0)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
