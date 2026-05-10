"""M18K COCO anotasyonlarını YOLO formatına dönüştürür ve ``data/processed`` düzenini üretir."""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from utils import coco_to_yolo, seg_to_bbox, setup_dirs

logger = logging.getLogger(__name__)

CLASS_NAMES: tuple[str, ...] = ("mushroom",)


def _project_root() -> Path:
    """Proje kök dizini."""
    return Path(__file__).resolve().parent.parent


def _default_m18k_extracted(root: Path) -> Path:
    """İndirme betiğinin ürettiği varsayılan çıkarma kökü."""
    return root / "data" / "raw" / "m18k" / "extracted"


def _find_coco_json(search_root: Path) -> Path:
    """
    COCO biçimli JSON dosyasını arar.

    Args:
        search_root: Tarama başlangıç dizini.

    Returns:
        Seçilen JSON dosya yolu.

    Raises:
        FileNotFoundError: Uygun dosya bulunamazsa.
    """
    candidates: list[Path] = []
    for p in search_root.rglob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and "images" in data and "annotations" in data and "categories" in data:
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"COCO JSON bulunamadı: {search_root}")
    candidates.sort(key=lambda x: x.stat().st_size, reverse=True)
    chosen = candidates[0]
    logger.info("COCO dosyası: %s", chosen)
    return chosen


def _build_category_map(categories: list[dict[str, Any]]) -> dict[int, int]:
    """
    Tüm COCO ``category_id`` değerlerini tek YOLO sınıfına (0: mushroom) eşler.

    Önceki çok sınıflı etiketler (ör. id 1) dahil hepsi 0 olarak yazılır.

    Args:
        categories: COCO ``categories`` listesi.

    Returns:
        ``{coco_category_id: 0}``

    Raises:
        ValueError: Kategori listesi boşsa.
    """
    if not categories:
        raise ValueError("Kategori listesi boş.")
    return {int(c["id"]): 0 for c in categories}


def _resolve_image_path(extract_root: Path, file_name: str) -> Path | None:
    """
    COCO ``file_name`` için yerel görüntü yolunu çözer.

    Args:
        extract_root: Veri seti çıkarıldığı üst dizin.
        file_name: COCO içindeki göreli dosya yolu veya ad.

    Returns:
        Var olan dosya yolu veya ``None``.
    """
    direct = extract_root / file_name
    if direct.is_file():
        return direct
    base = Path(file_name).name
    if "depth" in base.lower():
        return None
    for p in extract_root.rglob(base):
        if p.is_file() and "depth" not in str(p).lower():
            return p
    return None


def _is_rgb_path(path: Path) -> bool:
    """Derinlik haritası yollarını ele."""
    s = path.name.lower()
    return "depth" not in s and "disparity" not in s


def _flatten_rel_name(file_name: str) -> str:
    """Çıktı dosya adı çakışmalarını azaltmak için göreli yolu düzleştirir."""
    parts = file_name.replace("\\", "/").split("/")
    stem = "__".join(parts)
    if not stem.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        stem += ".jpg"
    return stem


def _annotation_to_yolo_line(
    ann: dict[str, Any],
    img_w: int,
    img_h: int,
    cat_map: dict[int, int],
) -> tuple[int, float, float, float, float] | None:
    """
    Tek COCO annotation kaydını YOLO koordinatlarına çevirir.

    Returns:
        Tuple veya görünür kutu yoksa ``None``.
    """
    cid = int(ann["category_id"])
    if cid not in cat_map:
        yolo_c = 0
    else:
        yolo_c = cat_map[cid]
    bbox = ann.get("bbox")
    if not bbox or len(bbox) != 4:
        seg = ann.get("segmentation")
        if not seg:
            return None
        if isinstance(seg, list) and seg and isinstance(seg[0], list):
            poly = seg[0]
        elif isinstance(seg, list):
            poly = seg
        else:
            return None
        bbox = list(seg_to_bbox(poly))
    fake_ann = {"bbox": bbox, "class_id": yolo_c}
    return coco_to_yolo(fake_ann, img_w, img_h)


def _clear_label_caches(processed_root: Path) -> None:
    """``.cache`` dosyalarını işlenmiş veri ağacından siler."""
    removed = 0
    for p in processed_root.rglob("*.cache"):
        try:
            p.unlink()
            removed += 1
        except OSError as exc:
            logger.warning("Önbellek silinemedi %s: %s", p, exc)
    if removed:
        logger.info("%d adet .cache dosyası silindi.", removed)


def write_datasets_yaml(root: Path) -> None:
    """
    ``data/datasets.yaml`` dosyasını YOLO eğitimi için yazar.

    Args:
        root: Proje kökü.
    """
    yaml_path = root / "data" / "datasets.yaml"
    processed_abs = (root / "data" / "processed").resolve()
    content = (
        f"path: '{processed_abs}'\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "nc: 1\n"
        "names: ['mushroom']\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    logger.info("YOLO veri yapılandırması: %s", yaml_path)


def prepare_m18k(m18k_root: Path, processed_root: Path, seed: int = 42) -> None:
    """
    M18K COCO verisini ``train`` / ``val`` / ``test`` YOLO düzenine dönüştürür.

    Args:
        m18k_root: Çıkarılmış arşiv kökü (içinde görseller + JSON).
        processed_root: ``data/processed`` hedefi.
        seed: Bölme için rastgele tohum.
    """
    coco_path = _find_coco_json(m18k_root)
    data = json.loads(coco_path.read_text(encoding="utf-8", errors="ignore"))
    cat_map = _build_category_map(list(data["categories"]))

    images_by_id: dict[int, dict[str, Any]] = {int(i["id"]): i for i in data["images"]}
    anns_by_image: dict[int, list[dict[str, Any]]] = {}
    for a in data["annotations"]:
        iid = int(a["image_id"])
        anns_by_image.setdefault(iid, []).append(a)

    usable: list[int] = []
    for iid, info in images_by_id.items():
        fn = str(info["file_name"])
        p = _resolve_image_path(m18k_root, fn)
        if p is None or not _is_rgb_path(p):
            continue
        usable.append(iid)

    random.seed(seed)
    random.shuffle(usable)
    n = len(usable)
    if n == 0:
        raise RuntimeError("Kullanılabilir RGB görüntü bulunamadı.")

    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    if n_train < 1:
        n_train = 1
    if n_val < 1 and n > 2:
        n_val = 1
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)
    train_ids = set(usable[:n_train])
    val_ids = set(usable[n_train : n_train + n_val])
    test_ids = set(usable[n_train + n_val :])

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}

    for sp, ids in splits.items():
        (processed_root / sp / "images").mkdir(parents=True, exist_ok=True)
        (processed_root / sp / "labels").mkdir(parents=True, exist_ok=True)

    class_counts: Counter[int] = Counter()
    total_ann = 0

    for sp, id_set in splits.items():
        for iid in id_set:
            info = images_by_id[iid]
            fn = str(info["file_name"])
            src = _resolve_image_path(m18k_root, fn)
            if src is None:
                continue
            w, h = int(info["width"]), int(info["height"])
            out_name = _flatten_rel_name(fn)
            dst_img = processed_root / sp / "images" / out_name
            dst_lbl = processed_root / sp / "labels" / (Path(out_name).stem + ".txt")
            shutil.copy2(src, dst_img)

            lines: list[str] = []
            for ann in anns_by_image.get(iid, []):
                row = _annotation_to_yolo_line(ann, w, h, cat_map)
                if row is None:
                    continue
                c, xc, yc, ww, hh = row
                class_counts[c] += 1
                total_ann += 1
                lines.append(f"{c} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
            dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    _clear_label_caches(processed_root)
    write_datasets_yaml(_project_root())

    def _count_images(sub: str) -> int:
        return len(list((processed_root / sub / "images").glob("*")))

    logger.info("--- M18K -> YOLO özeti ---")
    logger.info("Train görüntü: %d", _count_images("train"))
    logger.info("Val görüntü: %d", _count_images("val"))
    logger.info("Test görüntü: %d", _count_images("test"))
    logger.info("Toplam annotation (kutu): %d", total_ann)
    logger.info("Sınıf dağılımı (YOLO id): %s", dict(class_counts))
    for i, name in enumerate(CLASS_NAMES):
        logger.info("  %s (id=%d): %d", name, i, class_counts.get(i, 0))


def _resolve_under_root(path: Path | None, root: Path, default: Path) -> Path:
    """
    CLI yolunu çözer: mutlak değilse ``root`` altında birleştirir.

    Args:
        path: Kullanıcı girdisi veya ``None``.
        root: Proje kökü.
        default: ``path`` boşsa kullanılacak varsayılan.

    Returns:
        Mutlak, normalize edilmiş ``Path``.
    """
    if path is None:
        return default.resolve()
    p = Path(path)
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser(description="M18K -> YOLO veri hazırlığı")
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="M18K COCO kök dizini (örn. data/raw/m18k/dataset)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="İşlenmiş YOLO çıktısı (varsayılan: data/processed)",
    )
    parser.add_argument(
        "--m18k-root",
        type=Path,
        default=None,
        help="[Eski ad] --source ile aynı; her ikisi verilirse --source kullanılır",
    )
    args = parser.parse_args()

    root = _project_root()
    setup_dirs(root)

    src_arg = args.source if args.source is not None else args.m18k_root
    m18k_root = _resolve_under_root(
        src_arg,
        root,
        _default_m18k_extracted(root),
    )
    processed = _resolve_under_root(
        args.dest,
        root,
        root / "data" / "processed",
    )

    if not m18k_root.is_dir():
        logger.error(
            "M18K dizini yok: %s — yolu kontrol edin veya önce src/download_data.py çalıştırın.",
            m18k_root,
        )
        raise SystemExit(1)

    prepare_m18k(m18k_root, processed)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
