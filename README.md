# Harvest-Eye / Mantar Hasat Tespiti

## Hızlı Başlangıç

### Gereksinimler

- Python 3.11+
- NVIDIA GPU (CPU'da da çalışır, yavaş)
- `models/detection/best.pt` (eğitimden sonra oluşur)

### Kurulum

```powershell
git clone https://github.com/mithrandir3010/harvest-eye
cd harvest-eye
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Kontrol listesi (ilk kurulum)

- [ ] `.venv/` oluşturuldu ve aktif
- [ ] `pip install` tamamlandı
- [ ] `models/detection/best.pt` mevcut (train veya indir)
- [ ] `mantar.jpg` / `mantar.mp4` yerelde mevcut (git'e alınmaz)

### Çalıştırma

```powershell
# Görüntü üzerinde test
python -m harvest_eye.pipeline.video_stream --source mantar.jpg

# Video üzerinde test
python -m harvest_eye.pipeline.video_stream --source mantar.mp4

# Testler
pytest tests/ -v

# Model eğitimi
python src\train_detection.py --model yolov8n --epochs 100
```

## Türkçe

**Harvest-Eye**, kültür mantarı görüntülerinde **tespit** (YOLOv8) ve **olgunluk sınıflandırması** (EfficientNet / timm, planlanan) için bir araştırma ve üretim iskeletidir. Windows ve NVIDIA GPU (CUDA 12.1) hedeflenir; CUDA yoksa betikler otomatik olarak CPU’ya düşer.

### Kurulum

1. Python 3.10+ ve [NVIDIA sürücüsü](https://www.nvidia.com/Download/index.aspx) kurulu olsun.
2. Sanal ortam önerilir:

```powershell
cd c:\Users\tommy\Desktop\harvest-eye
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Not:** `requirements.txt` içindeki PyTorch satırları `--extra-index-url` ile CUDA 12.1 tekerleklerini kullanır. Yalnızca CPU kurulumu istiyorsanız bu satırları [PyTorch kurulum sayfasından](https://pytorch.org/get-started/locally/) CPU sürümüyle değiştirin.

3. (İsteğe bağlı) Proje kökünde düzenlenebilir kurulum:

```powershell
pip install -e .
```

### Kullanım

```powershell
# Veri indirme (M18K Drive + Mendeley Oyster; Oyster arşivi RAR ise 7-Zip veya pip install rarfile + UnRAR gerekebilir)
python src\download_data.py

# M18K COCO -> YOLO; data/datasets.yaml üretir/günceller
python src\prepare_data.py
# Özel konumlar:
python src\prepare_data.py --source data\raw\m18k\dataset --dest data\processed

# Model 1 — YOLOv8 eğitimi
python src\train_detection.py --model yolov8n --epochs 100

# Tek görüntü tahmini
python src\inference.py --source path\to\image.jpg
```

`data/datasets.yaml` içindeki `path: processed`, dosyanın `data/` klasörüne göre `data/processed` anlamına gelir (YOLO’nun göreli yol çözümlemesi).

### Veri kaynakları ve atıf

- **M18K** (Google Drive, COCO segmentasyon / tespit): Zakeri ve ark. — depo ve sürümler: [abdollahzakeri/m18k](https://github.com/abdollahzakeri/m18k); makale: *Computers* 2025, [DOI 10.3390/computers14050199](https://doi.org/10.3390/computers14050199).
- **Annotated Oyster Mushroom Images** (Mendeley Data): Duman ve ark., [DOI 10.17632/hf55tkx489.1](https://doi.org/10.17632/hf55tkx489.1), [veri sayfası](https://data.mendeley.com/datasets/hf55tkx489/1).

### Model mimarisi

- **Model 1 — Tespit:** Ultralytics YOLOv8 (`yolov8n` / `s` / `m`), iki sınıf: `white_button`, `baby_bella`. Eğitim çıktıları `runs/detection/`; en iyi ağırlık `models/detection/best.pt`; Raspberry Pi hedefi için ONNX export `models/detection/best.onnx`.
- **Model 2 — Olgunluk:** timm üzerinde EfficientNet (veya eşdeğeri) ile Oyster veri kümesine göre sınıflandırma — `src/train_maturity.py` şu an yer tutucu; ham veri `data/raw/oyster/` altında.

### Yol haritası

1. M18K indirme ve YOLO hazırlığını stabilize etme; sınıf eşlemesini veri kümesi sürümlerine göre doğrulama.
2. Oyster RAR çıkarma ve etiket şemasına göre olgunluk veri hattı + `train_maturity.py`.
3. İki aşamalı boru hattı: tespit kırpması → olgunluk sınıflandırması.
4. ONNX/TensorRT ve saha testi (ör. Raspberry Pi / edge).

---

## English

**Harvest-Eye** is a research and engineering scaffold for **mushroom detection** (YOLOv8) and **maturity classification** (EfficientNet / timm, planned) on cultivated mushroom imagery. It targets **Windows + NVIDIA GPU (CUDA 12.1)**; scripts fall back to **CPU** when CUDA is unavailable.

### Setup

1. Use Python 3.10+ and an up-to-date NVIDIA driver.
2. Create a virtual environment and install dependencies:

```powershell
cd c:\Users\tommy\Desktop\harvest-eye
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

PyTorch lines assume CUDA 12.1 wheels via `--extra-index-url`. For CPU-only, replace them using the [official PyTorch install selector](https://pytorch.org/get-started/locally/).

3. Optional editable install: `pip install -e .`

### Usage

```powershell
python src\download_data.py
python src\prepare_data.py
python src\prepare_data.py --source data\raw\m18k\dataset --dest data\processed
python src\train_detection.py --model yolov8n --epochs 100
python src\inference.py --source path\to\image.jpg
```

In `data/datasets.yaml`, `path: processed` resolves to `data/processed` relative to the `data/` folder where the YAML lives.

### Dataset sources

- **M18K** — [GitHub](https://github.com/abdollahzakeri/m18k); citation: Zakeri et al., *Computers* 2025, [DOI 10.3390/computers14050199](https://doi.org/10.3390/computers14050199).
- **Oyster mushroom dataset** — [Mendeley Data](https://data.mendeley.com/datasets/hf55tkx489/1), [DOI 10.17632/hf55tkx489.1](https://doi.org/10.17632/hf55tkx489.1).

### Architecture

- **Model 1:** YOLOv8 detection, two classes. Weights under `models/detection/`; ONNX for edge deployment.
- **Model 2:** Planned EfficientNet-based maturity classifier on the Oyster dataset (`train_maturity.py` placeholder).

### Roadmap

Align M18K preprocessing with dataset revisions; build maturity pipeline from Oyster annotations; end-to-end harvest-readiness scoring; optimize exported models for edge hardware.
