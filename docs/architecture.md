# Harvest Eye — Mimari

## Bileşenler
- **detection/** — YOLOv8 nesne tespiti
- **maturity/** — Hasat olgunluk kararı
- **projection/** — Projektör kontrol ve kalibrasyon
- **pipeline/** — Uçtan uca işleme akışı
- **io/** — Kamera ve görselleştirme

## Pipeline Akışı
Kamera → YOLOv8 → Hasat kararı → Homografi → Projektör
