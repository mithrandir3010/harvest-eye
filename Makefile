.PHONY: install test lint format run help

help:
	@echo Harvest Eye - Komutlar:
	@echo   make install   - Paketi kur
	@echo   make test      - Testleri calistir
	@echo   make lint      - Kod kalitesi kontrol
	@echo   make format    - Kodu formatla
	@echo   make run       - Ornek calistir

install:
	pip install -e .
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=harvest_eye --cov-report=term-missing

lint:
	ruff check harvest_eye/ scripts/

format:
	black harvest_eye/ scripts/ tests/

run:
	python -m harvest_eye.pipeline.video_stream --source mantar.jpg
