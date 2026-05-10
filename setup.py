"""Harvest-Eye: kurulum meta verisi (isteğe bağlı editable install)."""

from __future__ import annotations

from pathlib import Path

from setuptools import setup

ROOT = Path(__file__).resolve().parent


def _read_requirements() -> list[str]:
    """Parse requirements.txt; skip comment lines and --extra-index-url."""
    lines: list[str] = []
    req = ROOT / "requirements.txt"
    if not req.is_file():
        return lines
    for raw in req.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("--"):
            continue
        lines.append(line)
    return lines


setup(
    name="harvest-eye",
    version="0.1.0",
    description="Mushroom harvest detection (YOLOv8) and maturity classification",
    python_requires=">=3.10",
    install_requires=_read_requirements(),
    packages=[],
)
