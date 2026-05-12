"""Config modülü testleri."""
import pytest
from harvest_eye.config import (
    HarvestEyeConfig,
    DetectionConfig,
    HarvestConfig,
    ProjectionConfig,
)


def test_default_config():
    cfg = HarvestEyeConfig()
    assert cfg.detection.conf == 0.25
    assert cfg.harvest.mode == "shape"
    assert cfg.projection.enabled is False


def test_detection_config_defaults():
    cfg = DetectionConfig()
    assert 0 < cfg.conf < 1
    assert cfg.device == "auto"


def test_harvest_config_thresholds():
    cfg = HarvestConfig()
    assert 0 < cfg.circularity_thresh < 1
    assert cfg.aspect_thresh > 1.0


def test_config_from_yaml(tmp_path):
    yaml_content = """
detection:
  conf: 0.5
harvest:
  circularity_thresh: 0.8
projection:
  enabled: true
"""
    f = tmp_path / "test.yaml"
    f.write_text(yaml_content)
    cfg = HarvestEyeConfig.from_yaml(f)
    assert cfg.detection.conf == 0.5
    assert cfg.harvest.circularity_thresh == 0.8
    assert cfg.projection.enabled is True
