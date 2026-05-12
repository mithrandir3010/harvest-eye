"""ShapeClassifier testleri."""
import numpy as np
import pytest
from harvest_eye.maturity.shape_classifier import (
    ShapeClassifier,
    HarvestDecision,
)
from harvest_eye.detection.detector import Detection
from harvest_eye.config import HarvestConfig


def _make_detection(w_px: int = 100, h_px: int = 80) -> Detection:
    return Detection(
        cx=0.5, cy=0.5, w=0.1, h=0.1,
        conf=0.9, class_id=0, class_name="mushroom",
        cx_px=320, cy_px=240,
        w_px=w_px, h_px=h_px,
    )


def test_classifier_returns_decision():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    clf = ShapeClassifier()
    det = _make_detection()
    result = clf.classify(frame, det)
    assert result.decision in (HarvestDecision.HARVEST, HarvestDecision.WAIT)


def test_wide_bbox_harvested():
    """Geniş bounding box (yatay yayılmış) hasat kararı almalı."""
    cfg = HarvestConfig(aspect_thresh=1.1)
    clf = ShapeClassifier(cfg)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det = _make_detection(w_px=200, h_px=80)  # aspect = 2.5 > 1.1
    result = clf.classify(frame, det)
    assert result.aspect_ratio == pytest.approx(200 / 80, abs=0.01)


def test_classify_batch():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    clf = ShapeClassifier()
    dets = [_make_detection(), _make_detection(120, 90)]
    results = clf.classify_batch(frame, dets)
    assert len(results) == 2
