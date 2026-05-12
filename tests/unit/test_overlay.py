"""Overlay çizim testleri."""
import numpy as np
from harvest_eye.projection.overlay import draw_harvest_overlay
from harvest_eye.maturity.shape_classifier import ClassificationResult, HarvestDecision
from harvest_eye.detection.detector import Detection


def _make_result(decision: HarvestDecision) -> ClassificationResult:
    det = Detection(
        cx=0.5, cy=0.5, w=0.1, h=0.1,
        conf=0.9, class_id=0, class_name="mushroom",
        cx_px=320, cy_px=240, w_px=100, h_px=80,
    )
    return ClassificationResult(
        detection=det,
        decision=decision,
        circularity=0.7,
        aspect_ratio=1.2,
        confidence=0.8,
    )


def test_overlay_adds_header():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = [
        _make_result(HarvestDecision.HARVEST),
        _make_result(HarvestDecision.WAIT),
    ]
    out = draw_harvest_overlay(frame, results)
    # Başlık şeridi (48px) eklenince yükseklik artar
    assert out.shape[0] == 480 + 48
    assert out.shape[1] == 640


def test_overlay_empty_results():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = draw_harvest_overlay(frame, [])
    assert out.shape[0] == 480 + 48
