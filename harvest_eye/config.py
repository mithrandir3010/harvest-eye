from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml

ROOT = Path(__file__).parent.parent


@dataclass
class DetectionConfig:
    weights: Path = ROOT / "models/detection/best.pt"
    conf: float = 0.25
    iou: float = 0.35
    device: str = "auto"
    imgsz: int = 640


@dataclass
class HarvestConfig:
    mode: str = "shape"
    circularity_thresh: float = 0.75
    aspect_thresh: float = 1.15
    harvest_mm: float = 40.0
    px_to_mm: float = 0.3


@dataclass
class ProjectionConfig:
    enabled: bool = False
    homography_path: Path = ROOT / "data/calibration/homography.npy"
    green_color: tuple = (0, 220, 0)
    red_color: tuple = (0, 0, 220)
    dot_radius: int = 8
    display_width: int = 1280
    display_height: int = 720


@dataclass
class HarvestEyeConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    harvest: HarvestConfig = field(default_factory=HarvestConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "HarvestEyeConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(
            detection=DetectionConfig(**data.get("detection", {})),
            harvest=HarvestConfig(**data.get("harvest", {})),
            projection=ProjectionConfig(**data.get("projection", {})),
        )

    def to_yaml(self, path: Path) -> None:
        import dataclasses
        with open(path, "w") as f:
            yaml.dump(dataclasses.asdict(self), f, default_flow_style=False)
