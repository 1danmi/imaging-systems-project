from pathlib import Path

from pydantic_settings import BaseSettings


class ProcessorSettings(BaseSettings):
    root_dir: Path
    cache_dir: Path | None = None

    output_size: tuple[int, int] = (224, 224)
    to_rgb: bool = True

    normalize_mean: list[float] = [0.485, 0.456, 0.406]
    normalize_std: list[float] = [0.229, 0.224, 0.225]

    # Augmentation hyperparameters
    max_rotation: float = 10.0
    max_translate: float = 0.05
    brightness_factor: float = 0.1
    contrast_factor: float = 0.1
    noise_sigma: float = 0.01
    clahe_clip: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)

    persist_png: bool = True
    denylist_ext: list[str] = [".txt", ".json"]
