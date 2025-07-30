from pathlib import Path

from pydantic_settings import BaseSettings


class PredictorSettings(BaseSettings):
    model_path: Path
    images_root: Path
    output_path: Path
    batch_size: int = 32
    write_probs: bool = False
    device: str | None = None
