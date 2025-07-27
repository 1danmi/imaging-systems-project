from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainerSettings(BaseSettings):
    model_name: str = "SimpleCNN"

    batch_size: int = 16
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"  # "sgd"
    momentum: float = 0.9

    early_stop_patience: int = 5
    k_folds: int = 1
    val_ratio: float = 0.2

    class_weighted_loss: bool = True

    num_workers: int = 4 # max(os.cpu_count() - 1, 0) if os.cpu_count() else 0
    pin_memory: bool = True
    amp: bool = True

    seed: int | None = 42

    log_path: Path = Path("logs")
    ckpt_path: Path = Path("checkpoints")
    save_best_only: bool = True

    @computed_field
    @property
    def log_dir(self) -> Path:
        return self.log_path / self.model_name

    @computed_field
    @property
    def ckpt_dir(self) -> Path:
        return self.ckpt_path / self.model_name


