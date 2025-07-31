from typing import Any, Literal

from pydantic_settings import BaseSettings

AugName = Literal["none", "hflip", "rotate", "translate", "brightness_contrast", "noise", "clahe"]


class ExperimentConfig(BaseSettings):
    run_name: str
    augmentations: list[AugName]
    trainer_overrides: dict[str, Any] = {}
    notes: str = ""
