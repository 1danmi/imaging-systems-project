from typing import Any

from pydantic_settings import BaseSettings

from xray_data_processor import AugName


class ExperimentConfig(BaseSettings):
    run_name: str
    augmentations: list[AugName]
    trainer_overrides: dict[str, Any] = {}
    notes: str = ""
