from pathlib import Path

import torch

from trainer import Trainer
from models import CapsNet
from predictor import Predictor
from experiment_runner import ExperimentRunner
from xray_data_processor import XRayDataProcessor, AugName
from config import ProcessorSettings, TrainerSettings, ExperimentConfig, PredictorSettings


def train_capsnet():
    root = Path("data/train")
    proc_settings = ProcessorSettings(root_dir=root)
    processor = XRayDataProcessor(proc_settings)
    trainer_settings = TrainerSettings(model_name="CapsNet")

    dataset = processor.make_dataset(["none"], persist=True)
    num_classes = len({lbl.item() for _, lbl in dataset})
    model = CapsNet(in_channels=3 if proc_settings.to_rgb else 1, num_classes=num_classes)

    trainer = Trainer(model, trainer_settings, num_classes=num_classes)
    results = trainer.fit(dataset)
    print("Training summary:", results)


def train_capsnet_multiple_augs():
    root = Path("data/train")
    proc_set = ProcessorSettings(root_dir=root)

    AUG_SETS: list[list[AugName]] = [
        [],
        ["hflip", "rotate", "translate"],
        ["hflip", "rotate", "translate", "noise"],
        ["hflip", "rotate", "translate", "brightness_contrast"],
        ["hflip", "rotate", "translate", "clahe"],
        ["hflip", "rotate", "translate", "noise", "brightness_contrast", "clahe"],
    ]

    configs = [ExperimentConfig(run_name=f"aug_{i}", augmentations=augs) for i, augs in enumerate(AUG_SETS)]

    defaults = TrainerSettings(model_name="CapsNet", k_folds=5, val_ratio=0.2)
    runner = ExperimentRunner(proc_settings=proc_set, base_model_cls=CapsNet, trainer_defaults=defaults)

    summary = runner.run_all(configs)
    print(summary["csv_path"])


def predict_capsnet():
    proc_cfg = ProcessorSettings(root_dir=Path("data/train"))
    settings = PredictorSettings(
        model_path=Path("checkpoints/CapsNet/best_model.pth"),
        images_root=Path("data/train"),
        output_path=Path("predictions_capsnet.csv"),
        write_probs=True,
    )
    pred = Predictor(proc_cfg, base_model_cls=CapsNet)
    pred.run(settings)


if __name__ == "__main__":
    predict_capsnet()
