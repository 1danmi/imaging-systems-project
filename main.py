from pathlib import Path
from typing import Literal

from predictor import Predictor
from models import SimpleCNN, ResNet18, CapsNet
from config import ProcessorSettings, PredictorSettings

ModelName = Literal["simplecnn", "resnet18", "capsnet"]

model_to_class: dict[ModelName, type] = {"simplecnn": SimpleCNN, "resnet18": ResNet18, "capsnet": CapsNet}
model_to_weights_path: dict[ModelName, Path] = {
    "simplecnn": Path("trained_models/best_model_SimpleCNN.pth"),
    "resnet18": Path("trained_models/best_model_ResNet18.pth"),
    "capsnet": Path("trained_models/best_model_CapsNet.pth"),
}


def predict(images_path: Path | str, model_name: ModelName, output_path: Path | str):
    proc_cfg = ProcessorSettings(root_dir=Path(images_path))
    settings = PredictorSettings(
        model_path=model_to_weights_path[model_name],
        images_root=Path(images_path),
        output_path=Path(output_path),
        write_label_idx=False,
        write_probs=False,
    )
    pred = Predictor(proc_cfg, base_model_cls=model_to_class[model_name])
    pred.run(settings)


def main():
    model_idx = input(
        """Please select model:
    1. SimpleCNN
    2. ResNet18
    3. CapsNet
"""
    )
    match model_idx:
        case "1":
            model_name: ModelName = "simplecnn"
        case "2":
            model_name: ModelName = "resnet18"
        case "3":
            model_name: ModelName = "capsnet"
        case _:
            raise ValueError("Invalid model selection")

    images_path = input("Please enter path to images folder: ")
    images_path = images_path or "data/train"
    output_path = input("Please enter path to output file: ")
    output_path = output_path or f"predictions/predictions_{model_name}.csv"
    print("Predicting...")
    predict(images_path=images_path, model_name=model_name, output_path=output_path)
    print("Done.")


if __name__ == "__main__":
    main()
