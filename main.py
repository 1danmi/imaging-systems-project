from pathlib import Path
from typing import Literal


from predictor import Predictor
from models import SimpleCNN, ResNet18, CapsNet
from config import ProcessorSettings, PredictorSettings

ModelName = Literal["simplecnn", "resnet18", "capsnet"]

model_to_class: dict[ModelName, type] = {"simplecnn": SimpleCNN, "resnet18": ResNet18, "capsnet": CapsNet}


def predict(images_path: Path | str, model_weights_path: Path | str, model: ModelName, output_path: Path | str):
    proc_cfg = ProcessorSettings(root_dir=Path(images_path))
    settings = PredictorSettings(
        model_path=Path(model_weights_path),
        images_root=Path(images_path),
        output_path=Path(output_path),
        write_probs=True,
    )
    pred = Predictor(proc_cfg, base_model_cls=model_to_class[model])
    pred.run(settings)


def main():
    input("""Please select model:""")


if __name__ == "__main__":
    main()
