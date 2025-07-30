from pathlib import Path

import torch

from trainer import Trainer
from models import SimpleCNN
from predictor import Predictor
from experiment_runner import ExperimentRunner
from xray_data_processor import XRayDataProcessor, AugName
from config import ProcessorSettings, TrainerSettings, ExperimentConfig, PredictorSettings


def train_model(dataset, processor_settings: ProcessorSettings, trainer_settings: TrainerSettings):
    print("Creating model...")
    num_classes = len({lbl.item() for _, lbl in dataset})
    model = SimpleCNN(in_channels=3 if processor_settings.to_rgb else 1, num_classes=num_classes)

    trainer = Trainer(model, trainer_settings, num_classes=num_classes)

    print("Starting training...")
    results = trainer.fit(dataset)
    print("Training summary:", results)


def load_model(device: torch.device, trainer_settings: TrainerSettings):

    ckpt_path = trainer_settings.ckpt_dir / "best_model.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    num_classes = len(ckpt["class_names"])
    model = SimpleCNN(in_channels=3, num_classes=num_classes)  # match what you trained
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    return model


def train_simple_cnn():
    root = Path("data/train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    proc_settings = ProcessorSettings(root_dir=root)
    processor = XRayDataProcessor(proc_settings)
    trainer_settings = TrainerSettings()

    # checkpoint = torch.load(ckpt_path, map_location="cpu")
    ckpt_path = trainer_settings.ckpt_dir / "best_model.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    num_classes = len(ckpt["class_names"])
    model = SimpleCNN(in_channels=3, num_classes=num_classes)  # match what you trained
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    img_path = Path("data/train/03/012.jpeg")
    # Reuse processor preprocess to stay consistent
    pil_img = processor.open_and_resize(img_path)
    tensor = processor.to_tensor_and_normalize(pil_img)

    with torch.inference_mode():
        logits = model(tensor.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1)[0].cpu()
        pred_idx = int(probs.argmax())
        pred_name = ckpt["class_names"][pred_idx]

    print("Predicted:", pred_name, "(id:", pred_idx, ")")
    print("Probs:", probs.tolist())

    print("Done. Launch TensorBoard with: \n\n    tensorboard --logdir", trainer_settings.log_dir, "\n")


def train_simple_cnn_multiple_augs():
    root = Path("data/train")
    proc_set = ProcessorSettings(root_dir=root)

    AUG_SETS: list[list[AugName]] = [
        ["hflip", "rotate", "translate"],
        ["hflip", "rotate", "translate", "noise"],
        ["hflip", "rotate", "translate", "brightness_contrast"],
        ["hflip", "rotate", "translate", "clahe"],
        ["hflip", "rotate", "translate", "noise", "brightness_contrast", "clahe"],
    ]

    configs = [ExperimentConfig(run_name=f"aug_{i}", augmentations=augs) for i, augs in enumerate(AUG_SETS)]

    runner = ExperimentRunner(
        proc_settings=proc_set, base_model_cls=SimpleCNN, trainer_defaults=TrainerSettings(k_folds=5, val_ratio=0.2)
    )

    summary = runner.run_all(configs)
    print(summary["csv_path"])


def predict():
    proc_cfg = ProcessorSettings(root_dir=Path("data/train"))
    settings = PredictorSettings(
        model_path=Path("checkpoints/SimpleCNN/best_model.pth"),
        images_root=Path("data/train"),
        output_path=Path("predictions.csv"),
        write_probs=True,
    )
    pred = Predictor(proc_cfg, base_model_cls=SimpleCNN)
    pred.run(settings)


if __name__ == "__main__":
    train_simple_cnn_multiple_augs()
