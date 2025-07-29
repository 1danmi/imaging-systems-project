import csv
import json
import datetime as dt
from pathlib import Path
from typing import Any, Callable

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from trainer import Trainer
from xray_data_processor import XRayDataProcessor
from config import ProcessorSettings, TrainerSettings, ExperimentConfig
from torch.utils.data import DataLoader


class ExperimentRunner:
    def __init__(
        self,
        proc_settings: ProcessorSettings,
        base_model_cls: Callable[..., torch.nn.Module],
        trainer_defaults: TrainerSettings | None = None,
        experiments_root: Path = Path("experiments"),
    ):
        self.proc_settings = proc_settings
        self.base_model_cls = base_model_cls
        self.trainer_defaults = trainer_defaults or TrainerSettings(k_folds=5)
        self.experiments_root = experiments_root
        self.experiments_root.mkdir(parents=True, exist_ok=True)

        self.processor = XRayDataProcessor(self.proc_settings)
        self.processor.scan()

        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.experiments_root / f"session_{ts}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.mapping_path = self.session_dir / "model_mapping.json"
        self.csv_path = self.session_dir / f"results_{ts}.csv"
        self.html_path = self.session_dir / f"summary_{ts}.html"

        self.mapping: dict[str, str] = {}
        self.rows: list[dict[str, Any]] = []

    @staticmethod
    def _confusion_over_folds(trainer: Trainer, dataset, settings: TrainerSettings):
        targets = np.array([dataset[i][1] for i in range(len(dataset))])
        skf = StratifiedKFold(n_splits=settings.k_folds, shuffle=True, random_state=settings.seed)

        all_true = []
        all_pred = []
        device = trainer.device
        model = trainer.model.to(device)
        model.eval()

        with torch.no_grad():
            for train_idx, val_idx in skf.split(np.zeros(len(targets)), targets):
                for idx in val_idx:
                    x, y = dataset[idx]
                    x = x.unsqueeze(0).to(device)
                    logits = model(x)
                    pred = logits.argmax(dim=1).item()
                    all_true.append(int(y))
                    all_pred.append(pred)

        cm = confusion_matrix(all_true, all_pred, labels=list(range(len(np.unique(targets)))))
        acc = (np.array(all_true) == np.array(all_pred)).mean()
        return cm, float(acc)

    @staticmethod
    def _save_confusion_matrix(cm: np.ndarray, path: Path) -> Path:
        fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, int(val), ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
        return path

    def _write_html_summary(self, df: pd.DataFrame, best_run: str | None) -> None:
        html = [
            "<html><head><meta charset='utf-8'><title>Experiment Summary</title></head><body>",
            f"<h1>Experiment session – {dt.datetime.now().isoformat()}</h1>",
        ]
        if best_run:
            html.append(f"<h2>Best run: {best_run}</h2>")
        html.append(df.to_html(index=False, escape=False))
        html.append("</body></html>")
        self.html_path.write_text("\n".join(html), encoding="utf-8")

    def _run_single(self, exp_cfg: ExperimentConfig) -> dict[str, Any]:
        print(f"\n=== Running experiment: {exp_cfg.run_name} ===")

        records = self.processor.records
        targets = np.array([lbl for _, lbl in records])
        num_classes = len(np.unique(targets))

        trainer_kwargs = self.trainer_defaults.model_dump(exclude={"ckpt_dir", "log_dir"})
        trainer_kwargs.update(exp_cfg.trainer_overrides)
        trainer_settings = TrainerSettings(**trainer_kwargs)

        run_dir = self.session_dir / exp_cfg.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        trainer_settings.log_path = run_dir / "tb"
        trainer_settings.ckpt_path = run_dir / "ckpt"

        skf = StratifiedKFold(n_splits=trainer_settings.k_folds, shuffle=True, random_state=trainer_settings.seed)

        in_ch = 3 if self.proc_settings.to_rgb else 1

        fold_results = []
        val_datasets = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets), start=1):
            train_records = [records[i] for i in train_idx]
            val_records = [records[i] for i in val_idx]
            train_ds = self.processor.make_dataset(exp_cfg.augmentations, train_records, persist=True)
            val_ds = self.processor.make_dataset(["none"], val_records, persist=True)
            val_datasets.append(val_ds)

            model = self.base_model_cls(in_channels=in_ch, num_classes=num_classes)
            trainer = Trainer(model, trainer_settings, num_classes=num_classes)
            res = trainer._train_loop(train_ds, val_ds, fold_id=fold)
            fold_results.append(res)

        val_losses = [f["best_val_loss"] for f in fold_results]
        val_accs = [f.get("best_val_acc", 0.0) for f in fold_results]
        best_epochs = [f["best_epoch"] for f in fold_results]

        # Evaluate best models on their validation sets
        all_true = []
        all_pred = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for res, val_ds in zip(fold_results, val_datasets):
            ckpt = torch.load(res["ckpt_path"], map_location=device)
            model = self.base_model_cls(in_channels=in_ch, num_classes=num_classes)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device).eval()
            loader = DataLoader(
                val_ds,
                batch_size=trainer_settings.batch_size,
                shuffle=False,
                num_workers=trainer_settings.num_workers,
                pin_memory=trainer_settings.pin_memory,
            )
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(device)
                    logits = model(x)
                    preds = logits.argmax(dim=1).cpu().tolist()
                    all_pred.extend(preds)
                    all_true.extend(y.tolist())

        cm = confusion_matrix(all_true, all_pred, labels=list(range(num_classes)))
        cm_path = self._save_confusion_matrix(cm, run_dir / "confusion_matrix.png")

        best_fold_idx = int(np.argmin(val_losses))
        best_ckpt = fold_results[best_fold_idx]["ckpt_path"]
        self.mapping[exp_cfg.run_name] = str(best_ckpt)

        row = {
            "run_name": exp_cfg.run_name,
            "augmentations": "+".join(exp_cfg.augmentations),
            "notes": exp_cfg.notes,
            "val_loss_mean": float(np.mean(val_losses)),
            "val_loss_std": float(np.std(val_losses)),
            "best_epoch_mean": float(np.mean(best_epochs)),
            "val_acc_mean": float(np.mean(val_accs)),
            "ckpt_path": str(best_ckpt),
            "cm_path": str(cm_path),
        }

        for k in ["lr", "batch_size", "epochs", "optimizer", "weight_decay", "early_stop_patience", "k_folds"]:
            row[k] = trainer_kwargs.get(k)
        return row

    def run_all(self, configs: list[ExperimentConfig]) -> dict[str, Any]:
        best_acc = -1.0
        best_run_name = None

        for cfg in configs:
            res = self._run_single(cfg)
            self.rows.append(res)
            # Track best by mean_val_acc
            if res.get("val_acc_mean", -1) > best_acc:
                best_acc = res["val_acc_mean"]
                best_run_name = res["run_name"]

        df = pd.DataFrame(self.rows)
        df.to_csv(self.csv_path, index=False)

        with open(self.mapping_path, "w") as f:
            json.dump(self.mapping, f, indent=2)

        self._write_html_summary(df, best_run_name)

        return {
            "csv_path": self.csv_path,
            "html_path": self.html_path,
            "mapping_path": self.mapping_path,
            "best_run_name": best_run_name,
        }

    def predict_folder(self, best_run_name: str, images_dir: Path, output_csv: Path) -> None:
        """Load a saved model by run name and predict all images in a folder."""
        if not self.mapping:
            if self.mapping_path.exists():
                self.mapping = json.loads(self.mapping_path.read_text())
            else:
                raise FileNotFoundError("Mapping JSON not found. Run experiments first.")

        if best_run_name not in self.mapping:
            raise KeyError(f"Run name {best_run_name} not in mapping.")

        ckpt = torch.load(self.mapping[best_run_name], map_location="cpu")

        in_ch = 3 if self.proc_settings.to_rgb else 1
        num_classes = len(ckpt.get("class_names", [])) or 3
        model = self.base_model_cls(in_channels=in_ch, num_classes=num_classes)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Preprocess images with same processor settings (no aug, just resize/normalize)
        paths = []
        preds0 = []
        preds1 = []
        probs_all = []

        for img_path in sorted(images_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                continue
            pil = self.processor.open_and_resize(img_path)
            ten = self.processor.to_tensor_and_normalize(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(ten)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred0 = int(probs.argmax())
                pred1 = pred0 + 1  # shift to 1..3
            paths.append(str(img_path))
            preds0.append(pred0)
            preds1.append(pred1)
            probs_all.append(probs.tolist())

        # Save CSV
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["image_path", "pred_label0", "pred_label1"] + [f"prob_{i}" for i in range(num_classes)]
            writer.writerow(header)
            for p, a0, a1, pr in zip(paths, preds0, preds1, probs_all):
                writer.writerow([p, a0, a1] + pr)
        print(f"Saved predictions to {output_csv}")
