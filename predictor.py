import csv
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from models import SimpleCNN
from xray_data_processor import XRayDataProcessor
from config import ProcessorSettings, PredictorSettings


class Predictor:
    def __init__(self, processor_settings: ProcessorSettings, base_model_cls: Callable[..., torch.nn.Module]):
        self.proc_settings = processor_settings
        self.base_model_cls = base_model_cls
        self.processor = XRayDataProcessor(processor_settings)

    def _load_model(self, ckpt_path: Path, device: torch.device):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        num_classes = ckpt.get("num_classes", len(ckpt.get("class_names", [])) or 3)
        in_ch = 3 if self.proc_settings.to_rgb else 1
        model = self.base_model_cls(in_channels=in_ch, num_classes=num_classes)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()
        return model, num_classes

    def _collect_images(self, images_path: Path) -> list[Path]:
        exts = {".jpg", ".jpeg", ".png"}
        return sorted([p for p in Path(images_path).rglob("*") if p.suffix.lower() in exts])

    def _infer(
        self,
        model: torch.nn.Module,
        device: torch.device,
        paths: Sequence[Path],
        num_classes: int,
        batch_size: int,
        write_label_idx: bool,
        write_probs: bool,
    ) -> list[list[Any]]:
        sm = torch.nn.Softmax(dim=1)
        out_rows: list[list[Any]] = []
        with torch.inference_mode():
            for i in range(0, len(paths), batch_size):
                chunk = paths[i : i + batch_size]
                batch = [self.processor.to_tensor_and_normalize(self.processor.open_and_resize(p)) for p in chunk]
                batch_t = torch.stack(batch).to(device)
                logits = model(batch_t)
                if logits.shape[1] != num_classes:
                    raise ValueError(
                        f"Model output classes ({logits.shape[1]}) != expected num_classes ({num_classes})"
                    )
                probs = sm(logits).cpu().numpy()
                preds = probs.argmax(axis=1)
                for path, pred0, prob_vec in zip(chunk, preds, probs):
                    pred1 = int(pred0) + 1  # shift 0→1, etc.
                    row = [str(path), pred1]
                    if write_label_idx:
                        row.append(int(pred0))
                    if write_probs:
                        # ensure length matches num_classes
                        if len(prob_vec) != num_classes:
                            raise ValueError("Probability vector length mismatch num_classes")
                        row += [float(x) for x in prob_vec]
                    out_rows.append(row)
        return out_rows

    def _write_csv(
        self, output_path: Path, rows: list[list[Any]], num_classes: int, write_label_idx: bool, write_probs: bool
    ) -> None:
        header = ["image_path", "pred_label"]
        if write_label_idx:
            header += ["pred_label_idx"]
        if write_probs:
            header += [f"prob_{i}" for i in range(num_classes)]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    def run(self, config: PredictorSettings) -> Path:
        device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model, num_classes = self._load_model(ckpt_path=config.model_path, device=device)
        img_paths = self._collect_images(images_path=config.images_root)
        rows = self._infer(
            model=model,
            device=device,
            paths=img_paths,
            num_classes=num_classes,
            batch_size=config.batch_size,
            write_label_idx=config.write_label_idx,
            write_probs=config.write_probs,
        )
        self._write_csv(
            output_path=config.output_path,
            rows=rows,
            num_classes=num_classes,
            write_label_idx=config.write_label_idx,
            write_probs=config.write_probs,
        )
        print(f"Wrote {len(rows)} rows to {config.output_path}")
        return config.output_path
