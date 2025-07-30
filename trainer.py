import time
import random
from typing import Any
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold

from config.trainer_settings import TrainerSettings


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainerSettings,
        num_classes: int,
        class_names: list[str] | None = None,
        device: torch.device | None = None,
    ):
        self.model = model
        self.config = config
        self.num_classes = num_classes
        self.class_names = list(class_names) if class_names else [str(i) for i in range(num_classes)]

        # Use GPU if available
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Seeds for reproducibility
        if self.config.seed is not None:
            self._set_seed(self.config.seed)

        # Save initial weights to reset between folds
        self._initial_state = deepcopy(model.state_dict())

        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self.config.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.config.log_path))
        # Use new torch.amp API; select device if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scaler = torch.amp.GradScaler(device=device, enabled=self.config.amp)

    def _fit_single_split(self, dataset: Dataset) -> dict[str, Any]:
        # Split train/val
        n = len(dataset)
        indices = np.arange(n)
        np.random.shuffle(indices)
        val_size = int(self.config.val_ratio * n)
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

        train_ds = Subset(dataset, train_idx.tolist())
        val_ds = Subset(dataset, val_idx.tolist())

        return self._train_loop(train_ds, val_ds, fold_id=None)

    def _fit_kfold(self, dataset: Dataset) -> dict[str, Any]:
        # Extract labels to stratify
        targets = np.array([dataset[i][1] for i in range(len(dataset))])
        skf = StratifiedKFold(n_splits=self.config.k_folds, shuffle=True, random_state=self.config.seed)

        fold_results = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets), start=1):
            self._reset_model()
            train_ds = Subset(dataset, train_idx.tolist())
            val_ds = Subset(dataset, val_idx.tolist())
            res = self._train_loop(train_ds, val_ds, fold_id=fold)
            fold_results.append(res)

        return {"folds": fold_results}

    def _train_loop(self, train_ds: Dataset, val_ds: Dataset, fold_id: int | None) -> dict[str, Any]:
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

        # if self._config.class_weighted_loss:
        #     class_counts = np.bincount([train_ds[i][1] for i in range(len(train_ds))], minlength=self.num_classes)
        #     weights = 1.0 / np.maximum(class_counts, 1)
        #     weights = weights / weights.sum() * self.num_classes  # normalize
        #     weight_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
        #     criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        # else:
        criterion = nn.CrossEntropyLoss()

        # Optimizer
        if self.config.optimizer.lower() == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        elif self.config.optimizer.lower() == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.lr,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError("Unsupported optimizer")

        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

        self.model.to(self.device)

        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_epoch = -1
        ckpt_path = self.config.ckpt_dir / (f"best_model_fold{fold_id}.pth" if fold_id else "best_model.pth")

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self._run_epoch(train_loader, optimizer, criterion, train=True, epoch=epoch)
            val_loss, val_acc = self._run_epoch(val_loader, optimizer, criterion, train=False, epoch=epoch)

            scheduler.step(val_loss)

            # TensorBoard logging
            tag = f"fold_{fold_id}" if fold_id else "single"
            self.writer.add_scalar(f"{tag}/train_loss", train_loss, epoch)
            self.writer.add_scalar(f"{tag}/val_loss", val_loss, epoch)
            self.writer.add_scalar(f"{tag}/train_acc", train_acc, epoch)
            self.writer.add_scalar(f"{tag}/val_acc", val_acc, epoch)

            # Early stopping check
            improved = val_loss < best_val_loss - 1e-6
            if improved:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch
                if self.config.save_best_only:
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "settings": self.config.model_dump(),
                            "class_names": self.class_names,
                        },
                        ckpt_path,
                    )
            elif (epoch - best_epoch) >= self.config.early_stop_patience:
                print(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch} (val_loss={best_val_loss:.4f}).")
                break

            dt = time.time() - t0
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} acc={train_acc:.3f} | val_loss={val_loss:.4f} acc={val_acc:.3f} | {dt:.1f}s"
            )

        if not self.config.save_best_only:
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "settings": self.config.model_dump(),
                    "class_names": self.class_names,
                },
                ckpt_path,
            )

        return {
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "ckpt_path": ckpt_path,
        }

    def _run_epoch(self, loader: DataLoader, optimizer, criterion, train: bool, epoch: int) -> tuple[float, float]:
        mode = "train" if train else "eval"
        self.model.train(mode == "train")

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(loader, desc=f"{mode} epoch {epoch}", leave=False)

        for batch in pbar:
            inputs, targets = batch
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            if train:
                optimizer.zero_grad()
                device = "cuda" if torch.cuda.is_available() else "cpu"
                with torch.amp.autocast(device_type=device, enabled=self.config.amp):
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                with torch.no_grad():
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)

            preds = outputs.argmax(dim=1)
            correct = (preds == targets).sum().item()
            batch_size = targets.size(0)

            total_loss += loss.item() * batch_size
            total_correct += correct
            total_samples += batch_size

            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return avg_loss, avg_acc

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _reset_model(self) -> None:
        """Restore the model weights to their initial state."""
        self.model.load_state_dict(self._initial_state)

    def fit(self, dataset: Dataset) -> dict[str, Any]:
        if self.config.k_folds == 1:
            return self._fit_single_split(dataset)
        else:
            return self._fit_kfold(dataset)

    def predict(self, tensor: torch.Tensor) -> tuple[int, torch.Tensor]:
        self.model.eval()
        with torch.inference_mode():
            x = tensor.unsqueeze(0).to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu()
            pred = int(probs.argmax().item())
        return pred, probs
