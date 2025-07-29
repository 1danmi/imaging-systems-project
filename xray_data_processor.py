import json
import hashlib

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Literal, Callable, Sequence

import cv2
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import TensorDataset

from config.processor_settings import ProcessorSettings

IMG = Image.Image


AugName = Literal["none", "hflip", "rotate", "translate", "brightness_contrast", "noise", "clahe"]


class XRayDataProcessor:
    def __init__(self, config: ProcessorSettings):
        self._config = config
        if self._config.cache_dir is None:
            self._config.cache_dir = self._config.root_dir.parent / f"{self._config.root_dir.name}_cache"
        self._config.cache_dir.mkdir(parents=True, exist_ok=True)

        self._records: list[tuple[Path, int]] = []
        self._manifest_path = self._config.cache_dir / "cache_manifest.json"
        self._manifest: dict[str, dict] = self._load_manifest()

        self._aug_fns: dict[AugName, Callable[[IMG], IMG]] = {
            "none": lambda im: im,
            "hflip": self._hflip,
            "rotate": self._rotate,
            "translate": self._translate,
            "brightness_contrast": self._aug_brightness_contrast,
            "noise": self._add_noise,
            "clahe": self._clahe,
        }

    def _load_manifest(self) -> dict[str, dict]:
        if self._manifest_path.exists():
            try:
                return json.loads(self._manifest_path.read_text())
            except Exception as e:
                return {}
        return {}

    def _save_manifest(self) -> None:
        self._manifest_path.write_text(json.dumps(self._manifest, indent=2))

    @property
    def records(self) -> list[tuple[Path, int]]:
        """Return a copy of the internal list of image paths and labels."""
        return list(self._records)

    def open_and_resize(self, path: Path) -> IMG:
        img = Image.open(path).convert("L")  # 'L' => 8-bit grayscale [0,255]
        height, width = self._config.output_size
        return img.resize((width, height), Image.BILINEAR)

    def to_tensor_and_normalize(self, img: IMG) -> Tensor:
        arr = np.asarray(img, dtype=np.float32) / 255  # (H, W)
        if self._config.to_rgb:
            arr = np.stack([arr, arr, arr], axis=0)  # (3, H, W)
        else:
            arr = arr[None, ...]  # (1, H, W)

        tensor = torch.from_numpy(arr)
        mean = torch.tensor(self._config.normalize_mean[: tensor.shape[0]]).view(-1, 1, 1)
        std = torch.tensor(self._config.normalize_std[: tensor.shape[0]]).view(-1, 1, 1)
        return (tensor - mean) / std

    def _hflip(self, img: IMG) -> IMG:
        """Horizontal flip"""
        return img.transpose(Image.FLIP_LEFT_RIGHT)

    def _rotate(self, img: IMG) -> Image:
        """Random small rotation around the center"""
        deg = float(np.random.uniform(-self._config.max_rotation, self._config.max_rotation))
        return img.rotate(deg, resample=Image.BILINEAR, fillcolor=0)

    def _translate(self, img: IMG) -> IMG:
        """Shift the image to the sides a little bit"""
        w, h = img.size
        max_dx = int(self._config.max_translate * w)
        max_dy = int(self._config.max_translate * h)
        dx = int(np.random.randint(-max_dx, max_dx + 1))
        dy = int(np.random.randint(-max_dy, max_dy + 1))
        return img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), resample=Image.BILINEAR, fillcolor=0)

    def _aug_brightness_contrast(self, img: IMG) -> IMG:
        """Brightness & contrast jitter"""
        arr = np.asarray(img, dtype=np.float32)
        b = 1.0 + float(np.random.uniform(-self._config.brightness_factor, self._config.brightness_factor))
        c = 1.0 + float(np.random.uniform(-self._config.contrast_factor, self._config.contrast_factor))

        arr_b = arr * b
        mu = arr_b.mean()
        arr_bc = (arr_b - mu) * c + mu
        arr_bc = np.clip(arr_bc, 0, 255).astype(np.uint8)
        return Image.fromarray(arr_bc, mode="L")

    def _add_noise(self, img: IMG) -> IMG:
        """Add some random noise"""
        arr = np.asarray(img, dtype=np.float32) / 255.0
        noise = np.random.normal(0.0, self._config.noise_sigma, size=arr.shape).astype(np.float32)
        out = np.clip(arr + noise, 0.0, 1.0)
        out = (out * 255).astype(np.uint8)
        return Image.fromarray(out, mode="L")

    def _clahe(self, img: IMG) -> IMG:
        """Contrast Limited Adaptive Histogram Equalization. Weird augmentation method I found online.

        Meant to boost local contrast while limiting noise amplification.
        """
        arr = np.asarray(img, dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=self._config.clahe_clip, tileGridSize=self._config.clahe_tile_grid)
        out = clahe.apply(arr)
        return Image.fromarray(out, mode="L")

    def _hash_for_aug(self, orig_path: Path, aug_name: str) -> str:
        """Hash for caching"""
        h = hashlib.sha1()
        h.update(str(orig_path).encode())
        h.update(aug_name.encode())
        relevant = {
            "max_rotation": self._config.max_rotation,
            "max_translate": self._config.max_translate,
            "brightness_factor": self._config.brightness_factor,
            "contrast_factor": self._config.contrast_factor,
            "noise_sigma": self._config.noise_sigma,
            "clahe_clip": self._config.clahe_clip,
            "clahe_tile_grid": self._config.clahe_tile_grid,
            "output_size": self._config.output_size,
        }
        h.update(json.dumps(relevant, sort_keys=True).encode())
        return h.hexdigest()[:12]

    def _augmented_path(self, orig_path: Path, aug_name: str, aug_hash: str) -> Path:
        rel = orig_path.relative_to(self._config.root_dir)
        stem = orig_path.stem
        ext = ".png" if self._config.persist_png else ".jpg"
        return self._config.cache_dir / rel.parent / f"{stem}__{aug_name}__{aug_hash}{ext}"

    def scan(self) -> None:
        """Scan `root_dir` for images and build the internal list of (path, label)."""
        self._records.clear()
        class_dirs = sorted([p for p in self._config.root_dir.iterdir() if p.is_dir()])
        if not class_dirs:
            raise FileNotFoundError(f"No class folders under {self._config.root_dir}")

        for label, cdir in enumerate(class_dirs):
            for pattern in ("*.jpg", "*.jpeg", "*.png"):
                for f in cdir.glob(pattern):
                    if f.suffix.lower() in self._config.denylist_ext:
                        continue
                    self._records.append((f, label))
        if not self._records:
            raise FileNotFoundError("No images found – check extensions/denylist.")

    def make_dataset(
        self,
        augmentations: list[AugName],
        records: Sequence[tuple[Path, int]] | None = None,
        persist: bool = True,
    ) -> TensorDataset:
        """Create a ``TensorDataset`` from the given records applying the
        specified augmentations."""
        if records is None:
            records = self._records
        if not records:
            raise RuntimeError("No image records provided")
        print(f"Augmenting dataset with: {augmentations}")

        images: list[Tensor] = []
        labels: list[Tensor] = []

        print("Loading original images...")
        for path, label in records:
            img = self.open_and_resize(path)
            tensor = self.to_tensor_and_normalize(img)
            images.append(tensor)
            labels.append(torch.tensor(label, dtype=torch.long))

        for aug_name in augmentations:
            if aug_name not in self._aug_fns:
                raise ValueError(f"Unknown augmentation '{aug_name}'")
            print(f"Augmenting with {aug_name}...")
            aug_fn = self._aug_fns[aug_name]

            for path, label in records:
                aug_hash = self._hash_for_aug(path, aug_name)
                out_path = self._augmented_path(path, aug_name, aug_hash)

                if persist and out_path.exists():
                    aug_img = Image.open(out_path).convert("L")
                else:
                    base = self.open_and_resize(path)  # no normalization yet
                    aug_img = aug_fn(base)
                    if persist:
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        save_path = out_path.with_suffix(".png" if self._config.persist_png else ".jpg")
                        aug_img.save(save_path)
                        self._manifest[str(save_path)] = {
                            "origin": str(path),
                            "aug": aug_name,
                            "hash": aug_hash,
                        }
                tensor = self.to_tensor_and_normalize(aug_img)
                images.append(tensor)
                labels.append(torch.tensor(label, dtype=torch.long))

        # Persist manifest
        self._save_manifest()

        X = torch.stack(images)  # (N,C,H,W)
        y = torch.stack(labels)  # (N,)
        return TensorDataset(X, y)

    # Backwards compatible alias
    def augment_dataset(
        self,
        augmentations: list[AugName],
        persist: bool = True,
    ) -> TensorDataset:
        return self.make_dataset(augmentations, records=None, persist=persist)


if __name__ == "__main__":
    root = Path("data/train")
    settings = ProcessorSettings(root_dir=root)
    proc = XRayDataProcessor(settings)
    proc.scan()
    ds = proc.make_dataset(["none", "hflip", "rotate", "translate", "b&c", "noise", "clahe"], persist=True)
    print("Dataset size:", len(ds))
    print("Tensor shapes:", ds.tensors[0].shape, ds.tensors[1].shape)
