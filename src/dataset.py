from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import (FOLDER_TO_LABEL, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, SEED)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
Sample = Tuple[Path, int]


def load_image_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_rgb(rgb: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    shrinking = rgb.shape[0] > size or rgb.shape[1] > size
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(rgb, (size, size), interpolation=interp)


_normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

eval_transform = transforms.Compose([transforms.ToTensor(), _normalize])

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    _normalize,
])


def preprocess_for_model(rgb: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
    resized = resize_rgb(rgb)
    return eval_transform(resized).unsqueeze(0), resized


class DefectDataset(Dataset):
    def __init__(self, samples: List[Sample], train: bool = False):
        self.samples = samples
        self.transform = train_transform if train else eval_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        rgb = resize_rgb(load_image_rgb(path))
        return self.transform(rgb), label


def list_samples(split_dir: Path) -> List[Sample]:
    samples: List[Sample] = []
    for folder_name, label in FOLDER_TO_LABEL.items():
        folder = split_dir / folder_name
        if folder.exists():
            samples += [(p, label) for p in sorted(folder.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]
    return samples


def subsample_per_class(samples: List[Sample], max_per_class: int, seed: int) -> List[Sample]:
    if max_per_class <= 0:
        return samples
    rng = np.random.default_rng(seed)
    kept: List[Sample] = []
    for label in sorted({l for _, l in samples}):
        group = [s for s in samples if s[1] == label]
        if len(group) > max_per_class:
            idx = rng.choice(len(group), size=max_per_class, replace=False)
            group = [group[i] for i in sorted(idx)]
        kept += group
    return kept


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def build_dataloaders(
    data_dir: Path,
    batch_size: int = 32,
    max_train_per_class: int = 400,
    max_eval_per_class: int = 200,
    val_fraction: float = 0.2,
    num_workers: int = 0,
    seed: int = SEED,
) -> Dict:
    train_all = list_samples(data_dir / "train")
    test_samples = list_samples(data_dir / "test")
    if not train_all or not test_samples:
        raise FileNotFoundError(
            f"No images found under {data_dir}/train and /test. Run:  python download_data.py"
        )

    labels = [l for _, l in train_all]
    train_s, val_s = train_test_split(train_all, test_size=val_fraction, stratify=labels, random_state=seed)
    train_s = subsample_per_class(train_s, max_train_per_class, seed)
    val_s = subsample_per_class(val_s, max_eval_per_class, seed)

    gen = torch.Generator().manual_seed(seed)
    common = dict(batch_size=batch_size, num_workers=num_workers, worker_init_fn=_seed_worker)
    loaders = {
        "train": DataLoader(DefectDataset(train_s, train=True), shuffle=True, generator=gen, **common),
        "val": DataLoader(DefectDataset(val_s), shuffle=False, **common),
        "test": DataLoader(DefectDataset(test_samples), shuffle=False, **common),
    }
    counts = {name: {"perfect": sum(l == 0 for _, l in s), "defective": sum(l == 1 for _, l in s)}
              for name, s in (("train", train_s), ("val", val_s), ("test", test_samples))}
    return {"loaders": loaders, "counts": counts, "test_samples": test_samples, "train_samples": train_s}
