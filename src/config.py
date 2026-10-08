import os
import random
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
CHECKPOINT_PATH = MODELS_DIR / "best_model.pt"

SEED = 42

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLASS_NAMES = ["Perfect", "Defective"]
PERFECT_IDX, DEFECTIVE_IDX = 0, 1
FOLDER_TO_LABEL = {"perfect": PERFECT_IDX, "defective": DEFECTIVE_IDX}


def seed_everything(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_data_source(data_dir: Path = DATA_DIR) -> str:
    """Returns 'kaggle-casting', 'synthetic' or 'unknown' (from data/raw/SOURCE.txt)."""
    f = Path(data_dir) / "SOURCE.txt"
    return f.read_text().strip() if f.exists() else "unknown"
