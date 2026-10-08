import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "_download"
KAGGLE_SLUG = "ravirajsinh45/real-life-industrial-dataset-of-casting-product"
SOURCE_FILE = DATA_DIR / "SOURCE.txt"

KAGGLE_SETUP_STEPS = """
Kaggle credentials not found or not working.

To download the real dataset:
  1. Make a free account at https://www.kaggle.com
  2. Settings -> API -> Create New Token (this downloads kaggle.json)
  3. Put kaggle.json in ~/.kaggle/ (Windows: C:\\Users\\<you>\\.kaggle\\)
     or set the KAGGLE_USERNAME and KAGGLE_KEY environment variables
  4. Run again: python download_data.py --force

No API? Download the zip manually from the dataset page, unzip it, then run:
  python download_data.py --from-folder "C:\\path\\to\\unzipped\\folder"

Stopping here: a model trained on synthetic data will NOT work on real products.
(If you only want to test that the pipeline runs, add --allow-synthetic.)
"""


def has_kaggle_credentials() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def organize_casting(download_dir: Path, out_dir: Path) -> int:
    mapping = {"def_front": "defective", "ok_front": "perfect"}
    copied = 0
    for folder in download_dir.rglob("*"):
        if not (folder.is_dir() and folder.name in mapping and folder.parent.name in ("train", "test")):
            continue
        dst = out_dir / folder.parent.name / mapping[folder.name]
        dst.mkdir(parents=True, exist_ok=True)
        for img in folder.iterdir():
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"} and not (dst / img.name).exists():
                shutil.copy2(img, dst / img.name)
                copied += 1
    return copied


def download_from_kaggle(replace: bool = False) -> bool:
    if not has_kaggle_credentials():
        print("No Kaggle credentials found.")
        return False
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {KAGGLE_SLUG} (about 100 MB, one-off) ...")
        api.dataset_download_files(KAGGLE_SLUG, path=str(DOWNLOAD_DIR), unzip=True, quiet=False)
        if replace and DATA_DIR.exists():
            shutil.rmtree(DATA_DIR)  # only now that the download succeeded
        n = organize_casting(DOWNLOAD_DIR, DATA_DIR)
        if n == 0:
            print("Downloaded, but could not find the expected def_front/ok_front folders.")
            return False
        shutil.rmtree(DOWNLOAD_DIR, ignore_errors=True)
        SOURCE_FILE.write_text("kaggle-casting\n")
        print(f"Done: {n} images organized under {DATA_DIR}")
        return True
    except Exception as exc:
        print(f"Kaggle download failed: {type(exc).__name__}: {exc}")
        return False


def make_clean_part(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = size / 2 + rng.uniform(-8, 8)
    cy = size / 2 + rng.uniform(-8, 8)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    disc = np.clip(1.0 - (r / (size * 0.46)) ** 4, 0.0, 1.0)

    texture = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), 2.0) * 14
    grain = rng.normal(0, 4, (size, size)).astype(np.float32)
    brightness = rng.uniform(-15, 15)

    gray = 45 + 95 * disc + texture * disc + grain + brightness * disc
    tint = rng.uniform(0.95, 1.05, 3)
    img = np.stack([gray * tint[0], gray * tint[1], gray * tint[2]], axis=-1)
    return np.clip(img, 0, 255).astype(np.float32)


def add_scratch(img: np.ndarray, rng: np.random.Generator) -> None:
    size = img.shape[0]
    layer = np.zeros(img.shape[:2], np.float32)
    x, y = rng.uniform(size * 0.25, size * 0.75, 2)
    angle = rng.uniform(0, 2 * np.pi)
    points = [(x, y)]
    for _ in range(int(rng.integers(3, 6))):
        angle += rng.uniform(-0.35, 0.35)
        step = rng.uniform(12, 30)
        x, y = x + step * np.cos(angle), y + step * np.sin(angle)
        points.append((x, y))
    pts = np.array(points, np.int32).reshape(-1, 1, 2)
    cv2.polylines(layer, [pts], False, 1.0, thickness=int(rng.integers(1, 3)), lineType=cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (0, 0), 0.8)
    delta = rng.choice([-1, 1]) * rng.uniform(45, 85)
    img += layer[..., None] * delta


def add_dent(img: np.ndarray, rng: np.random.Generator) -> None:
    size = img.shape[0]
    mask = np.zeros(img.shape[:2], np.float32)
    center = (int(rng.uniform(size * 0.3, size * 0.7)), int(rng.uniform(size * 0.3, size * 0.7)))
    axes = (int(rng.integers(7, 18)), int(rng.integers(5, 14)))
    cv2.ellipse(mask, center, axes, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 3.0)
    shift = int(rng.integers(3, 6))
    img += (-45 * mask + 30 * np.roll(mask, shift, axis=1))[..., None]


def make_defective_part(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    img = make_clean_part(rng, size)
    n_scratches = int(rng.integers(0, 3))
    n_dents = int(rng.integers(0, 2))
    if n_scratches + n_dents == 0:
        n_scratches = 1
    for _ in range(n_scratches):
        add_scratch(img, rng)
    for _ in range(n_dents):
        add_dent(img, rng)
    return np.clip(img, 0, 255).astype(np.float32)


def generate_synthetic(per_class_train: int, per_class_test: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    print(f"Generating synthetic data: {per_class_train}/class train, {per_class_test}/class test ...")
    for split, n in (("train", per_class_train), ("test", per_class_test)):
        for label, maker in (("perfect", make_clean_part), ("defective", make_defective_part)):
            out = DATA_DIR / split / label
            out.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                img = maker(rng)
                rgb_u8 = np.clip(img, 0, 255).astype(np.uint8)
                cv2.imwrite(str(out / f"{label}_{i:04d}.jpg"), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
    SOURCE_FILE.write_text("synthetic\n")
    print(f"Done. Synthetic dataset written to {DATA_DIR}")


def count_images() -> int:
    return sum(1 for p in DATA_DIR.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) if DATA_DIR.exists() else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the casting defect dataset or generate a synthetic one")
    parser.add_argument("--synthetic", action="store_true", help="skip Kaggle, generate synthetic data")
    parser.add_argument("--force", action="store_true", help="replace any existing data")
    parser.add_argument("--from-folder", type=str, default=None,
                        help="use a manually downloaded + unzipped Kaggle casting dataset in this folder")
    parser.add_argument("--allow-synthetic", action="store_true",
                        help="if Kaggle fails, fall back to synthetic data instead of stopping (pipeline test only)")
    parser.add_argument("--per-class-train", type=int, default=300, help="synthetic: train images per class")
    parser.add_argument("--per-class-test", type=int, default=80, help="synthetic: test images per class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    existing = count_images()
    if existing and not args.force:
        source = SOURCE_FILE.read_text().strip() if SOURCE_FILE.exists() else "unknown"
        print(f"Found {existing} images in {DATA_DIR} (source: {source}). Use --force to replace them.")
        if source == "synthetic":
            print("WARNING: this is SYNTHETIC data. A model trained on it will not work on real products.\n"
                  "         Set up Kaggle credentials and run: python download_data.py --force")
        return

    if args.from_folder:
        src = Path(args.from_folder)
        if not src.exists():
            raise SystemExit(f"Folder not found: {src}")
        n = organize_casting(src, DATA_DIR)
        if n == 0:
            raise SystemExit("No def_front / ok_front folders found under that path. "
                             "Point --from-folder at the unzipped dataset (the folder that contains train and test).")
        SOURCE_FILE.write_text("kaggle-casting\n")
        print(f"Done: {n} images organized under {DATA_DIR}")
        return

    if args.synthetic:
        if DATA_DIR.exists():
            shutil.rmtree(DATA_DIR)
        generate_synthetic(args.per_class_train, args.per_class_test, args.seed)
        return

    if download_from_kaggle(replace=args.force):
        return

    print(KAGGLE_SETUP_STEPS)
    if not args.allow_synthetic:
        raise SystemExit(1)
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    generate_synthetic(args.per_class_train, args.per_class_test, args.seed)


if __name__ == "__main__":
    main()
