# Product Defect Detection

Classifies product images as Defective or Perfect using a pretrained ResNet and shows where the model looked with Grad-CAM. Runs fine on a laptop CPU (about 5-10 minutes of training with the defaults).

Stack: Python, OpenCV, PyTorch/torchvision, scikit-learn, Streamlit.

## Setup

Run everything from inside the project folder. Python 3.10-3.12.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip

# Linux / Windows (CPU build of PyTorch)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
# macOS
pip install -r requirements.txt
```

## Usage

```bash
python download_data.py     # Kaggle data, or synthetic data if no credentials
python train.py             # trains, prints metrics, saves model and plots
streamlit run app.py        # web demo
```

## Kaggle data

Dataset: [Real-life industrial dataset of casting product](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)

1. Create a Kaggle API token (Settings -> API -> Create New Token).
2. Put `kaggle.json` in `~/.kaggle/` or set `KAGGLE_USERNAME` and `KAGGLE_KEY`.
3. Run `python download_data.py --force`.

Without credentials the script now stops with instructions. To generate a synthetic dataset anyway (pipeline test only, the model will not work on real products), run `python download_data.py --synthetic --force` or add `--allow-synthetic`. `train.py` and the app warn when the data is synthetic.

## Options

```bash
python train.py --arch resnet50
python train.py --max-train-per-class 1000
python train.py --max-train-per-class 0       # use all training images
python train.py --fn-cost 50                  # missed defects cost 50x a false alarm
python download_data.py --synthetic --force
```

## Layout

```
download_data.py     dataset download / synthetic generator
train.py             training, metrics, confusion matrix, Grad-CAM export
app.py               Streamlit app
src/config.py        paths, seed, image size, class names
src/dataset.py       preprocessing, augmentation, dataloaders
src/model.py         ResNet setup, freezing, checkpoint loading
src/gradcam.py       Grad-CAM and plotting helpers
notebooks/           data exploration notebook
```

## How it works

Training has two phases. First the ResNet backbone is frozen and only the new 2-class head is trained. Then the last block (`layer4`) is unfrozen and trained with a 10x smaller learning rate. The best epoch is picked by F1 on a validation split, and final numbers come from the separate test split.

Preprocessing: read with OpenCV, BGR to RGB, resize to 224x224, (training only) flips, rotation and brightness/contrast jitter, then ImageNet normalization.

Grad-CAM uses the gradient of the "Defective" score on the last conv block to build a heatmap. The map is coarse (7x7), so treat it as a rough indication.

## Metrics and false negatives

`train.py` reports accuracy, precision, recall and F1 on the test set, with Defective as the positive class. A missed defect usually costs much more than a false alarm, so `--fn-cost` and `--fp-cost` (default 20:1) are used to choose the decision threshold on the validation set. Test results are shown at both 0.5 and the tuned threshold, and the tuned one becomes the app default. The app also has a slider to change it.

Results are written to `models/metrics.json` and `assets/confusion_matrix.png`.

## Limitations

- Default training uses a CPU-friendly subset (400 images per class).
- The model only knows the defect types and lighting present in its training data.
- Grad-CAM is a debugging aid, not proof the model is right.
- Check any threshold against data from your own line.

## Troubleshooting

- "No images found": run `python download_data.py` first.
- Pretrained weights warning: connect to the internet once so torchvision can download them.
- Too slow: lower `--max-train-per-class` or `--finetune-epochs`.
- `ModuleNotFoundError: src`: run commands from the project folder.
- Streamlit says no model: run `python train.py` first.
