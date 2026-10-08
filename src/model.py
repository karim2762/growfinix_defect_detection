from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models

from src.config import CLASS_NAMES

SUPPORTED = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
}


def build_model(arch: str = "resnet18", pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    if arch not in SUPPORTED:
        raise ValueError(f"arch must be one of {list(SUPPORTED)}")
    constructor, weights = SUPPORTED[arch]

    model = None
    if pretrained:
        try:
            model = constructor(weights=weights)
        except Exception as exc:
            print(f"WARNING: could not load pretrained weights ({type(exc).__name__}: {exc}).\n"
                  "         Falling back to RANDOM weights; accuracy will be much worse.\n"
                  "         Connect to the internet once so torchvision can cache the weights.")
    if model is None:
        model = constructor(weights=None)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, len(CLASS_NAMES)),
    )
    return model


def unfreeze_last_block(model: nn.Module) -> None:
    for param in model.layer4.parameters():
        param.requires_grad = True


def set_train_mode(model: nn.Module) -> None:
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) and not any(p.requires_grad for p in module.parameters()):
            module.eval()


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def load_checkpoint(path: Path, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ckpt["arch"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt
