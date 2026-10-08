from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.config import CLASS_NAMES, DEFECTIVE_IDX
from src.dataset import load_image_rgb, preprocess_for_model


class GradCAM:

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handle = target_layer.register_forward_hook(self._on_forward)

    def _on_forward(self, module, inputs, output):
        self.activations = output.detach()
        output.register_hook(lambda grad: setattr(self, "gradients", grad.detach()))

    def remove(self) -> None:
        self._handle.remove()

    def __call__(self, x: torch.Tensor, target_class: int) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        x = x.clone().requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(x)
            self.model.zero_grad(set_to_none=True)
            logits[0, target_class].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1))[0]
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]), interpolation=cv2.INTER_LINEAR)
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 1e-8 else np.zeros_like(cam)
        probs = torch.softmax(logits.detach(), dim=1)[0].cpu().numpy()
        return cam, probs


def overlay_heatmap(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat_bgr = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 1 - alpha, heat_rgb, alpha, 0)


def explain_image(model: nn.Module, rgb: np.ndarray, device: str = "cpu",
                  target_class: int = DEFECTIVE_IDX) -> Dict:
    x, resized = preprocess_for_model(rgb)
    explainer = GradCAM(model, model.layer4[-1])
    try:
        cam, probs = explainer(x.to(device), target_class)
    finally:
        explainer.remove()
    return {"probs": probs, "cam": cam, "resized": resized, "overlay": overlay_heatmap(resized, cam)}


def save_gradcam_grid(model: nn.Module, samples: Sequence[Tuple[Path, int]], out_path: Path,
                      device: str = "cpu", threshold: float = 0.5, title: str = "") -> None:
    n = len(samples)
    if n == 0:
        return
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.4), squeeze=False)
    for col, (path, true_label) in enumerate(samples):
        result = explain_image(model, load_image_rgb(path), device)
        p_def = float(result["probs"][DEFECTIVE_IDX])
        pred = DEFECTIVE_IDX if p_def >= threshold else 1 - DEFECTIVE_IDX
        ok = "OK" if pred == true_label else "WRONG"
        axes[0, col].imshow(result["resized"])
        axes[0, col].set_title(f"true: {CLASS_NAMES[true_label]}", fontsize=10)
        axes[1, col].imshow(result["overlay"])
        axes[1, col].set_title(f"pred: {CLASS_NAMES[pred]} ({p_def:.0%} defect) {ok}", fontsize=9)
        for row in (0, 1):
            axes[row, col].axis("off")
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"Saved {out_path}")


def pick_examples(samples: List[Tuple[Path, int]], label: int, n: int, seed: int = 0) -> List[Tuple[Path, int]]:
    pool = [s for s in samples if s[1] == label]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[i] for i in sorted(idx)]
