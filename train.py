import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.config import (ASSETS_DIR, CHECKPOINT_PATH, CLASS_NAMES, DATA_DIR, DEFECTIVE_IDX, MODELS_DIR,
                        SEED, read_data_source, seed_everything)
from src.dataset import build_dataloaders
from src.gradcam import pick_examples, save_gradcam_grid
from src.model import build_model, count_parameters, set_train_mode, unfreeze_last_block


def run_epoch(model, loader, criterion, device, optimizer=None) -> Tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    if training:
        set_train_mode(model)
    else:
        model.eval()
    total_loss, n = 0.0, 0
    all_labels, all_probs = [], []
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            n += len(labels)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(torch.softmax(logits.detach(), dim=1)[:, DEFECTIVE_IDX].cpu().numpy())
    return total_loss / n, np.concatenate(all_labels), np.concatenate(all_probs)


def compute_metrics(y_true: np.ndarray, p_defect: np.ndarray, threshold: float = 0.5) -> Dict:
    y_pred = (p_defect >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def pick_threshold(y_true: np.ndarray, p_defect: np.ndarray, fn_cost: float, fp_cost: float) -> float:
    best_t, best_key = 0.5, None
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        m = compute_metrics(y_true, p_defect, t)
        cost = fn_cost * m["fn"] + fp_cost * m["fp"]
        key = (cost, abs(t - 0.5))
        if best_key is None or key < best_key:
            best_t, best_key = float(t), key
    return best_t


def print_metrics(title: str, m: Dict, fn_cost: float, fp_cost: float) -> None:
    cost = fn_cost * m["fn"] + fp_cost * m["fp"]
    print(f"\n--- {title} (threshold={m['threshold']:.2f}) ---")
    print(f"Accuracy {m['accuracy']:.3f} | Precision {m['precision']:.3f} | Recall {m['recall']:.3f} | F1 {m['f1']:.3f}")
    print("Confusion matrix (rows = truth, columns = prediction):")
    print(f"              pred Perfect  pred Defective")
    print(f"true Perfect  {m['tn']:>12}  {m['fp']:>14}   <- false alarms (FP)")
    print(f"true Defective{m['fn']:>12}  {m['tp']:>14}   <- missed defects (FN)")
    print(f"Cost-weighted total = {fn_cost:g} x FN + {fp_cost:g} x FP = {cost:g}")


def plot_confusion_matrices(metrics_list, titles, out_path) -> None:
    fig, axes = plt.subplots(1, len(metrics_list), figsize=(5 * len(metrics_list), 4.4), squeeze=False)
    for ax, m, title in zip(axes[0], metrics_list, titles):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], CLASS_NAMES)
        ax.set_yticks([0, 1], CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{title}\nrecall={m['recall']:.2f} precision={m['precision']:.2f}", fontsize=10)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-phase transfer learning for defect detection")
    parser.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--head-epochs", type=int, default=3, help="phase 1 epochs (head only)")
    parser.add_argument("--finetune-epochs", type=int, default=4, help="phase 2 epochs (head + layer4)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-lr", type=float, default=1e-3, help="phase 1: new layer can learn fast")
    parser.add_argument("--finetune-lr", type=float, default=1e-4, help="phase 2: small, to avoid forgetting")
    parser.add_argument("--max-train-per-class", type=int, default=400, help="cap for CPU speed; 0 = use all")
    parser.add_argument("--max-eval-per-class", type=int, default=200, help="cap for val set; 0 = use all")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fn-cost", type=float, default=20.0, help="relative cost of a MISSED defect")
    parser.add_argument("--fp-cost", type=float, default=1.0, help="relative cost of a FALSE ALARM")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-pretrained", action="store_true", help="debug only: random weights")
    args = parser.parse_args()

    seed_everything(args.seed)
    data_source = read_data_source(Path(args.data_dir))
    if data_source == "synthetic":
        print("\n" + "!" * 70 +
              "\nWARNING: training on SYNTHETIC data. The model will not work on real products."
              "\n         Run: python download_data.py --force   (needs Kaggle credentials)"
              "\n" + "!" * 70 + "\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = time.time()
    print(f"Device: {device} | arch: {args.arch} | seed: {args.seed}")

    data = build_dataloaders(Path(args.data_dir), batch_size=args.batch_size, max_train_per_class=args.max_train_per_class,
                             max_eval_per_class=args.max_eval_per_class, num_workers=args.num_workers,
                             seed=args.seed)
    loaders = data["loaders"]
    print("Images per split:", json.dumps(data["counts"]))

    model = build_model(args.arch, pretrained=not args.no_pretrained).to(device)

    c = data["counts"]["train"]
    totals = np.array([c["perfect"], c["defective"]], dtype=np.float32)
    weights = torch.tensor(totals.sum() / (2 * totals), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best = {"key": (-1.0, 0.0)}

    def run_phase(name: str, epochs: int, optimizer, scheduler) -> None:
        trainable, total = count_parameters(model)
        print(f"\n===== {name}: {epochs} epochs | trainable params {trainable:,} of {total:,} =====")
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, y_tr, p_tr = run_epoch(model, loaders["train"], criterion, device, optimizer)
            val_loss, y_va, p_va = run_epoch(model, loaders["val"], criterion, device)
            scheduler.step()
            val_m = compute_metrics(y_va, p_va)
            train_acc = float(((p_tr >= 0.5).astype(int) == y_tr).mean())
            print(f"[{name}] epoch {epoch}/{epochs} | train loss {train_loss:.4f} acc {train_acc:.3f} | "
                  f"val loss {val_loss:.4f} acc {val_m['accuracy']:.3f} F1 {val_m['f1']:.3f} "
                  f"recall {val_m['recall']:.3f} | lr {optimizer.param_groups[0]['lr']:.2e} | {time.time() - t0:.0f}s")
            key = (val_m["f1"], -val_loss)
            if key > best["key"]:
                best["key"] = key
                torch.save({"model_state": model.state_dict(), "arch": args.arch, "class_names": CLASS_NAMES,
                            "val_f1": float(val_m["f1"]), "phase": name, "epoch": epoch,
                            "decision_threshold": 0.5, "data_source": data_source}, CHECKPOINT_PATH)
                print(f"    -> new best model saved (val F1 {val_m['f1']:.3f})")

    head_params = [p for p in model.parameters() if p.requires_grad]
    opt1 = torch.optim.Adam(head_params, lr=args.head_lr)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=max(args.head_epochs, 1))
    run_phase("phase1-head", args.head_epochs, opt1, sched1)

    unfreeze_last_block(model)
    params2 = [p for p in model.parameters() if p.requires_grad]
    opt2 = torch.optim.Adam(params2, lr=args.finetune_lr)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=max(args.finetune_epochs, 1))
    run_phase("phase2-finetune", args.finetune_epochs, opt2, sched2)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    print(f"\nLoaded best model: {ckpt['phase']} epoch {ckpt['epoch']} (val F1 {ckpt['val_f1']:.3f})")

    _, y_val, p_val = run_epoch(model, loaders["val"], criterion, device)
    threshold = pick_threshold(y_val, p_val, args.fn_cost, args.fp_cost)
    _, y_test, p_test = run_epoch(model, loaders["test"], criterion, device)

    default_m = compute_metrics(y_test, p_test, 0.5)
    tuned_m = compute_metrics(y_test, p_test, threshold)
    print_metrics("TEST, standard threshold", default_m, args.fn_cost, args.fp_cost)
    print_metrics(f"TEST, cost-aware threshold (FN costs {args.fn_cost:g}x a false alarm)", tuned_m,
                  args.fn_cost, args.fp_cost)

    ckpt["decision_threshold"] = float(threshold)
    torch.save(ckpt, CHECKPOINT_PATH)
    (MODELS_DIR / "metrics.json").write_text(json.dumps(
        {"arch": args.arch, "fn_cost": args.fn_cost, "fp_cost": args.fp_cost,
         "test_standard": default_m, "test_cost_aware": tuned_m}, indent=2))
    plot_confusion_matrices([default_m, tuned_m], ["threshold 0.50", f"cost-aware threshold {threshold:.2f}"],
                            ASSETS_DIR / "confusion_matrix.png")

    test_samples = data["test_samples"]
    examples = pick_examples(test_samples, 1, 3) + pick_examples(test_samples, 0, 3)
    save_gradcam_grid(model, examples, ASSETS_DIR / "gradcam_examples.png", device, threshold,
                      title="Grad-CAM: where the model looks for defects (red = strongest evidence)")
    y_pred = (p_test >= threshold).astype(int)
    missed = [test_samples[i] for i in np.where((y_test == 1) & (y_pred == 0))[0][:4]]
    if missed:
        save_gradcam_grid(model, missed, ASSETS_DIR / "gradcam_missed_defects.png", device, threshold,
                          title="Missed defects (false negatives): what did the model look at?")

    print(f"\nTotal time: {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
