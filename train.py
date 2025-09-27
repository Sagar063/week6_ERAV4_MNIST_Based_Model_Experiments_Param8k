
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST multi-model trainer
- Loads MNIST
- Dataloaders + stats
- Visualizations
- Normalization
- Train & Eval each epoch
- Plots: accuracy, loss, confusion matrix
- Aggregates results into CSV summaries
"""
import argparse
import os
import random
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils as tv_utils

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# Import models
import model1 as M1
import model2 as M2
import model3 as M3
# Robust model summary import
try:
    from torchinfo import summary as _model_summary   # preferred
    _SUMMARY_LIB = "torchinfo"
except Exception:
    try:
        from torchsummary import summary as _model_summary  # fallback
        _SUMMARY_LIB = "torchsummary"
    except Exception:
        _model_summary = None
        _SUMMARY_LIB = None

from tqdm import tqdm 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_model_summary(model: nn.Module, exp_name: str):
    print("\n" + "="*80)
    print(f"Device: {DEVICE}")
    n_params = count_params(model)
    print(f"Experiment: {exp_name} | Trainable params: {n_params:,}")
    if _model_summary is None:
        print("(Install 'torchinfo' or 'torchsummary' to see a layer-by-layer summary)")
    else:
        try:
            if _SUMMARY_LIB == "torchinfo":
                # torchinfo expects NCHW with batch dim
                _model_summary(model, input_size=(1, 1, 28, 28), device=str(DEVICE))
            else:
                # torchsummary expects CHW (batch handled internally)
                _model_summary(model, input_size=(1, 28, 28), device=str(DEVICE))
        except Exception as e:
            print(f"(Model summary failed: {e})")
    print("="*80 + "\n")



def compute_dataset_stats(dataset) -> Dict[str, float]:
    # MNIST is single-channel; compute stats from the entire dataset
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    n_pixels = 0
    sum_ = 0.0
    sumsq_ = 0.0
    vmin = float("inf")
    vmax = float("-inf")
    for x, _ in loader:
        # x in [0,1], shape [B,1,28,28]
        vmin = min(vmin, x.min().item())
        vmax = max(vmax, x.max().item())
        b = x.shape[0]
        n_pixels += b * 28 * 28
        sum_ += x.sum().item()
        sumsq_ += (x ** 2).sum().item()
    mean = sum_ / n_pixels
    var = (sumsq_ / n_pixels) - (mean ** 2)
    std = math.sqrt(max(var, 1e-12))
    return {"min": vmin, "max": vmax, "mean": mean, "std": std, "var": var}

def show_samples(dataset, save_path: Path, n: int = 32, title: str = ""):
    loader = DataLoader(dataset, batch_size=n, shuffle=True, num_workers=0)
    x, y = next(iter(loader))
    grid = tv_utils.make_grid(x, nrow=8, padding=2)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(title)
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def accuracy_from_logits(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
        running_acc += (out.argmax(1) == yb).float().sum().item()
        n += xb.size(0)
    return running_loss / n, running_acc / n

@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    n = 0
    all_preds = []
    all_targets = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = criterion(out, yb)
        running_loss += loss.item() * xb.size(0)
        running_acc += (out.argmax(1) == yb).float().sum().item()
        n += xb.size(0)
        all_preds.append(out.argmax(1).cpu().numpy())
        all_targets.append(yb.cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    return running_loss / n, running_acc / n, y_true, y_pred

def plot_curves(history: pd.DataFrame, save_dir: Path, exp_name: str):
    # Accuracy
    plt.figure(figsize=(7,5))
    plt.plot(history["epoch"], history["train_acc"], label="Train Acc")
    plt.plot(history["epoch"], history["test_acc"], label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy - {exp_name}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(save_dir / f"acc_{exp_name}.png", bbox_inches="tight")
    plt.close()

    # Loss
    plt.figure(figsize=(7,5))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss - {exp_name}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(save_dir / f"loss_{exp_name}.png", bbox_inches="tight")
    plt.close()

def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path, exp_name: str):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(10), yticks=np.arange(10),
           xticklabels=list(range(10)), yticklabels=list(range(10)),
           ylabel="True label", xlabel="Predicted label",
           title=f"Confusion Matrix - {exp_name}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # annotate
    fmt = "d"
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# def build_transforms(norm_mean: float, norm_std: float, aug: bool=False):
#     train_tfms = []
#     if aug:
#         train_tfms += [
#             transforms.RandomRotation(10),
#             transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
#             transforms.RandomPerspective(distortion_scale=0.1, p=0.1),
#             transforms.RandomErasing(p=0.2, scale=(0.02,0.08), ratio=(0.3,3.3))
#         ]
#     train_tfms += [transforms.ToTensor(), transforms.Normalize((norm_mean,), (norm_std,))]
#     test_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((norm_mean,), (norm_std,))])
#     return transforms.Compose(train_tfms), test_tfms

def build_transforms(norm_mean: float, norm_std: float, aug: bool = False):
    """
    Two modes:
      - No aug: ToTensor -> Normalize
      - Aug   : RandomRotation (PIL) -> ToTensor -> Normalize
    NOTE: We do NOT use RandomErasing here to avoid tensor-only ops before ToTensor.
    """
    if not aug:
        train_tfms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((norm_mean,), (norm_std,)),
        ])
    else:
        # rotation first (on PIL), then tensor+normalize
        train_tfms = transforms.Compose([
            transforms.RandomRotation((-7.0, 7.0), fill=(0,)),  # 0 keeps MNIST background black
            transforms.ToTensor(),
            transforms.Normalize((norm_mean,), (norm_std,)),
        ])

    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((norm_mean,), (norm_std,)),
    ])
    return train_tfms, test_tfms


def get_data(data_dir: Path, batch_size: int, aug: bool, use_stats_from_train: bool=True):
    # First, load raw ToTensor() for stats and sample pics
    base_train = datasets.MNIST(str(data_dir), train=True, download=True, transform=transforms.ToTensor())
    base_test  = datasets.MNIST(str(data_dir), train=False, download=True, transform=transforms.ToTensor())

    stats = compute_dataset_stats(base_train) if use_stats_from_train else {"mean":0.1307,"std":0.3081,"min":0.0,"max":1.0,"var":0.3081**2}

    # Save sample images (pre-normalization)
    (data_dir / "results" / "plots").mkdir(parents=True, exist_ok=True)
    show_samples(base_train, data_dir / "results" / "plots" / "train_samples.png", title="Train samples")
    show_samples(base_test, data_dir / "results" / "plots" / "test_samples.png", title="Test samples")

    train_tfms, test_tfms = build_transforms(stats["mean"], stats["std"], aug=aug)
    train_ds = datasets.MNIST(str(data_dir), train=True, download=False, transform=train_tfms)
    test_ds  = datasets.MNIST(str(data_dir), train=False, download=False, transform=test_tfms)

    pin = torch.cuda.is_available() or (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=pin, persistent_workers=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin, persistent_workers=False)
    return train_loader, test_loader, stats

def make_optimizer(optim_name: str, params, lr: float, weight_decay: float=0.0, momentum: float=0.9):
    if optim_name.lower() == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim_name.lower() == "sgd":
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {optim_name}")

def run_experiment(exp_cfg: Dict[str, Any], device=DEVICE, results_dir: Path=Path("results")) -> Dict[str, Any]:
    exp_name = exp_cfg["name"]
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, test_loader, stats = get_data(Path("."), batch_size=exp_cfg.get("batch_size", 128), aug=exp_cfg.get("augment", False), use_stats_from_train=True)

    # Model
    model = exp_cfg["model_fn"]().to(device)
    print_model_summary(model, exp_name)
    n_params = count_params(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(exp_cfg.get("optimizer","adam"), model.parameters(), lr=exp_cfg.get("lr", 1e-3), weight_decay=exp_cfg.get("weight_decay", 0.0))
    scheduler = None
    if exp_cfg.get("use_steplr", False):
        scheduler = StepLR(optimizer, step_size=exp_cfg.get("steplr_step", 5), gamma=exp_cfg.get("steplr_gamma", 0.5))

    # Train
    epochs = exp_cfg.get("epochs", 10)
    history_rows = []
    best_acc = -1.0
    best_epoch = -1
    start_time = time.time()
    for epoch in tqdm(range(1, epochs+1), desc=f"[{exp_name}] Training", leave=True):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        history_rows.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "test_loss": te_loss, "test_acc": te_acc, "lr": optimizer.param_groups[0]["lr"]})
        if te_acc > best_acc:
            best_acc = te_acc
            best_epoch = epoch
            # Save checkpoint
            ckpt_dir = Path("checkpoints") / exp_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(),
                        "exp_cfg": exp_cfg,
                        "epoch": epoch,
                        "test_acc": te_acc},
                       ckpt_dir / "best.pt")

        # print(f"[{exp_name}] Epoch {epoch:02d}/{epochs} | Train: loss={tr_loss:.4f} acc={tr_acc*100:.2f}% | Test: loss={te_loss:.4f} acc={te_acc*100:.2f}%")
        tqdm.write(f"[{exp_name}] Epoch {epoch:02d}/{epochs} | "
               f"Train: loss={tr_loss:.4f} acc={tr_acc*100:.2f}% | "
               f"Test:  loss={te_loss:.4f} acc={te_acc*100:.2f}%")

    total_time = time.time() - start_time
    history = pd.DataFrame(history_rows)
    history.to_csv(results_dir / f"metrics_{exp_name}.csv", index=False)
    plot_curves(history, plots_dir, exp_name)
    plot_confusion(y_true, y_pred, plots_dir / f"cm_{exp_name}.png", exp_name)

    # Classification report
    cls_rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(cls_rep).to_csv(results_dir / f"classification_{exp_name}.csv")

    # Summaries
    final_row = {
        "exp_name": exp_name,
        "params": n_params,
        "best_test_acc": best_acc,
        "best_epoch": best_epoch,
        "final_test_acc": history_rows[-1]["test_acc"],
        "epochs": epochs,
        "train_time_sec": total_time,
        "augment": exp_cfg.get("augment", False),
        "optimizer": exp_cfg.get("optimizer", "adam"),
        "lr": exp_cfg.get("lr", 1e-3),
        "use_steplr": exp_cfg.get("use_steplr", False),
    }
    return final_row

def build_experiments(args, selected_prefixes, run_all_families):
    exps = []

    # ===== model1.py =====
    # exps.append({"name": "m1_big", "model_fn": M1.Model1_Big,   "epochs": args.epochs, "optimizer": args.optimizer, "lr": args.lr, "batch_size": args.batch_size})
    # exps.append({"name": "m1_light","model_fn": M1.Model1_Light,"epochs": args.epochs, "optimizer": args.optimizer, "lr": args.lr, "batch_size": args.batch_size})
    # exps.append({"name": "m1_skeleton", "model_fn": M1.Model1_Skeleton, "epochs": args.epochs, "optimizer": args.optimizer, "lr": args.lr, "batch_size": args.batch_size})

    exps.append({"name": "m1_big", "model_fn": M1.Model1_Big,   "epochs": args.epochs, "optimizer": args.optimizer, "lr": 0.01, "batch_size": args.batch_size})
    exps.append({"name": "m1_light","model_fn": M1.Model1_Light,"epochs": args.epochs, "optimizer": args.optimizer, "lr": 0.01, "batch_size": args.batch_size})
    exps.append({"name": "m1_skeleton", "model_fn": M1.Model1_Skeleton, "epochs": args.epochs, "optimizer": args.optimizer, "lr": 0.01, "batch_size": args.batch_size})
    # ===== model2.py =====
    exps.append({"name": "m2_light_bn",        "model_fn": M2.Model2_Light_BN,        "epochs": args.epochs, "optimizer": args.optimizer, "lr": args.lr, "batch_size": args.batch_size})
    exps.append({"name": "m2_light_do",        "model_fn": M2.Model2_Light_DO,        "epochs": args.epochs, "optimizer": args.optimizer, "lr": args.lr, "batch_size": args.batch_size})
    exps.append({"name": "m2_light_bn_do",     "model_fn": M2.Model2_Light_BN_DO,     "epochs": args.epochs, "optimizer": args.optimizer, "lr": args.lr, "batch_size": args.batch_size})
    exps.append({"name": "m2_light_bn_do_gap", "model_fn": M2.Model2_Light_BN_DO_GAP, "epochs": args.epochs, "optimizer": args.optimizer, "lr": args.lr, "batch_size": args.batch_size})

    # ===== model3.py =====
    exps.append({"name": "m3_capacity",           "model_fn": M3.Model3_CapacityUp,  "epochs": min(args.epochs, 15), "optimizer": args.optimizer, "lr": args.lr, "batch_size": args.batch_size})
    exps.append({"name": "m3_poolfix",            "model_fn": M3.Model3_PoolFixed,   "epochs": min(args.epochs, 15), "optimizer": args.optimizer, "lr": args.lr, "batch_size": args.batch_size})
    exps.append({"name": "m3_poolfix_aug",        "model_fn": M3.Model3_PoolFixed,   "epochs": min(args.epochs, 15), "optimizer": args.optimizer, "lr": args.lr, "batch_size": args.batch_size,"augment": True})
    exps.append({"name": "m3_poolfix_steplr",     "model_fn": M3.Model3_PoolFixed,  "epochs": min(args.epochs, 15),  "optimizer": args.optimizer, "lr": args.lr, "batch_size": args.batch_size,                  "use_steplr": True, "steplr_step": 5, "steplr_gamma": 0.5})
    exps.append({"name": "m3_poolfix_aug_steplr", "model_fn": M3.Model3_PoolFixed,  "epochs": min(args.epochs, 15),  "optimizer": args.optimizer, "lr": args.lr, "batch_size": args.batch_size, "augment": True, "use_steplr": True, "steplr_step": 5, "steplr_gamma": 0.5})
    exps.append({"name": "m3_depthwiseConv_aug_steplr","model_fn": M3.Model3_TinyTarget,  "epochs": min(args.epochs, 15), "optimizer": args.optimizer, "lr": args.lr, "batch_size": args.batch_size, "augment": True, "use_steplr": True, "steplr_step": 5, "steplr_gamma": 0.5})

    # Family filter
    if not run_all_families:
        exps = [e for e in exps if any(e["name"].startswith(pfx) for pfx in selected_prefixes)]

    # --only filter (intersection with the above)
    if args.only:
        only_names = {n.strip() for n in args.only.split(",") if n.strip()}
        exps = [e for e in exps if e["name"] in only_names]

    return exps


def main():
    parser = argparse.ArgumentParser()
    # family selector (positional, optional): supports "all_model", "model1", "model2", "model3"
    parser.add_argument("family", nargs="?", choices=["all_model", "model1", "model2", "model3"],
                        help="Run a specific model family or all_model. If omitted, defaults to all_model.")

    # convenience flags (optional). These OR with the positional arg.
    parser.add_argument("--model1", action="store_true", help="Run all model1 experiments")
    parser.add_argument("--model2", action="store_true", help="Run all model2 experiments")
    parser.add_argument("--model3", action="store_true", help="Run all model3 experiments")

    # run-filter within the selected family/families
    parser.add_argument("--only", type=str, default="", help="Comma-separated experiment names to run (e.g., m1_big,m2_light_bn)")

    # training args
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per run")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam","sgd"])
    parser.add_argument("--lr", type=float, default=1e-3 ) # default= 0.05 for sgd
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

        # Determine selected families
    selected_prefixes = set()

    if args.family in (None, "all_model"):
        # default to all if nothing else chosen
        pass
    elif args.family == "model1":
        selected_prefixes.add("m1_")
    elif args.family == "model2":
        selected_prefixes.add("m2_")
    elif args.family == "model3":
        selected_prefixes.add("m3_")

    # OR with explicit flags
    if args.model1: selected_prefixes.add("m1_")
    if args.model2: selected_prefixes.add("m2_")
    if args.model3: selected_prefixes.add("m3_")

    # If still none specified, run all families
    run_all_families = (len(selected_prefixes) == 0)

    set_seed(args.seed)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run experiments
    summaries = []
    exps = build_experiments(args, selected_prefixes, run_all_families)
    for cfg in exps:
        row = run_experiment(cfg, device=DEVICE, results_dir=results_dir)
        summaries.append(row)

    # Save summary
    if summaries:
        df = pd.DataFrame(summaries)
        df.sort_values(by="best_test_acc", ascending=False, inplace=True)
        df.to_csv(results_dir / "summary.csv", index=False)
        print("\n=== Summary ===")
        print(df.to_string(index=False))

if __name__ == "__main__":
    
    main()
