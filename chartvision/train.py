"""Training loop with a strict time-based split and honest baselines."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import CLASS_NAMES, Config
from .dataset import (ChartWindowDataset, Sample, class_weights, label_distribution,
                      purge_overlap, time_split)
from .model import ChartCNN, multitask_loss

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _evaluate(model: ChartCNN, loader: DataLoader, device: torch.device, cfg: Config) -> dict:
    model.eval()
    n = 0
    correct = np.zeros(len(cfg.horizons))
    confusion = np.zeros((len(cfg.horizons), 3, 3), dtype=np.int64)
    path_se = 0.0
    with torch.no_grad():
        for img, labels, path in loader:
            img = img.to(device)
            logits, pred_path = model(img)
            pred = logits.argmax(-1).cpu().numpy()
            lab = labels.numpy()
            for h in range(len(cfg.horizons)):
                correct[h] += (pred[:, h] == lab[:, h]).sum()
                for t, p in zip(lab[:, h], pred[:, h]):
                    confusion[h, t, p] += 1
            path_se += float(((pred_path.cpu() - path) ** 2).sum())
            n += len(img)
    out = {
        "n": n,
        "accuracy": {h: float(correct[i] / max(n, 1)) for i, h in enumerate(cfg.horizons)},
        "confusion": {h: confusion[i].tolist() for i, h in enumerate(cfg.horizons)},
        "path_rmse": float(np.sqrt(path_se / max(n * cfg.path_horizon, 1))),
    }
    # precision of the "up" call: when the model says up, how often is it up?
    for i, h in enumerate(cfg.horizons):
        col = confusion[i][:, 2]
        out.setdefault("up_precision", {})[h] = float(col[2] / col.sum()) if col.sum() else float("nan")
        col = confusion[i][:, 0]
        out.setdefault("down_precision", {})[h] = float(col[0] / col.sum()) if col.sum() else float("nan")
    return out


def _baselines(val: list[Sample], cfg: Config) -> dict:
    """Two sanity baselines: majority class, and 'momentum' (the MA keeps its
    current direction)."""
    res = {}
    for i, h in enumerate(cfg.horizons):
        labels = np.array([s.labels[i] for s in val])
        majority = np.bincount(labels, minlength=3).argmax()
        maj_acc = float((labels == majority).mean())
        # momentum: sign of the MA slope over the last 5 days of the window
        mom_pred = []
        for s in val:
            ma = s.series[:, 0]
            slope = ma[-1] / ma[-6] - 1.0
            mom_pred.append(2 if slope > cfg.flat_threshold else 0 if slope < -cfg.flat_threshold else 1)
        mom_acc = float((np.array(mom_pred) == labels).mean())
        res[h] = {"majority_class": CLASS_NAMES[majority], "majority_acc": maj_acc,
                  "momentum_acc": mom_acc}
    return res


def train(samples: list[Sample], cfg: Config, device: torch.device | None = None) -> Path:
    set_seed(cfg.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_s, val_s, cutoff = time_split(samples, cfg.val_fraction_days)
    train_s = purge_overlap(train_s, cutoff, max(max(cfg.horizons), cfg.path_horizon))
    log.info("split at %s: %d train windows, %d validation windows",
             cutoff.date(), len(train_s), len(val_s))
    log.info("train label distribution: %s", json.dumps(label_distribution(train_s, cfg.horizons)))
    log.info("val   label distribution: %s", json.dumps(label_distribution(val_s, cfg.horizons)))

    train_loader = DataLoader(ChartWindowDataset(train_s, cfg), batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    val_loader = DataLoader(ChartWindowDataset(val_s, cfg), batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers)

    if device.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    model = ChartCNN(len(cfg.channels), len(cfg.horizons), cfg.path_horizon,
                     width=cfg.model_width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr,
                                                total_steps=cfg.epochs * len(train_loader))
    cw = class_weights(train_s, len(cfg.horizons)).to(device)

    best_acc, best_state, history = -1.0, None, []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, tot, parts = time.time(), 0.0, {"cls": 0.0, "path": 0.0}
        for img, labels, path in train_loader:
            img, labels, path = img.to(device), labels.to(device), path.to(device)
            logits, pred_path = model(img)
            loss, p = multitask_loss(logits, pred_path, labels, path, cw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
            for k in parts:
                parts[k] += p[k]
        nb = len(train_loader)
        ev = _evaluate(model, val_loader, device, cfg)
        mean_acc = float(np.mean(list(ev["accuracy"].values())))
        history.append({"epoch": epoch, "train_loss": tot / nb,
                        "train_cls": parts["cls"] / nb, "train_path": parts["path"] / nb,
                        "val_accuracy": ev["accuracy"], "val_path_rmse": ev["path_rmse"]})
        log.info("epoch %2d  loss %.4f (cls %.4f path %.4f)  val acc %s  path rmse %.4f  [%.0fs]",
                 epoch, tot / nb, parts["cls"] / nb, parts["path"] / nb,
                 {h: round(a, 3) for h, a in ev["accuracy"].items()}, ev["path_rmse"], time.time() - t0)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    final = _evaluate(model, val_loader, device, cfg)
    base = _baselines(val_s, cfg)
    log.info("best validation accuracy: %s", final["accuracy"])
    log.info("baselines: %s", json.dumps(base))

    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    ckpt = cfg.model_dir / "chart_cnn.pt"
    torch.save({
        "state_dict": best_state,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(cfg).items()},
        "cutoff": str(cutoff.date()),
        "validation": final,
        "baselines": base,
        "history": history,
        "class_names": CLASS_NAMES,
    }, ckpt)
    with open(cfg.model_dir / "training_report.json", "w") as f:
        json.dump({"cutoff": str(cutoff.date()), "n_train": len(train_s), "n_val": len(val_s),
                   "validation": final, "baselines": base, "history": history}, f, indent=2)
    log.info("saved %s", ckpt)
    return ckpt


def load_model(ckpt_path: Path, device: torch.device | None = None) -> tuple[ChartCNN, dict]:
    device = device or torch.device("cpu")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ck["config"]
    model = ChartCNN(len(c["channels"]), len(c["horizons"]), c["path_horizon"],
                     width=int(c.get("model_width", 16))).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
