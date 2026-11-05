"""Training / evaluation with a strict time-based split and honest baselines.

* :func:`fit`            - train a model on given train/val samples, return it
* :func:`predict_samples` - probabilities, scores and paths for a sample list
* :func:`evaluate`       - full metric report (CNN + baselines) on a sample list
* :func:`train`          - the simple one-split entry used by scripts/3_train.py
* :func:`walk_forward`   - expanding-window cross validation
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from .config import CLASS_NAMES, Config
from .dataset import (ChartWindowDataset, Sample, Thresholds, assign_labels, class_weights,
                      compute_thresholds, flat_features, label_distribution, purge_overlap,
                      split_at, time_split, walk_forward_folds)
from .metrics import summarize_horizon
from .model import ChartCNN, multitask_loss

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    return device


def _loader(samples: list[Sample], cfg: Config, shuffle: bool) -> DataLoader:
    return DataLoader(ChartWindowDataset(samples, cfg), batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers, drop_last=shuffle and len(samples) > cfg.batch_size)


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------
@torch.no_grad()
def predict_samples(model: ChartCNN, samples: list[Sample], cfg: Config,
                    device: torch.device | None = None) -> dict[str, np.ndarray]:
    device = device or next(model.parameters()).device
    model.eval()
    probs, paths = [], []
    for img, _, _ in _loader(samples, cfg, shuffle=False):
        logits, path = model(img.to(device))
        probs.append(torch.softmax(logits, -1).cpu().numpy())
        paths.append(path.cpu().numpy())
    probs = np.concatenate(probs) if probs else np.zeros((0, len(cfg.horizons), 3))
    paths = np.concatenate(paths) if paths else np.zeros((0, cfg.path_horizon))
    return {
        "probs": probs,                                   # [N, H, 3]
        "pred": probs.argmax(-1),                         # [N, H]
        "score": probs[:, :, 2] - probs[:, :, 0],         # [N, H]  p_up - p_down
        "path": paths,
        "labels": np.stack([s.labels for s in samples]) if samples else np.zeros((0, len(cfg.horizons)), dtype=int),
        "returns": np.stack([s.future_returns for s in samples]) if samples else np.zeros((0, len(cfg.horizons))),
        "dates": np.array([s.end_date for s in samples], dtype="datetime64[ns]"),
        "momentum": np.array([s.momentum for s in samples], dtype=np.float32),
        "tickers": np.array([s.ticker for s in samples]),
    }


def evaluate(out: dict[str, np.ndarray], cfg: Config, thr: Thresholds,
             lr_pred: np.ndarray | None = None, lr_score: np.ndarray | None = None) -> dict:
    """Metric report per horizon from the arrays of :func:`predict_samples`."""
    report: dict = {"n": int(len(out["labels"])), "horizons": {}}
    for i, h in enumerate(cfg.horizons):
        m = summarize_horizon(out["labels"][:, i], out["pred"][:, i], out["score"][:, i],
                              out["returns"][:, i], out["dates"], h, out["momentum"], thr[h])
        if lr_pred is not None:
            m["logreg_acc"] = float((lr_pred[:, i] == out["labels"][:, i]).mean())
            m["logreg_signal"] = __import__("chartvision.metrics", fromlist=["signal_metrics"]) \
                .signal_metrics(lr_score[:, i], out["returns"][:, i], out["dates"], h)
        report["horizons"][h] = m
    return report


def logreg_baseline(train: list[Sample], val: list[Sample], cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """A linear model on the same (normalised) numbers the image is drawn from."""
    Xtr, Xva = flat_features(train), flat_features(val)
    preds, scores = [], []
    for i in range(len(cfg.horizons)):
        ytr = np.array([s.labels[i] for s in train])
        clf = LogisticRegression(max_iter=300, C=0.1, n_jobs=-1)
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xva)
        preds.append(p.argmax(1))
        scores.append(p[:, 2] - p[:, 0] if p.shape[1] == 3 else np.zeros(len(Xva)))
    return np.stack(preds, 1), np.stack(scores, 1)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def fit(train_s: list[Sample], val_s: list[Sample], cfg: Config, device: torch.device | None = None,
        init_state: dict | None = None, lr: float | None = None, epochs: int | None = None,
        select_on: str = "val", verbose: bool = True) -> tuple[ChartCNN, list[dict]]:
    """Train a ChartCNN. Labels must already be assigned.

    ``init_state`` warm-starts from an existing model (fine-tuning). The
    returned model is the epoch with the best mean validation accuracy
    (``select_on="val"``) or simply the last epoch (``select_on="last"``,
    which is the honest choice when the validation set is also the test set).
    """
    device = device or get_device()
    epochs = epochs or cfg.epochs
    lr = lr or cfg.lr
    set_seed(cfg.seed)

    model = ChartCNN(len(cfg.channels), len(cfg.horizons), cfg.path_horizon, width=cfg.model_width).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    train_loader = _loader(train_s, cfg, shuffle=True)
    if len(train_loader) == 0:
        return model, []
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(train_loader))
    cw = class_weights(train_s, len(cfg.horizons)).to(device)

    best, best_state, history = -1.0, None, []
    for epoch in range(1, epochs + 1):
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
            tot += loss.item()
            for k in parts:
                parts[k] += p[k]
        nb = len(train_loader)
        rec = {"epoch": epoch, "train_loss": tot / nb, "train_cls": parts["cls"] / nb,
               "train_path": parts["path"] / nb, "seconds": time.time() - t0}
        if val_s:
            out = predict_samples(model, val_s, cfg, device)
            acc = {h: float((out["pred"][:, i] == out["labels"][:, i]).mean()) for i, h in enumerate(cfg.horizons)}
            rec["val_accuracy"] = acc
            mean_acc = float(np.mean(list(acc.values())))
            if select_on == "val" and mean_acc > best:
                best, best_state = mean_acc, copy.deepcopy(model.state_dict())
        history.append(rec)
        if verbose:
            log.info("epoch %2d  loss %.4f (cls %.4f path %.4f)  val acc %s  [%.0fs]", epoch, rec["train_loss"],
                     rec["train_cls"], rec["train_path"],
                     {h: round(a, 3) for h, a in rec.get("val_accuracy", {}).items()}, rec["seconds"])
    if select_on == "val" and best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def run_split(samples: list[Sample], cfg: Config, cutoff, end=None, device=None,
              with_logreg: bool = True, select_on: str = "last", verbose: bool = True,
              init_state: dict | None = None, lr: float | None = None, epochs: int | None = None):
    """Train on everything before ``cutoff`` and evaluate on [cutoff, end)."""
    train_s, val_s = split_at(samples, cutoff, end)
    train_s = purge_overlap(train_s, cutoff, max(max(cfg.horizons), cfg.path_horizon))
    thr = compute_thresholds(train_s, cfg)
    assign_labels(train_s, thr, cfg.horizons)
    assign_labels(val_s, thr, cfg.horizons)
    if verbose:
        log.info("cutoff %s: %d train / %d val windows; thresholds %s", cutoff.date() if hasattr(cutoff, "date") else cutoff,
                 len(train_s), len(val_s), {h: tuple(round(x, 4) for x in t) for h, t in thr.items()})
        log.info("train labels %s", json.dumps(label_distribution(train_s, cfg.horizons)))
    model, history = fit(train_s, val_s, cfg, device, init_state=init_state, lr=lr, epochs=epochs,
                         select_on=select_on, verbose=verbose)
    out = predict_samples(model, val_s, cfg, device)
    lr_pred = lr_score = None
    if with_logreg and len(train_s) and len(val_s):
        lr_pred, lr_score = logreg_baseline(train_s, val_s, cfg)
    report = evaluate(out, cfg, thr, lr_pred, lr_score)
    report["cutoff"] = str(getattr(cutoff, "date", lambda: cutoff)())
    report["n_train"] = len(train_s)
    report["thresholds"] = {h: list(t) for h, t in thr.items()}
    report["history"] = history
    return model, report, out, thr


def compact(report: dict) -> dict:
    """The handful of numbers worth printing per horizon."""
    rows = {}
    for h, m in report["horizons"].items():
        rows[h] = {
            "acc": round(m["accuracy"], 3),
            "bal_acc": round(m["balanced_accuracy"], 3),
            "majority": round(m["majority_acc"], 3),
            "momentum": round(m["momentum_acc"], 3),
            "logreg": round(m["logreg_acc"], 3) if "logreg_acc" in m else None,
            "ic": round(m["ic"], 3),
            "mom_ic": round(m["momentum_ic"], 3),
            "spread_pct": round(m["spread_mean_pct"], 3) if "spread_mean_pct" in m else None,
            "spread_t": round(m["spread_t"], 2) if "spread_t" in m else None,
            "up_prec": round(m["up_precision"], 3),
            "down_prec": round(m["down_precision"], 3),
        }
    return rows


def train(samples: list[Sample], cfg: Config, device: torch.device | None = None) -> Path:
    """One time split, save checkpoint + report (used by scripts/3_train.py)."""
    device = device or get_device()
    _, _, cutoff = time_split(samples, cfg.val_fraction_days)
    model, report, _, thr = run_split(samples, cfg, cutoff, device=device, select_on="val")
    log.info("validation summary: %s", json.dumps(compact(report), default=str))

    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    ckpt = cfg.model_dir / "chart_cnn.pt"
    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(cfg).items()},
        "thresholds": {int(h): list(t) for h, t in thr.items()},
        "cutoff": report["cutoff"],
        "validation": report,
        "class_names": CLASS_NAMES,
    }, ckpt)
    with open(cfg.model_dir / "training_report.json", "w") as f:
        json.dump({"cutoff": report["cutoff"], "n_train": report["n_train"], "n_val": report["n"],
                   "summary": compact(report), "validation": report}, f, indent=2, default=str)
    log.info("saved %s", ckpt)
    return ckpt


def walk_forward(samples: list[Sample], cfg: Config, n_folds: int = 3, device=None,
                 with_logreg: bool = True, verbose: bool = False) -> dict:
    """Expanding-window CV. Returns per-fold compact tables and the pooled
    metrics over all out-of-sample windows."""
    device = device or get_device()
    folds = walk_forward_folds(samples, n_folds)
    fold_reports, pooled = [], []
    for k, (cutoff, end) in enumerate(folds):
        _, report, out, thr = run_split(samples, cfg, cutoff, end, device, with_logreg, "last", verbose)
        log.info("fold %d (%s -> %s): %s", k + 1, cutoff.date(), end.date() if end is not None else "end",
                 json.dumps(compact(report), default=str))
        fold_reports.append({"cutoff": str(cutoff.date()), "summary": compact(report), "n": report["n"]})
        pooled.append((out, thr))
    # pool out-of-sample predictions (thresholds differ slightly per fold; labels were set per fold)
    keys = ["pred", "score", "labels", "returns", "dates", "momentum"]
    merged = {k: np.concatenate([o[k] for o, _ in pooled]) for k in keys}
    thr_mean = {h: (float(np.mean([t[h][0] for _, t in pooled])), float(np.mean([t[h][1] for _, t in pooled])))
                for h in cfg.horizons}
    pooled_report = evaluate(merged, cfg, thr_mean)
    return {"folds": fold_reports, "pooled": compact(pooled_report), "pooled_full": pooled_report}


def load_model(ckpt_path: Path, device: torch.device | None = None) -> tuple[ChartCNN, dict]:
    device = device or torch.device("cpu")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ck["config"]
    model = ChartCNN(len(c["channels"]), len(c["horizons"]), c["path_horizon"],
                     width=int(c.get("model_width", 16))).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
