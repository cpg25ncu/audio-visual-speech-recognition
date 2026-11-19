"""Minimal, fast training loop.

The whole dataset fits in GPU memory (a few hundred clips), so batches are drawn by indexing
pre-loaded tensors instead of using a DataLoader. Multi-condition training samples, for every
clip and every epoch, one of several pre-computed noisy versions of that clip ("feature bank").
Model selection uses validation accuracy with early stopping; the test split is never touched
here.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    epochs: int = 80
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 20
    label_smoothing: float = 0.05
    time_shift: float = 0.1          # max fraction of the sequence to shift (with zero fill)
    time_mask: float = 0.1           # max fraction of frames masked once per sample
    feat_mask: float = 0.1           # max fraction of feature channels masked once per sample
    device: str = "auto"
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, tcfg: dict, **overrides) -> TrainConfig:
        keys = {"epochs", "batch_size", "lr", "weight_decay", "patience", "label_smoothing", "device"}
        kw = {k: v for k, v in tcfg.items() if k in keys}
        kw.update(overrides)
        return cls(**kw)


def augment(x: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    """Random time shift + one time mask + one feature-channel mask. ``x``: (B, D, T) or (B, T, H, W)."""
    if x.dim() == 4:                    # ROI sequences: shift in time and random brightness/contrast
        B, T = x.shape[:2]
        shift = int(cfg.time_shift * T)
        if shift:
            x = torch.roll(x, int(torch.randint(-shift, shift + 1, (1,))), dims=1)
        gain = 1 + 0.2 * (torch.rand(B, 1, 1, 1, device=x.device) - 0.5)
        bias = 0.1 * (torch.rand(B, 1, 1, 1, device=x.device) - 0.5)
        return (x * gain + bias).clamp(0, 1)
    B, D, T = x.shape
    x = x.clone()
    if cfg.time_shift > 0:
        shift = int(cfg.time_shift * T)
        s = int(torch.randint(-shift, shift + 1, (1,)))
        x = torch.roll(x, s, dims=2)
        if s > 0:
            x[:, :, :s] = 0
        elif s < 0:
            x[:, :, s:] = 0
    if cfg.time_mask > 0:
        w = torch.randint(0, max(1, int(cfg.time_mask * T)) + 1, (B,), device=x.device)
        t0 = (torch.rand(B, device=x.device) * (T - w).clamp(min=1)).long()
        ar = torch.arange(T, device=x.device)[None]
        x = x.masked_fill(((ar >= t0[:, None]) & (ar < (t0 + w)[:, None]))[:, None, :], 0.0)
    if cfg.feat_mask > 0 and D > 4:
        w = torch.randint(0, max(1, int(cfg.feat_mask * D)) + 1, (B,), device=x.device)
        f0 = (torch.rand(B, device=x.device) * (D - w).clamp(min=1)).long()
        ar = torch.arange(D, device=x.device)[None]
        x = x.masked_fill(((ar >= f0[:, None]) & (ar < (f0 + w)[:, None]))[:, :, None], 0.0)
    return x


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray | torch.Tensor, device: torch.device | None = None,
                  batch_size: int = 128) -> np.ndarray:
    device = device or next(model.parameters()).device
    model.eval()
    X = torch.as_tensor(X, dtype=torch.float32)
    out = []
    for i in range(0, len(X), batch_size):
        out.append(torch.softmax(model(X[i:i + batch_size].to(device)), dim=1).cpu())
    return torch.cat(out).numpy()


def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    return float((probs.argmax(1) == y).mean())


def fit(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
        cfg: TrainConfig, seed: int = 0, verbose: bool = False) -> tuple[nn.Module, dict]:
    """Train ``model``.

    ``X_train`` is ``(N, ...)`` or a feature bank ``(N, V, ...)`` when ``extra['bank']`` is True:
    each epoch draws one of the V versions per clip.
    """
    set_seed(seed)
    device = resolve_device(cfg.device)
    model = model.to(device)
    bank = bool(cfg.extra.get("bank", False))
    Xt = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_train, dtype=torch.long, device=device)
    Xv = torch.as_tensor(X_val, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = int(np.ceil(len(yt) / cfg.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, epochs=cfg.epochs,
                                                steps_per_epoch=steps_per_epoch, pct_start=0.15)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    history = {"train_loss": [], "train_acc": [], "val_acc": [], "val_nll": []}
    best_acc, best_nll, best_state, best_epoch, waited = -1.0, np.inf, None, 0, 0
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(yt), device=device)
        tot_loss, tot_correct = 0.0, 0
        for i in range(steps_per_epoch):
            idx = perm[i * cfg.batch_size:(i + 1) * cfg.batch_size]
            if bank:
                v = torch.randint(0, Xt.shape[1], (len(idx),), device=device)
                xb = Xt[idx, v]
            else:
                xb = Xt[idx]
            xb, yb = augment(xb, cfg), yt[idx]
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot_loss += float(loss.detach()) * len(idx)
            tot_correct += int((logits.argmax(1) == yb).sum())
        pv = predict_proba(model, Xv, device)
        val_acc = accuracy(pv, y_val)
        val_nll = float(-np.log(pv[np.arange(len(y_val)), y_val] + 1e-8).mean())
        history["train_loss"].append(tot_loss / len(yt))
        history["train_acc"].append(tot_correct / len(yt))
        history["val_acc"].append(val_acc)
        history["val_nll"].append(val_nll)
        if verbose:
            print(f"epoch {epoch:3d} loss {tot_loss / len(yt):.3f} val {val_acc:.3f} nll {val_nll:.3f}")
        # select on accuracy, break ties (common once validation saturates) with lower NLL
        better_acc = val_acc > best_acc + 1e-9
        tie_lower_nll = abs(val_acc - best_acc) <= 1e-9 and val_nll < best_nll
        significant = better_acc or (tie_lower_nll and val_nll < best_nll - 1e-3)
        if better_acc or tie_lower_nll:
            best_acc, best_nll, best_epoch = val_acc, val_nll, epoch
            best_state = copy.deepcopy(model.state_dict())
        waited = 0 if significant else waited + 1
        if waited >= cfg.patience:
            break
    model.load_state_dict(best_state)
    history.update(best_val_acc=best_acc, best_epoch=best_epoch)
    return model, history
