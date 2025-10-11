import argparse
from dataclasses import dataclass
import os
import re
from typing import Dict, Optional, Tuple, Any

os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["DISABLE_TORCH_COMPILE"] = "1"
os.environ["TRANSFORMERS_NO_COMPILE"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from numpy.typing import NDArray
from sklearn.utils import shuffle as sk_shuffle
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, pearsonr


# Optional: avoid error spam from Torch Dynamo
torch._dynamo.config.suppress_errors = False

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
# random.seed(SEED)
# np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# torch.use_deterministic_algorithms(True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adaptive steering with (B, D) hidden states (already pooled):

- Steering direction v from D2 = mean(toxic) - mean(nontoxic), UNNORMALIZED
- Linear probe (SGDClassifier) trained on D1
- Adaptive alpha via 1-D logistic calibration on s = ((h - m) · v) / ||v||
- Steering step uses unnormalized v but scales by 1/||v|| for stability
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score


# -----------------------------
# Data containers
# -----------------------------

@dataclass
class DatasetPack:
    """
    Container for a single dataset at one layer:
    h: (B, D) hidden states (already pooled)
    y: (B,) labels in {0,1}
    groups: (B,) optional grouping ids for split (e.g., prompt_id). If you don't have them, pass np.arange(B).
    """
    h: torch.Tensor          # (B, D)
    y: torch.Tensor          # (B,)
    groups: np.ndarray       # (B,)


# -----------------------------
# Loaders (plug your I/O here)
# -----------------------------

def load_dataset_D1() -> DatasetPack:
    """
    TODO: Replace with your real loading.
    Must return (B, D) hidden states and labels for the SAME layer as D2.
    """
    raise NotImplementedError("Implement loading for D1 here.")

def load_dataset_D2() -> DatasetPack:
    """
    TODO: Replace with your real loading.
    """
    raise NotImplementedError("Implement loading for D2 here.")


# -----------------------------
# Steering vector from D2
# -----------------------------

@dataclass
class SteeringPack:
    v: torch.Tensor        # (D,)   steering direction (unnormalized)
    v_norm: float          # scalar ||v||
    midpoint: torch.Tensor # (D,)   0.5*(mu_pos + mu_neg)
    mu_pos: torch.Tensor   # (D,)
    mu_neg: torch.Tensor   # (D,)

def compute_steering_from_D2(D2: DatasetPack) -> SteeringPack:
    """
    v = mean_pos - mean_neg from D2 (B, D).
    """
    with torch.no_grad():
        h = D2.h
        y = D2.y.bool()
        assert (y.sum() > 0) and ((~y).sum() > 0), "D2 must contain both classes."

        mu_pos = h[y].mean(dim=0)      # (D,)
        mu_neg = h[~y].mean(dim=0)     # (D,)
        v = mu_pos - mu_neg            # (D,)   <-- UNNORMALIZED
        v_norm = float(v.norm(p=2).clamp_min(1e-8))
        midpoint = 0.5 * (mu_pos + mu_neg)

        return SteeringPack(v=v, v_norm=v_norm, midpoint=midpoint,
                            mu_pos=mu_pos, mu_neg=mu_neg)


# -----------------------------
# Probe on D1 (optional metrics)
# -----------------------------

@dataclass
class ProbePack:
    scaler: StandardScaler
    clf: SGDClassifier

def train_probe_on_D1(D1: DatasetPack) -> Tuple[ProbePack, Dict[str, float]]:
    """
    Train an SGDClassifier on D1 (B, D).
    Uses grouped split if groups are provided; else fakes with arange.
    """
    X = D1.h.cpu().numpy()          # (N, D)
    y = D1.y.cpu().numpy().astype(int)
    groups = D1.groups if D1.groups is not None else np.arange(len(y))

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups))

    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4,
                        early_stopping=True, n_iter_no_change=5,
                        class_weight="balanced", random_state=42)
    clf.fit(Xtr_s, ytr)

    proba = clf.predict_proba(Xte_s)[:, list(clf.classes_).index(1)]
    auc = roc_auc_score(yte, proba)
    ap  = average_precision_score(yte, proba)
    acc = (clf.predict(Xte_s) == yte).mean()

    return ProbePack(scaler=scaler, clf=clf), {"auc": float(auc), "ap": float(ap), "acc": float(acc)}


# -----------------------------
# 1-D logistic calibration on D1
# -----------------------------

@dataclass
class CalibPack:
    a: float
    b: float

def fit_1d_calibration_on_D1(D1: DatasetPack, steer: SteeringPack) -> CalibPack:
    """
    Fit p(y=1 | s) = sigmoid(a * s + b) on a small calibration split of D1,
    where s = ((h - midpoint) · v) / ||v||, with (B, D) features.
    """
    H = D1.h
    y = D1.y.cpu().numpy().astype(int)
    groups = D1.groups if D1.groups is not None else np.arange(len(y))

    m = steer.midpoint.to(H.device)  # (D,)
    v = steer.v.to(H.device)         # (D,)
    s = ((H - m) @ v) / steer.v_norm # (B,)
    s_np = s.cpu().numpy().reshape(-1, 1)

    # small split for calibration (20%)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
    tr_idx, cal_idx = next(gss.split(s_np, y, groups))
    s_cal, y_cal = s_np[cal_idx], y[cal_idx]

    lr = LogisticRegression(solver="lbfgs")
    lr.fit(s_cal, y_cal)
    a = float(lr.coef_[0, 0])
    b = float(lr.intercept_[0])
    return CalibPack(a=a, b=b)


# -----------------------------
# Runtime adaptive steering (unnormalized v)
# -----------------------------

# def adaptive_step_prob(h: torch.Tensor,
#                        steer: SteeringPack,
#                        calib: CalibPack,
#                        gamma: float = 2.0,
#                        tau: float = 0.5,
#                        alpha_max: float = 5.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     h:     (..., D) hidden state(s) to steer
#     steer: contains v (unnormalized), v_norm, midpoint
#     calib: a,b for 1-D logistic calibration on s
#     returns: (h_steered, p, alpha)
#     """
#     v = steer.v.to(h.device)               # (D,)
#     m = steer.midpoint.to(h.device)        # (D,)
#     v_norm = steer.v_norm + 1e-8

#     # 1-D score
#     s = ((h - m) @ v) / v_norm             # (...,)

#     # calibrated prob
#     p = torch.sigmoid(calib.a * s + calib.b)

#     # adaptive alpha
#     alpha = gamma * torch.clamp(p - tau, min=0.0)
#     alpha = torch.clamp(alpha, max=alpha_max)  # (...,)

#     # unnormalized v; scale step by 1/||v|| for stable meaning of alpha
#     delta = -(alpha / v_norm).unsqueeze(-1) * v  # (..., D)
#     h_steered = h + delta
#     return h_steered, p, alpha

def adaptive_step_prob(h: torch.Tensor,
                       steer: SteeringPack,
                       calib: CalibPack,
                       gamma: float = 2.0,
                       tau: float = 0.5,
                       alpha_max: float = 5.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    h:     (..., D) hidden state(s) to steer
    steer: contains v (unnormalized), v_norm, midpoint
    calib: a,b for 1-D logistic calibration on s
    returns: (h_steered, p, alpha)
    """
    v = steer.v.to(h.device)               # (D,)
    m = steer.midpoint.to(h.device)        # (D,)
    v_norm = steer.v_norm + 1e-8

    # 1-D score for calibration
    s = ((h - m) @ v) / v_norm             # normalized projection (dimensionless)

    # calibrated probability
    p = torch.sigmoid(calib.a * s + calib.b)

    # adaptive alpha (sample-dependent)
    alpha = gamma * torch.clamp(p - tau, min=0.0)
    alpha = torch.clamp(alpha, max=alpha_max)  # (...,)

    # ⚠️ steering with unnormalized v (same as your current implementation)
    delta = -alpha.unsqueeze(-1) * v       # (..., D)
    h_steered = h + delta
    return h_steered, p, alpha



# -----------------------------
# Main driver (outline)
# -----------------------------

def main():
    # 1) Load datasets (B, D)
    D2 = load_dataset_D2()                    # for steering vector
    D1 = load_dataset_D1()                    # for probe + calibration

    # 2) Steering vector from D2 (unnormalized)
    steer = compute_steering_from_D2(D2)
    print(f"[steer] ||v|| = {steer.v_norm:.4f}")

    # 3) (Optional) train linear probe on D1 for reporting
    probe, metrics = train_probe_on_D1(D1)
    print(f"[probe@D1] AUC={metrics['auc']:.3f}  AP={metrics['ap']:.3f}  ACC={metrics['acc']:.3f}")

    # 4) Fit 1-D logistic calibration on D1 (tiny split)
    calib = fit_1d_calibration_on_D1(D1, steer)
    print(f"[calib] a={calib.a:.4f}  b={calib.b:.4f}")

    # 5) Example: steer a batch from D1
    H = D1.h
    Hs, p, alpha = adaptive_step_prob(H, steer, calib, gamma=2.0, tau=0.5, alpha_max=5.0)
    print(f"[runtime] p: {p.min().item():.3f}..{p.max().item():.3f} ; alpha mean={alpha.mean().item():.3f}")

    # If you need the probe’s scaler for later use:
    _ = probe  # scaler in probe.scaler, classifier in probe.clf


if __name__ == "__main__":
    main()
