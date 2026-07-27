#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_utils.py

Rigorous evaluation utilities for the Spectral-LDM augmentation experiments.

This module fixes two problems present in the original experiment scripts:

  (1) TEST-SET LEAKAGE via early stopping.
      The original code called:
          model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], early_stopping_rounds=20)
      which selects the number of boosting rounds by watching the held-out TEST
      set. The number of trees is a real hyperparameter, so this leaks the test
      set into model selection and contradicts the paper's claim that no test
      samples are used during classification.

      Fix: `train_xgb_no_leak` carves an internal validation split out of the
      TRAINING data (stratified) and uses THAT for early stopping. The test set
      is only ever touched for final scoring.

  (2) NO UNCERTAINTY QUANTIFICATION.
      The original code reported a single number per metric from a single
      seed / single split, which is exactly the basis reviewers used to argue
      the gains might be within noise.

      Fix: this module provides
        - bootstrap confidence intervals on the test set (resample test rows),
        - multi-seed repetition (variance from model-training stochasticity),
        - a DeLong test for comparing two AUCs on the same test set.

These utilities are deliberately model-agnostic where possible and have no
dependency on the diffusion code, so they can be imported by
experiment_balancing.py, augmentation-benchmark.py, and the vanilla-compare
scripts alike.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from xgboost import XGBClassifier


# =====================================================================
# 1. LEAK-FREE XGBOOST TRAINING
# =====================================================================

def make_xgb(seed: int, scale_pos_weight: float, n_estimators: int = 400,
             early_stopping_rounds: int = 20):
    """Construct an XGBClassifier configured for early stopping.

    Note n_estimators is set generously (400) because with a PROPER internal
    validation set, early stopping will pick the real optimum well below the
    cap. In the leaky original, 200 trees were used with the test set as the
    early-stopping monitor.
    """
    return XGBClassifier(
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric="logloss",
        tree_method="hist",
    )


def train_xgb_no_leak(X_tr: np.ndarray, y_tr: np.ndarray,
                      seed: int = 42,
                      val_ratio: float = 0.15,
                      n_estimators: int = 400,
                      early_stopping_rounds: int = 20):
    """Train XGBoost with early stopping on an INTERNAL validation split.

    The test set is never passed to this function. Early stopping monitors a
    stratified hold-out carved from the training data only.

    Returns the fitted model and the number of trees actually used.
    """
    # Stratified internal validation split from TRAIN only.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr,
        test_size=val_ratio,
        random_state=seed,
        stratify=y_tr,
    )

    n_healthy = int(np.sum(y_fit == 0))
    n_cancer = int(np.sum(y_fit == 1))
    spw = n_healthy / (n_cancer + 1e-6)

    model = make_xgb(seed, spw, n_estimators, early_stopping_rounds)
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

    # best_iteration is 0-indexed; +1 = number of trees used
    best_iter = getattr(model, "best_iteration", None)
    n_trees = (best_iter + 1) if best_iter is not None else n_estimators
    return model, n_trees


# =====================================================================
# 2. METRICS
# =====================================================================

def _sens_spec(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sens, spec


def compute_metrics(y_true, prob, thresh: float = 0.5):
    """AUC (threshold-free) + accuracy/sens/spec at a fixed threshold."""
    y_pred = (prob >= thresh).astype(int)
    auc = roc_auc_score(y_true, prob)
    acc = accuracy_score(y_true, y_pred)
    sens, spec = _sens_spec(y_true, y_pred)
    return {"auc": auc, "acc": acc, "sens": sens, "spec": spec}


# =====================================================================
# 3. BOOTSTRAP CONFIDENCE INTERVALS (test-set resampling)
# =====================================================================

def bootstrap_ci(y_true: np.ndarray, prob: np.ndarray,
                 n_boot: int = 2000, alpha: float = 0.05,
                 thresh: float = 0.5, seed: int = 0):
    """Percentile bootstrap CIs for AUC/acc/sens/spec.

    Resamples the TEST rows with replacement. Stratified within each class so
    that every bootstrap replicate keeps both classes present (essential for a
    small 74/101 test set where an unstratified resample can occasionally drop
    a class and make AUC undefined).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)

    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]

    keys = ["auc", "acc", "sens", "spec"]
    acc = {k: [] for k in keys}

    for _ in range(n_boot):
        bpos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        bneg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        bidx = np.concatenate([bpos, bneg])
        yb, pb = y_true[bidx], prob[bidx]
        m = compute_metrics(yb, pb, thresh)
        for k in keys:
            acc[k].append(m[k])

    out = {}
    lo_q, hi_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    for k in keys:
        arr = np.array(acc[k])
        out[k] = {
            "mean": float(arr.mean()),
            "lo": float(np.percentile(arr, lo_q)),
            "hi": float(np.percentile(arr, hi_q)),
        }
    return out


# =====================================================================
# 4. DeLong TEST for two correlated AUCs (same test set)
# =====================================================================
# Implementation of the fast DeLong method (Sun & Xu, 2014) for the covariance
# of AUC estimates, used to test whether two AUCs measured on the SAME test set
# differ significantly.

def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def _calc_pvalue(aucs, sigma):
    import scipy.stats
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / (np.sqrt(np.dot(np.dot(l, sigma), l.T)) + 1e-12)
    p = 2 * (1 - scipy.stats.norm.cdf(z))
    return float(np.squeeze(z)), float(np.squeeze(p))


def delong_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray):
    """Test H0: AUC_a == AUC_b for two models on the SAME labels.

    Returns (auc_a, auc_b, z, p_value). Two-sided.
    """
    y_true = np.asarray(y_true)
    order = (-y_true).argsort(kind="stable")  # positives first
    label_1_count = int(y_true.sum())
    preds = np.vstack((prob_a, prob_b))[:, order]
    aucs, cov = _fast_delong(preds, label_1_count)
    z, p = _calc_pvalue(aucs, cov)
    return float(aucs[0]), float(aucs[1]), z, p


# =====================================================================
# 5. MULTI-SEED EVALUATION HARNESS
# =====================================================================

@dataclass
class StrategyResult:
    name: str
    n_train_h: int
    n_train_c: int
    features: int
    per_seed: dict = field(default_factory=dict)   # metric -> list over seeds
    n_trees: list = field(default_factory=list)
    # test-set predictions from the FIRST seed, kept for DeLong comparisons
    ref_prob: Optional[np.ndarray] = None
    ref_seed: Optional[int] = None

    def summary(self):
        out = {"strategy": self.name,
               "n_train_h": self.n_train_h,
               "n_train_c": self.n_train_c,
               "features": self.features}
        for metric, vals in self.per_seed.items():
            arr = np.array(vals)
            out[f"{metric}_mean"] = float(arr.mean())
            out[f"{metric}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        out["n_trees_median"] = float(np.median(self.n_trees)) if self.n_trees else 0.0
        return out


def evaluate_strategy(build_train_fn, X_te, y_te, name,
                      n_train_h, n_train_c,
                      seeds=(42, 1, 2, 3, 4),
                      val_ratio: float = 0.15,
                      thresh: float = 0.5):
    """Run one balancing/augmentation strategy across multiple seeds.

    build_train_fn(seed) -> (X_tr, y_tr)
        A callable that returns the training set for a given seed. It is a
        callable (not a fixed array) so that any STOCHASTIC data construction
        - synthetic sample generation, undersampling, SMOTE - is regenerated
        per seed. Deterministic strategies can simply ignore the seed argument
        and return the same arrays each call.

    Returns a StrategyResult with per-seed metrics and the first seed's test
    probabilities retained for DeLong tests.
    """
    res = StrategyResult(name=name, n_train_h=n_train_h, n_train_c=n_train_c,
                         features=X_te.shape[1])
    res.per_seed = {"auc": [], "acc": [], "sens": [], "spec": []}

    for si, seed in enumerate(seeds):
        X_tr, y_tr = build_train_fn(seed)
        model, n_trees = train_xgb_no_leak(
            X_tr, y_tr, seed=seed, val_ratio=val_ratio)
        prob_te = model.predict_proba(X_te)[:, 1]
        m = compute_metrics(y_te, prob_te, thresh)
        for k in res.per_seed:
            res.per_seed[k].append(m[k])
        res.n_trees.append(n_trees)
        if si == 0:
            res.ref_prob = prob_te
            res.ref_seed = seed

    return res
