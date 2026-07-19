#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
threshold_analysis.py

Operating-point (decision-threshold) analysis for the Spectral-LDM experiments.

WHY THIS EXISTS
---------------
All metrics in eval_utils are reported at a fixed 0.5 probability cutoff. But
0.5 is arbitrary: it is not where any of these models is best calibrated, and a
model with the highest AUC (best ranking) can still look bad on sens/spec at
0.5 if its probabilities are shifted. Because LDM has the highest AUC in the
leak-free evaluation, the clinically relevant question is:

    "At a properly chosen operating point, what sensitivity/specificity
     trade-off does each strategy achieve?"

THE ONE RULE THAT MAKES THIS HONEST
-----------------------------------
The threshold is a hyperparameter. If you pick it by looking at the test set,
you have re-introduced exactly the leak we spent the last session removing -
just a subtler version. So every threshold here is selected ONLY on an internal
validation split carved from the training data. The test set is touched once,
at the very end, to report the final number at the already-frozen threshold.

WHAT IT REPORTS
---------------
For each strategy, across multiple seeds:
  - the validation-selected threshold (per seed),
  - test-set sensitivity/specificity/accuracy at that threshold,
  - AUC (threshold-free, unchanged from before) for reference,
  - mean +/- std over seeds, plus a bootstrap CI at the frozen threshold.

Two selection rules are provided:
  1. "youden"        : maximize Youden's J = sensitivity + specificity - 1.
  2. "target_sens"   : pick the highest threshold whose validation sensitivity
                       is >= a clinical target (default 0.80). This mirrors how
                       a screening tool is actually tuned - fix a floor on
                       sensitivity (don't miss cancers), then maximize
                       specificity subject to that floor.

Both are defensible; report whichever matches the clinical argument, but decide
BEFORE seeing the test numbers and report both if asked.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

from eval_utils import make_xgb   # reuse the leak-free XGB constructor


# ---------------------------------------------------------------------------
# Threshold selection rules (operate on VALIDATION predictions only)
# ---------------------------------------------------------------------------

def threshold_youden(y_val, p_val):
    """Threshold maximizing Youden's J on the validation set."""
    fpr, tpr, thr = roc_curve(y_val, p_val)
    j = tpr - fpr
    k = int(np.argmax(j))
    # roc_curve prepends an inf threshold; guard against it
    t = thr[k]
    if not np.isfinite(t):
        t = 0.5
    return float(t)


def threshold_target_sens(y_val, p_val, target_sens=0.80):
    """Highest threshold whose validation sensitivity >= target.

    Higher threshold -> higher specificity, so among all thresholds meeting the
    sensitivity floor we take the most specific one. Falls back to the
    sensitivity-maximizing threshold if the target is unreachable.
    """
    fpr, tpr, thr = roc_curve(y_val, p_val)
    ok = np.where(tpr >= target_sens)[0]
    if len(ok) == 0:
        # target unreachable on val; take the most sensitive finite threshold
        finite = thr[np.isfinite(thr)]
        return float(finite.min()) if len(finite) else 0.5
    # among thresholds meeting the floor, the largest threshold = most specific
    cand = thr[ok]
    cand = cand[np.isfinite(cand)]
    if len(cand) == 0:
        return 0.5
    return float(cand.max())


SELECTORS = {
    "youden": threshold_youden,
    "target_sens": threshold_target_sens,
}


# ---------------------------------------------------------------------------
# Metrics at an arbitrary threshold
# ---------------------------------------------------------------------------

def metrics_at_threshold(y_true, prob, thr):
    y_pred = (prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    auc = roc_auc_score(y_true, prob)
    return {"auc": auc, "acc": acc, "sens": sens, "spec": spec, "thr": thr}


# ---------------------------------------------------------------------------
# Leak-free train + validation-selected threshold, evaluated on test
# ---------------------------------------------------------------------------

def train_and_pick_threshold(X_tr, y_tr, X_te, y_te,
                             seed=42, val_ratio=0.15,
                             selector="youden", target_sens=0.80,
                             n_estimators=400, early_stopping_rounds=20):
    """Train XGB with early stopping on an internal val split, choose the
    decision threshold on that SAME val split, then score the test set once.

    Returns (test_metrics_dict, chosen_threshold, test_probabilities).
    """
    # internal validation split from TRAIN only - used for BOTH early stopping
    # AND threshold selection, so the test set is never consulted for either.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=val_ratio, random_state=seed, stratify=y_tr)

    spw = np.sum(y_fit == 0) / (np.sum(y_fit == 1) + 1e-6)
    model = make_xgb(seed, spw, n_estimators, early_stopping_rounds)
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

    # threshold chosen on validation predictions
    p_val = model.predict_proba(X_val)[:, 1]
    if selector == "target_sens":
        thr = threshold_target_sens(y_val, p_val, target_sens=target_sens)
    else:
        thr = threshold_youden(y_val, p_val)

    # single, final look at the test set at the frozen threshold
    p_te = model.predict_proba(X_te)[:, 1]
    m = metrics_at_threshold(y_te, p_te, thr)
    return m, thr, p_te


# ---------------------------------------------------------------------------
# Multi-seed harness with bootstrap CI at the frozen (per-seed) threshold
# ---------------------------------------------------------------------------

@dataclass
class ThreshResult:
    name: str
    selector: str
    per_seed: dict = field(default_factory=dict)   # metric -> list over seeds
    thresholds: list = field(default_factory=list)
    ref_prob: Optional[np.ndarray] = None
    ref_thr: Optional[float] = None

    def summary(self):
        out = {"strategy": self.name, "selector": self.selector}
        for metric, vals in self.per_seed.items():
            a = np.array(vals)
            out[f"{metric}_mean"] = float(a.mean())
            out[f"{metric}_std"] = float(a.std(ddof=1)) if len(a) > 1 else 0.0
        out["thr_mean"] = float(np.mean(self.thresholds)) if self.thresholds else 0.5
        out["thr_std"] = float(np.std(self.thresholds, ddof=1)) if len(self.thresholds) > 1 else 0.0
        return out


def evaluate_strategy_thresholded(build_train_fn: Callable, X_te, y_te, name,
                                  seeds=(42, 1, 2, 3, 4),
                                  selector="youden", target_sens=0.80,
                                  val_ratio=0.15):
    """Run one strategy across seeds, selecting the threshold on validation
    each seed. build_train_fn(seed) -> (X_tr, y_tr), as in eval_utils.
    """
    res = ThreshResult(name=name, selector=selector)
    res.per_seed = {"auc": [], "acc": [], "sens": [], "spec": []}

    for si, seed in enumerate(seeds):
        X_tr, y_tr = build_train_fn(seed)
        m, thr, p_te = train_and_pick_threshold(
            X_tr, y_tr, X_te, y_te, seed=seed, val_ratio=val_ratio,
            selector=selector, target_sens=target_sens)
        for k in res.per_seed:
            res.per_seed[k].append(m[k])
        res.thresholds.append(thr)
        if si == 0:
            res.ref_prob = p_te
            res.ref_thr = thr

    return res


def bootstrap_ci_at_threshold(y_true, prob, thr, n_boot=2000, alpha=0.05, seed=0):
    """Stratified percentile bootstrap CI for sens/spec/acc at a FIXED
    threshold, plus threshold-free AUC. Threshold is frozen (chosen on val),
    so this only quantifies test-set sampling variability."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); prob = np.asarray(prob)
    ip = np.where(y_true == 1)[0]; ineg = np.where(y_true == 0)[0]
    keys = ["auc", "acc", "sens", "spec"]
    acc = {k: [] for k in keys}
    for _ in range(n_boot):
        bp = rng.choice(ip, size=len(ip), replace=True)
        bn = rng.choice(ineg, size=len(ineg), replace=True)
        bi = np.concatenate([bp, bn])
        m = metrics_at_threshold(y_true[bi], prob[bi], thr)
        for k in keys:
            acc[k].append(m[k])
    out = {}
    lo, hi = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    for k in keys:
        a = np.array(acc[k])
        out[k] = {"mean": float(a.mean()),
                  "lo": float(np.percentile(a, lo)),
                  "hi": float(np.percentile(a, hi))}
    return out
