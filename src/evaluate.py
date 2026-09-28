"""Metrik evaluasi, uji signifikansi, dan plot."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (confusion_matrix, f1_score, matthews_corrcoef,
                             roc_auc_score)


def directional_accuracy(y, pred):
    """Proporsi hari yang arahnya ditebak benar."""
    return float(np.mean(np.asarray(y) == np.asarray(pred)))


def metrics(y, pred, proba=None) -> dict:
    """Directional accuracy, F1 macro, MCC (+ AUC jika ada probabilitas)."""
    y, pred = np.asarray(y), np.asarray(pred)
    out = {
        "DA": directional_accuracy(y, pred),
        "F1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, pred) if len(set(pred)) > 1 else 0.0,
    }
    if proba is not None and len(set(y)) > 1:
        out["AUC"] = roc_auc_score(y, proba)
    return out


def binom_vs_majority(y, pred, majority_rate) -> float:
    """p-value satu sisi: apakah akurasi lebih tinggi dari majority baseline?"""
    k, n = int((np.asarray(y) == np.asarray(pred)).sum()), len(y)
    return float(stats.binomtest(k, n, majority_rate, alternative="greater").pvalue)


def bootstrap_da_ci(y, pred, n_boot=2000, seed=42):
    """CI 95% bootstrap untuk directional accuracy."""
    rng = np.random.default_rng(seed)
    ok = (np.asarray(y) == np.asarray(pred)).astype(float)
    bs = [rng.choice(ok, len(ok)).mean() for _ in range(n_boot)]
    return tuple(np.percentile(bs, [2.5, 97.5]))


def mcnemar_p(y, pred_a, pred_b) -> float:
    """Uji McNemar (exact) antara dua model pada data yang sama."""
    a = np.asarray(pred_a) == np.asarray(y)
    b = np.asarray(pred_b) == np.asarray(y)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)


def cm_table(y, pred) -> pd.DataFrame:
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return pd.DataFrame(cm, index=["aktual_0(tdk melemah)", "aktual_1(melemah)"],
                        columns=["pred_0", "pred_1"])
