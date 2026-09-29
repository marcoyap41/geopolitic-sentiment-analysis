"""
Langkah 3b - Model ML (LogReg L2 & XGBoost) untuk fitur kurs saja maupun kurs + NLP.

Protokol:
  1. Hyperparameter dipilih dengan TimeSeriesSplit (5 lipatan expanding) DI DALAM train.
  2. Metrik VALIDASI: model dilatih di train saja.
  3. Metrik TEST (dipakai sekali): model di-refit pada train+val dengan hyperparameter terpilih.
"""
import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import SEED
from dataset import split_xy

warnings.filterwarnings("ignore")

GRIDS = {
    "logreg": [{"C": c} for c in (0.001, 0.01, 0.1, 1.0)],
    "xgb": [{"max_depth": d, "n_estimators": n} for d, n in itertools.product((2, 3), (100, 300))],
}


def make_model(kind, **hp):
    if kind == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(penalty="l2", max_iter=2000, random_state=SEED, **hp))
    return XGBClassifier(learning_rate=0.05, subsample=0.8, colsample_bytree=0.5,
                         reg_lambda=5.0, min_child_weight=5, random_state=SEED,
                         eval_metric="logloss", n_jobs=2, **hp)


def tune(kind, X, y, n_splits=5):
    """Pilih hyperparameter dengan CV time-series di train (akurasi rata-rata)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best, best_score = None, -1
    for hp in GRIDS[kind]:
        scores = []
        for tr, va in tscv.split(X):
            m = make_model(kind, **hp).fit(X.iloc[tr], y.iloc[tr])
            scores.append((m.predict(X.iloc[va]) == y.iloc[va]).mean())
        if np.mean(scores) > best_score:
            best, best_score = hp, float(np.mean(scores))
    return best, best_score


def run_experiment(df, cols, kind):
    """Return dict: hp, cv_score, pred/proba untuk val (train-only) dan test (train+val)."""
    Xtr, ytr = split_xy(df, cols, "train")
    hp, cv = tune(kind, Xtr, ytr)
    Xva, yva = split_xy(df, cols, "val")
    m_val = make_model(kind, **hp).fit(Xtr, ytr)
    Xtv, ytv = split_xy(df, cols, ["train", "val"])
    Xte, yte = split_xy(df, cols, "test")
    m_te = make_model(kind, **hp).fit(Xtv, ytv)
    return {"hp": hp, "cv": cv,
            "val": (yva.values, m_val.predict(Xva), m_val.predict_proba(Xva)[:, 1]),
            "test": (yte.values, m_te.predict(Xte), m_te.predict_proba(Xte)[:, 1]),
            "model": m_te}
