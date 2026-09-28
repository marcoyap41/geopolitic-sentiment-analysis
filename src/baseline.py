"""
Langkah 3a - Baseline (hanya data kurs, tanpa NLP).

  * majority    : selalu menebak kelas mayoritas train
  * persistence : arah hari T = arah hari T-1
  * ARIMA       : ARIMA(p,0,q) pada return persen; order dipilih via AIC di train;
                  prediksi one-step-ahead (parameter tetap dari train) -> arah = tanda forecast
  * LR/XGB kurs : model ML dengan fitur lag kurs saja (lihat combined_model.py)
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


def majority_baseline(df):
    """Prediksi kelas mayoritas train untuk semua hari."""
    maj = int(df.loc[df["split"] == "train", "y"].mean() >= 0.5)
    return pd.Series(maj, index=df.index)


def persistence_baseline(df):
    """Tebak arah sama dengan hari kerja sebelumnya (hari pertama diisi mayoritas)."""
    return df["y"].shift(1).fillna(majority_baseline(df)).astype(int)


def select_arima_order(train_ret, max_p=2, max_q=2):
    """Grid kecil (p,q) dengan AIC terendah pada train."""
    best, best_aic = (0, 0), np.inf
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                aic = ARIMA(train_ret, order=(p, 0, q)).fit().aic
            except Exception:
                continue
            if aic < best_aic:
                best, best_aic = (p, q), aic
    return best


def arima_baseline(df):
    """Prediksi arah dari forecast one-step ARIMA (parameter dari train, tanpa refit)."""
    ret = df["ret_pct"].values
    is_tr = (df["split"] == "train").values
    p, q = select_arima_order(ret[is_tr])
    res = ARIMA(ret[is_tr], order=(p, 0, q)).fit()
    full = res.apply(ret, refit=False)          # filter Kalman: prediksi t hanya pakai data < t
    fc = full.get_prediction(start=0).predicted_mean
    pred = (fc > 0).astype(int)
    return pd.Series(pred, index=df.index), (p, q)
