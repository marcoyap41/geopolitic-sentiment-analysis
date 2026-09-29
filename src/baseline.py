"""
Model baseline time-series (hanya memakai data kurs, tanpa NLP).

  * majority    : selalu menebak kelas yang paling sering muncul di train
  * persistence : arah hari T dianggap sama dengan arah hari T-1 (naive baseline)
  * ARIMA(1,0,1): model time-series klasik pada return kurs; arah = tanda hasil prediksi
  * XGBoost kurs-saja: lihat model.py (fitur kurs saja)
"""
import warnings

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
ARIMA_ORDER = (1, 0, 1)


def majority_baseline(df):
    """Prediksi kelas mayoritas TRAIN untuk semua hari."""
    maj = int(df.loc[df["split"] == "train", "y"].mean() >= 0.5)
    return pd.Series(maj, index=df.index)


def persistence_baseline(df):
    """Tebak arah hari ini = arah kemarin (hari pertama diisi kelas mayoritas)."""
    return df["y"].shift(1).fillna(majority_baseline(df)).astype(int)


def arima_baseline(df):
    """ARIMA di-fit hanya pada return train. Untuk tiap hari, prediksi satu langkah ke depan
    hanya memakai data sebelum hari itu. Prediksi return > 0 -> rupiah melemah (kelas 1)."""
    ret = df["ret_pct"].values
    is_train = (df["split"] == "train").values
    fitted = ARIMA(ret[is_train], order=ARIMA_ORDER).fit()
    filtered = fitted.apply(ret, refit=False)          # parameter tetap dari train
    forecast = filtered.get_prediction(start=0).predicted_mean
    return pd.Series((forecast > 0).astype(int), index=df.index)
