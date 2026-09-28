"""Pemuatan dataset model: gabungan target/kurs, fitur kurs (lag), dan fitur NLP harian."""
import numpy as np
import pandas as pd

from config import DAILY_BASE, FEATURES_DAILY


def kurs_features(base: pd.DataFrame) -> pd.DataFrame:
    """Fitur dari riwayat kurs untuk hari T. SEMUA memakai data <= T-1 (shift)."""
    r = base["ret_pct"]
    f = pd.DataFrame(index=base.index)
    for k in range(1, 6):
        f[f"k_ret_lag{k}"] = r.shift(k)                     # return T-k
    f["k_dir_lag1"] = (base["ret"].shift(1) > 0).astype(float)
    f["k_absret_lag1"] = r.shift(1).abs()
    f["k_mean5"] = r.shift(1).rolling(5, min_periods=3).mean()
    f["k_std5"] = r.shift(1).rolling(5, min_periods=3).std()
    f["k_std10"] = r.shift(1).rolling(10, min_periods=5).std()
    f["k_dow"] = base["date"].dt.dayofweek                  # hari T diketahui sebelumnya
    return f


def load_dataset():
    """Return (df, feature_sets). df diindeks urut waktu, kolom: date, y, split, fitur."""
    base = pd.read_csv(DAILY_BASE, parse_dates=["date"])
    nlp = pd.read_csv(FEATURES_DAILY, parse_dates=["date"]).set_index("date")
    df = pd.concat([base.set_index("date"), kurs_features(base).set_axis(base["date"])], axis=1)
    df = df.join(nlp, how="left").reset_index().copy()

    cols = [c for c in df.columns]
    pick = lambda pre: [c for c in cols if c.startswith(pre)]
    fs = {
        "kurs": pick("k_"),
        "lex_t": pick("t_"),
        "lex_tk": pick("tk_"),
        "tfidf_t": pick("tfidf_t_"),
        "tfidf_tk": pick("tfidf_tk_"),
        "bow_t": pick("bow_t_"),
        "volcat": ["n_articles", "n_articles_lag1", "n_articles_r3"] + pick("cat_"),
    }
    return df, fs


def split_xy(df, cols, split):
    """Ambil X, y untuk satu/lebih split (str atau list)."""
    m = df["split"].isin([split] if isinstance(split, str) else split)
    return df.loc[m, cols], df.loc[m, "y"].astype(int)
