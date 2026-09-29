"""Menyiapkan tabel data harian: target, fitur kurs (lag), dan fitur NLP harian."""
import pandas as pd

from config import DAILY_BASE, FEATURES_DAILY


def kurs_features(base: pd.DataFrame) -> pd.DataFrame:
    """Fitur dari riwayat kurs untuk hari T. Semua memakai data sampai T-1 (shift),
    jadi tidak ada kebocoran informasi dari hari T."""
    r = base["ret_pct"]                       # perubahan kurs (%) per hari
    f = pd.DataFrame(index=base.index)
    for k in range(1, 6):
        f[f"k_ret_lag{k}"] = r.shift(k)       # return 1..5 hari sebelumnya
    f["k_mean5"] = r.shift(1).rolling(5, min_periods=3).mean()   # rata-rata 5 hari terakhir
    f["k_std5"] = r.shift(1).rolling(5, min_periods=3).std()     # volatilitas 5 hari terakhir
    return f


def load_dataset():
    """Return (df, feature_sets).
    df: satu baris per hari target, berisi date, y, split, dan semua fitur.
    feature_sets: dict nama set fitur -> daftar kolom."""
    base = pd.read_csv(DAILY_BASE, parse_dates=["date"])
    nlp = pd.read_csv(FEATURES_DAILY, parse_dates=["date"]).set_index("date")
    df = pd.concat([base.set_index("date"), kurs_features(base).set_axis(base["date"])], axis=1)
    df = df.join(nlp, how="left").reset_index()

    feature_sets = {
        "kurs": [c for c in df.columns if c.startswith("k_")],            # fitur kurs saja
        "lexicon": [c for c in df.columns if c.startswith("t_")],         # VADER + Loughran-McDonald (judul)
        "tfidf": [c for c in df.columns if c.startswith("tfidf_t_")],     # TF-IDF (judul) -> SVD
        "category": [c for c in df.columns if c.startswith("cat_")],      # proporsi kategori kata kunci
    }
    return df, feature_sets


def split_xy(df, cols, split):
    """Ambil X dan y untuk satu split ('train'/'val'/'test')."""
    m = df["split"] == split
    return df.loc[m, cols], df.loc[m, "y"].astype(int)
