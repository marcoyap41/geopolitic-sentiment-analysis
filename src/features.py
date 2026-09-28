"""
Langkah 2 - Ekstraksi fitur NLP (tanpa pre-trained embedding / transformer).

  1. Lexicon per artikel : VADER (compound, pos, neg) + Loughran-McDonald
     (negative, positive, uncertainty, litigious; dinormalisasi jumlah token).
  2. Agregasi harian     : mean/std/min/max, share_neg, jumlah artikel,
     komposisi kategori keyword, plus lag-1 dan rolling-3 (tanpa melihat masa depan).
  3. Vektor TF-IDF / BoW : 1 dokumen per hari target -> SVD.
     Vocabulary, IDF, dan SVD di-FIT HANYA pada hari train (anti-leakage).
Baris keluaran diindeks oleh `date` = hari kerja target T.
"""
import os
import re

import numpy as np
import pandas as pd
import pysentiment2
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import (ARTICLE_FEATURES, ARTICLES_PREPROCESSED, DAILY_BASE,
                    FEATURES_DAILY, SEED)

VARIANTS = {"t": "title", "tk": "title_kp"}      # kode singkat -> nama varian teks
LM_CATS = ["Negative", "Positive", "Uncertainty", "Litigious"]
CATEGORIES = ["macro", "us_politics", "central_bank", "diplomacy",
              "energy", "conflict", "currency", "dedollar"]
_TOKEN = re.compile(r"[a-z]+")


# ---------------------------------------------------------------- lexicon
def load_lm_dictionary() -> dict:
    """Muat kamus Loughran-McDonald (dari paket pysentiment2) -> {kategori: set kata}."""
    path = os.path.join(os.path.dirname(pysentiment2.__file__), "static", "LM.csv")
    lm = pd.read_csv(path)
    return {c: set(lm.loc[lm[c] > 0, "Word"].str.lower()) for c in LM_CATS}


def article_lexicon_features(df: pd.DataFrame) -> pd.DataFrame:
    """Skor VADER & LM per artikel untuk tiap varian teks."""
    vader, lm = SentimentIntensityAnalyzer(), load_lm_dictionary()
    out = pd.DataFrame(index=df.index)
    for code, name in VARIANTS.items():
        texts = df[f"text_{name}"].fillna("")
        v = texts.map(vader.polarity_scores)
        for k in ("compound", "pos", "neg"):
            out[f"{code}_vader_{k}"] = v.map(lambda d, k=k: d[k])
        toks = texts.str.lower().map(_TOKEN.findall)
        n = toks.map(len).clip(lower=1)
        for c in LM_CATS:
            words = lm[c]
            out[f"{code}_lm_{c.lower()}"] = toks.map(lambda t: sum(w in words for w in t)) / n
    return out


# ------------------------------------------------------------- agregasi harian
def aggregate_daily(arts: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Agregasi fitur artikel ke level hari target."""
    df = pd.concat([arts[["target_date", "keyword_category"]], feats], axis=1)
    g = df.groupby("target_date")
    daily = pd.DataFrame({"n_articles": g.size()})
    for code in VARIANTS:
        comp = f"{code}_vader_compound"
        daily[f"{code}_compound_mean"] = g[comp].mean()
        daily[f"{code}_compound_std"] = g[comp].std().fillna(0)
        daily[f"{code}_compound_min"] = g[comp].min()
        daily[f"{code}_compound_max"] = g[comp].max()
        daily[f"{code}_share_neg"] = g[comp].apply(lambda s: (s < -0.05).mean())
        daily[f"{code}_share_pos"] = g[comp].apply(lambda s: (s > 0.05).mean())
        daily[f"{code}_vader_pos_mean"] = g[f"{code}_vader_pos"].mean()
        daily[f"{code}_vader_neg_mean"] = g[f"{code}_vader_neg"].mean()
        for c in LM_CATS:
            daily[f"{code}_lm_{c.lower()}_mean"] = g[f"{code}_lm_{c.lower()}"].mean()
    cats = df["keyword_category"].fillna("").str.split("|").explode()
    cnt = (pd.crosstab(cats.index.map(df["target_date"]), cats)
           .reindex(columns=CATEGORIES, fill_value=0))
    cnt.index.name = "target_date"
    for c in CATEGORIES:
        daily[f"cat_{c}"] = cnt[c] / daily["n_articles"]
    daily.index.name = "date"
    return daily


def add_lag_features(daily: pd.DataFrame, cols) -> pd.DataFrame:
    """Lag-1 (bucket hari kerja sebelumnya) dan rolling-3 (T, T-1, T-2)."""
    daily = daily.sort_index()
    new = {}
    for c in cols:
        new[f"{c}_lag1"] = daily[c].shift(1)
        new[f"{c}_r3"] = daily[c].rolling(3, min_periods=1).mean()
    return pd.concat([daily, pd.DataFrame(new)], axis=1)


# ------------------------------------------------------------ vektor -> SVD
def daily_documents(arts: pd.DataFrame, variant: str) -> pd.Series:
    """Satu dokumen per hari target: gabungan teks bersih semua artikel hari itu."""
    return arts.groupby("target_date")[f"clean_{variant}"].apply(lambda s: " ".join(s.fillna("")))


def vectorize_svd(docs: pd.Series, is_train: pd.Series, kind="tfidf",
                  n_components=50, max_features=5000, prefix="tfidf"):
    """Fit vectorizer + SVD di train saja, transform semua hari."""
    common = dict(ngram_range=(1, 2), min_df=5, max_df=0.8, max_features=max_features)
    vec = TfidfVectorizer(sublinear_tf=True, **common) if kind == "tfidf" else CountVectorizer(**common)
    Xtr = vec.fit_transform(docs[is_train])
    svd = TruncatedSVD(n_components=n_components, random_state=SEED).fit(Xtr)
    Z = svd.transform(vec.transform(docs))
    cols = [f"{prefix}_svd{i:02d}" for i in range(n_components)]
    return pd.DataFrame(Z, index=docs.index, columns=cols), vec, svd


def main():
    arts = pd.read_csv(ARTICLES_PREPROCESSED, encoding="utf-8-sig", low_memory=False)
    arts["target_date"] = pd.to_datetime(arts["target_date"])
    base = pd.read_csv(DAILY_BASE, parse_dates=["date"]).set_index("date")

    print("Lexicon per artikel ...")
    feats = article_lexicon_features(arts)
    pd.concat([arts[["url", "target_date"]], feats], axis=1).to_csv(ARTICLE_FEATURES, index=False)

    print("Agregasi harian ...")
    daily = aggregate_daily(arts, feats)
    lag_cols = [f"{c}_{m}" for c in VARIANTS for m in
                ("compound_mean", "share_neg", "lm_negative_mean", "lm_uncertainty_mean")]
    daily = add_lag_features(daily, lag_cols + ["n_articles"])

    print("TF-IDF / BoW + SVD (fit hanya di train) ...")
    is_train_day = base["split"].reindex(daily.index).eq("train")
    parts = [daily]
    for code, name in VARIANTS.items():
        docs = daily_documents(arts, name).reindex(daily.index).fillna("")
        for kind in ("tfidf", "bow"):
            Z, vec, _ = vectorize_svd(docs, is_train_day, kind=kind, prefix=f"{kind}_{code}")
            parts.append(Z)
            print(f"  {kind}/{name}: vocab={len(vec.vocabulary_)}")
    out = pd.concat(parts, axis=1).reindex(base.index)
    out.to_csv(FEATURES_DAILY)
    print(f"features_daily: {out.shape}, NaN total={int(out.isna().sum().sum())}")


if __name__ == "__main__":
    main()
