"""
Ekstraksi fitur NLP (tanpa pre-trained embedding / transformer).

  1. Lexicon per artikel : VADER (compound, pos, neg) + Loughran-McDonald
     (negative, positive, uncertainty, litigious; dinormalisasi jumlah token).
  2. Agregasi harian     : rata-rata skor dan proporsi artikel negatif/positif per hari.
  3. TF-IDF              : 1 dokumen per hari target -> TruncatedSVD (50 dimensi).
     Vocabulary, IDF, dan SVD di-FIT HANYA pada hari train (anti-leakage).
Teks yang dipakai untuk model: judul ("t"). Varian judul+keypoints ("tk") hanya dihitung
lexicon-nya untuk perbandingan di notebook EDA.
Baris keluaran diindeks oleh `date` = hari kerja target T.
"""
import os
import re

import pandas as pd
import pysentiment2
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import (ARTICLE_FEATURES, ARTICLES_PREPROCESSED, DAILY_BASE,
                    FEATURES_DAILY, SEED)

VARIANTS = {"t": "title", "tk": "title_kp"}      # kode singkat -> nama varian teks
LM_CATS = ["Negative", "Positive", "Uncertainty", "Litigious"]
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
    """Gabungkan skor artikel menjadi satu baris per hari target (rata-rata / proporsi,
    supaya hari dengan banyak artikel tidak mendistorsi fitur)."""
    df = pd.concat([arts[["target_date"]], feats], axis=1)
    g = df.groupby("target_date")
    daily = pd.DataFrame({"n_articles": g.size()})
    for code in VARIANTS:
        comp = f"{code}_vader_compound"
        daily[f"{code}_compound_mean"] = g[comp].mean()
        daily[f"{code}_share_neg"] = g[comp].apply(lambda s: (s < -0.05).mean())
        daily[f"{code}_share_pos"] = g[comp].apply(lambda s: (s > 0.05).mean())
        for c in LM_CATS:
            daily[f"{code}_lm_{c.lower()}_mean"] = g[f"{code}_lm_{c.lower()}"].mean()
    daily.index.name = "date"
    return daily


# ------------------------------------------------------------ vektor -> SVD
def daily_documents(arts: pd.DataFrame, variant: str) -> pd.Series:
    """Satu dokumen per hari target: gabungan teks bersih semua artikel hari itu."""
    return arts.groupby("target_date")[f"clean_{variant}"].apply(lambda s: " ".join(s.fillna("")))


def vectorize_svd(docs: pd.Series, is_train: pd.Series, n_components=50, max_features=5000,
                  prefix="tfidf"):
    """TF-IDF (unigram + bigram) lalu TruncatedSVD. Di-fit di train saja, lalu transform semua hari."""
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.8,
                          max_features=max_features, sublinear_tf=True)
    Xtr = vec.fit_transform(docs[is_train])
    svd = TruncatedSVD(n_components=n_components, random_state=SEED).fit(Xtr)
    Z = svd.transform(vec.transform(docs))
    cols = [f"{prefix}_svd{i:02d}" for i in range(n_components)]
    return pd.DataFrame(Z, index=docs.index, columns=cols), vec


def main():
    arts = pd.read_csv(ARTICLES_PREPROCESSED, encoding="utf-8-sig", low_memory=False)
    arts["target_date"] = pd.to_datetime(arts["target_date"])
    base = pd.read_csv(DAILY_BASE, parse_dates=["date"]).set_index("date")

    print("Lexicon per artikel ...")
    feats = article_lexicon_features(arts)
    pd.concat([arts[["url", "target_date"]], feats], axis=1).to_csv(ARTICLE_FEATURES, index=False)

    print("Agregasi harian ...")
    daily = aggregate_daily(arts, feats)

    print("TF-IDF + SVD (fit hanya di train) ...")
    is_train_day = base["split"].reindex(daily.index).eq("train")
    docs = daily_documents(arts, "title").reindex(daily.index).fillna("")
    Z, vec = vectorize_svd(docs, is_train_day, prefix="tfidf_t")
    print(f"  vocabulary TF-IDF: {len(vec.vocabulary_)} kata/bigram")

    out = pd.concat([daily, Z], axis=1).reindex(base.index)
    out.to_csv(FEATURES_DAILY)
    print(f"features_daily: {out.shape}, NaN total={int(out.isna().sum().sum())}")


if __name__ == "__main__":
    main()
