"""
Langkah 1 - Preprocessing teks artikel.

Dua versi teks per artikel:
  * text_raw   : judul (+ keypoints jika ada) dengan pembersihan ringan
                 (untuk lexicon: huruf besar & tanda baca dipertahankan).
  * text_clean : lowercase, tanpa URL/angka/simbol, tanpa stopword, lemmatized
                 (untuk TF-IDF / Bag-of-Words).
Varian teks:
  * title      : hanya judul
  * title_kp   : judul + keypoints (jika keypoints kosong -> hanya judul)
"""
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from config import ARTICLES_PREPROCESSED, ARTICLES_TARGETED

for _p in ("stopwords", "wordnet", "omw-1.4"):
    nltk.download(_p, quiet=True)

_LEMMA = WordNetLemmatizer()
# Kata pengecualian stopword: negasi/kuantor penting untuk makna sentimen
_KEEP = {"no", "not", "nor", "against", "up", "down", "over", "under"}
STOP = set(stopwords.words("english")) - _KEEP
_URL = re.compile(r"https?://\S+|www\.\S+")
_NONALPHA = re.compile(r"[^a-z\s]")
_WS = re.compile(r"\s+")


def light_clean(text: str) -> str:
    """Pembersihan ringan: hapus URL, pemisah keypoints, spasi ganda."""
    if not isinstance(text, str):
        return ""
    text = _URL.sub(" ", text).replace(" | ", ". ")
    return _WS.sub(" ", text).strip()


def heavy_clean(text: str) -> str:
    """Lowercase, buang non-huruf, stopword, token pendek; lemmatize."""
    text = _NONALPHA.sub(" ", light_clean(text).lower())
    toks = [_LEMMA.lemmatize(t) for t in text.split() if len(t) > 2 and t not in STOP]
    return " ".join(toks)


def build_text_variants(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    title = df["title"].fillna("").map(light_clean)
    kp = df["keypoints"].fillna("").map(light_clean)
    df["text_title"] = title
    df["text_title_kp"] = (title + ". " + kp).str.strip(". ").where(kp != "", title)
    df["clean_title"] = df["text_title"].map(heavy_clean)
    df["clean_title_kp"] = df["text_title_kp"].map(heavy_clean)
    df["has_kp"] = (kp != "").astype(int)
    return df


def main():
    df = pd.read_csv(ARTICLES_TARGETED, encoding="utf-8-sig", low_memory=False)
    df = df[df["title"].notna() & (df["title"].str.strip() != "")]
    df = build_text_variants(df)
    keep = ["url", "pub_date", "target_date", "split", "keyword_category", "has_kp",
            "text_title", "text_title_kp", "clean_title", "clean_title_kp"]
    df[keep].to_csv(ARTICLES_PREPROCESSED, index=False, encoding="utf-8-sig")
    print(f"{len(df)} artikel; keypoints tersedia: {df['has_kp'].mean():.1%}")
    print(df[["text_title", "clean_title"]].head(3).to_string())


if __name__ == "__main__":
    main()
