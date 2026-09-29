"""Konfigurasi bersama: path, konstanta split, seed."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLEANED = ROOT / "data" / "cleaned"
PROCESSED = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"

KURS_FILE = CLEANED / "bi_jisdor_2021_2026.csv"
ARTICLES_FILE = CLEANED / "aligned_news_kurs.csv"

DAILY_BASE = PROCESSED / "daily_base.csv"
ARTICLES_TARGETED = PROCESSED / "articles_targeted.csv"
ARTICLES_PREPROCESSED = PROCESSED / "articles_preprocessed.csv"
ARTICLE_FEATURES = PROCESSED / "article_features.csv"
FEATURES_DAILY = PROCESSED / "features_daily.csv"

# Split kronologis (train, val, test) berdasarkan urutan hari kerja
SPLIT_FRACS = (0.6, 0.2, 0.2)
SEED = 42
