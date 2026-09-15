"""
=====================================================================
preprocessing_pipeline.py  -  Pipeline Pembersihan Data Berita
=====================================================================
Membaca semua file CSV di data/raw/, menggabungkan, membersihkan teks,
dan memfilter artikel yang relevan dengan geopolitik/USD.

Pipeline:
  1. Baca semua raw CSV (CNBC sitemap, RSS, GDELT)
  2. Deduplikasi berdasarkan URL
  3. Hapus URL bukan artikel (halaman kategori, index)
  4. Bersihkan teks judul (HTML entity, whitespace, suffix sumber)
  5. Filter keyword geopolitik (101 keyword, regex vectorized)
  6. Tambah kolom bantu: year, month, source_type
  7. Simpan ke data/cleaned/news_geopolitik_clean.csv

Cara pakai:
    python preprocessing_pipeline.py
    python preprocessing_pipeline.py --no-filter   (simpan semua tanpa filter)
"""

import pandas as pd
import re
import logging
import argparse
import html
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT_DIR   = Path(__file__).parent.parent
RAW_DIR    = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "data" / "cleaned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "news_geopolitik_clean.csv"

# ─── 101 Keyword Geopolitik & USD ────────────────────────────────────────────
GEOPOLITICAL_KEYWORDS = [
    # USD dan mata uang
    "dollar", "usd", "currency", "exchange rate", "forex", "fx rate",
    "devaluation", "revaluation", "depreciation", "appreciation",
    "dxy", "dollar index", "petrodollar", "stablecoin", "rupiah",
    # Bank sentral
    "fed", "federal reserve", "fomc", "rate hike", "rate cut",
    "interest rate", "monetary", "quantitative easing", "qe", "taper",
    "ecb", "boj", "bank of japan", "pboc", "central bank",
    # Geopolitik
    "geopolitic", "sanction", "trade war", "tariff", "war", "conflict",
    "ukraine", "russia", "china", "iran", "north korea", "taiwan",
    "israel", "gaza", "hamas", "hezbollah", "nato", "pentagon",
    "saudi", "opec", "brics", "g7", "g20", "coup", "nuclear",
    # Ekonomi makro
    "inflation", "deflation", "recession", "stagflation", "gdp",
    "economic", "economy", "imf", "world bank", "debt", "deficit",
    "surplus", "trade balance", "current account", "treasury",
    "bond yield", "yield curve", "credit rating", "default",
    # Energi dan komoditas
    "oil", "crude", "gas price", "energy crisis", "lng",
    "gold", "commodity", "supply chain",
    # Pasar keuangan
    "market crash", "bear market", "bull market", "stock market",
    "wall street", "banking crisis", "bank collapse", "bank run",
    "silicon valley bank", "credit suisse",
    # Perdagangan
    "export tariff", "trade deficit", "trade deal",
    "wto", "free trade", "protectionism", "decoupling",
]
GEOPOLITICAL_KEYWORDS = list(dict.fromkeys(GEOPOLITICAL_KEYWORDS))  # hapus duplikat
KEYWORD_PATTERN = re.compile(
    "|".join(re.escape(kw) for kw in GEOPOLITICAL_KEYWORDS),
    re.IGNORECASE
)

# Pattern URL yang BUKAN artikel (halaman kategori/index)
NON_ARTICLE_PATTERNS = re.compile(
    r"cnbc\.com/(us-top-news|world-top-news|europe-news|asia-news|"
    r"transportation|us-news|latest|cnbc-latest|market-insider|"
    r"financial-advisors|personal-finance|investing|markets|"
    r"business|technology|politics|entertainment|health|science|"
    r"select|pro|video|live-tv|digital-original|closing-bell|"
    r"mad-money|fast-money)/?$",
    re.IGNORECASE
)

# ─── Helper: Bersihkan teks judul ────────────────────────────────────────────

def clean_title(title: str) -> str:
    """
    Bersihkan teks judul artikel:
    - Decode HTML entities (&amp; &nbsp; &#39; dll)
    - Hapus suffix sumber berita
    - Normalisasi whitespace
    - Strip karakter non-printable
    """
    if not isinstance(title, str) or not title.strip():
        return ""

    # Decode HTML entities
    title = html.unescape(title)

    # Hapus suffix " - CNBC", "| Reuters", "| AP News" dll
    title = re.sub(r"\s*[-|]\s*(CNBC|Reuters|AP News|Bloomberg|FT|WSJ|BBC).*$",
                   "", title, flags=re.IGNORECASE)

    # Hapus karakter non-printable dan kontrol
    title = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", title)

    # Hapus tag HTML yang tersisa (jika ada)
    title = re.sub(r"<[^>]+>", "", title)

    # Normalisasi whitespace
    title = re.sub(r"\s+", " ", title).strip()

    return title


def is_article_url(url: str) -> bool:
    """Return True jika URL adalah halaman artikel, bukan halaman kategori."""
    if not isinstance(url, str):
        return False
    # URL artikel CNBC punya pola: /YYYY/MM/DD/slug.html
    if re.search(r"/\d{4}/\d{2}/\d{2}/", url):
        return True
    # URL non-CNBC (RSS dari Reuters/AP) biasanya juga artikel
    if "cnbc.com" not in url:
        return True
    # Cek apakah cocok dengan pola non-artikel
    if NON_ARTICLE_PATTERNS.search(url):
        return False
    return True


# ─── Step 1: Baca semua raw CSV ──────────────────────────────────────────────

def load_all_raw(raw_dir: Path) -> pd.DataFrame:
    """Baca semua CSV di folder raw dan gabungkan."""
    csvs = list(raw_dir.glob("*.csv"))
    if not csvs:
        log.error("Tidak ada file CSV di: %s", raw_dir)
        return pd.DataFrame()

    frames = []
    for csv_path in csvs:
        log.info("  Membaca: %s (%s KB)", csv_path.name, csv_path.stat().st_size // 1024)
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
            df["_source_file"] = csv_path.name
            frames.append(df)
        except Exception as e:
            log.warning("  Gagal baca %s: %s", csv_path.name, e)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    log.info("Total baris sebelum deduplikasi: %d", len(combined))
    return combined


# ─── Step 2-6: Pipeline utama ────────────────────────────────────────────────

def run_pipeline(df: pd.DataFrame, apply_filter: bool = True) -> pd.DataFrame:
    """
    Jalankan seluruh pipeline preprocessing.
    """
    log.info("\n=== PIPELINE PREPROCESSING ===")

    # Standardisasi nama kolom: semua sumber punya 'url', 'title', 'date'/'published'/'seendate'
    # Deteksi kolom tanggal
    date_col = None
    for candidate in ["date", "published", "seendate", "tanggal"]:
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col and date_col != "date":
        df = df.rename(columns={date_col: "date"})

    # Standardisasi kolom judul
    title_col = None
    for candidate in ["title", "Title", "headline"]:
        if candidate in df.columns:
            title_col = candidate
            break
    if title_col and title_col != "title":
        df = df.rename(columns={title_col: "title"})

    # Pastikan kolom wajib ada
    for col in ["url", "title", "date"]:
        if col not in df.columns:
            df[col] = ""

    # Step 2: Deduplikasi berdasarkan URL
    before = len(df)
    df = df.drop_duplicates(subset=["url"])
    log.info("Step 2 - Deduplikasi URL: %d -> %d (-%d duplikat)",
             before, len(df), before - len(df))

    # Step 3: Filter URL bukan artikel
    before = len(df)
    df = df[df["url"].apply(is_article_url)].copy()
    log.info("Step 3 - Filter non-artikel: %d -> %d (-%d halaman kategori)",
             before, len(df), before - len(df))

    # Step 4: Bersihkan teks judul
    df["title"] = df["title"].apply(clean_title)
    empty_titles = (df["title"] == "").sum()
    log.info("Step 4 - Bersihkan judul: %d judul kosong ditemukan", empty_titles)

    # Hapus baris dengan judul kosong
    df = df[df["title"] != ""].copy()

    # Step 5: Filter keyword geopolitik
    if apply_filter:
        before = len(df)
        mask = df["title"].str.contains(KEYWORD_PATTERN, na=False)
        df = df[mask].copy()
        log.info("Step 5 - Filter geopolitik: %d -> %d (-%d tidak relevan)",
                 before, len(df), before - len(df))
    else:
        log.info("Step 5 - Filter geopolitik: DILEWATI (--no-filter)")

    # Step 6: Tambah kolom bantu
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["year"]  = df["date"].str[:4]
    df["month"] = df["date"].str[5:7]

    # Tentukan tipe sumber berdasarkan nama file
    def infer_source(row):
        src = str(row.get("_source_file", "")).lower()
        url = str(row.get("url", "")).lower()
        if "gdelt" in src:
            return "gdelt"
        if "rss" in src:
            if "cnbc" in url:    return "rss_cnbc"
            if "reuters" in url: return "rss_reuters"
            if "ap" in url:      return "rss_ap"
            return "rss_other"
        if "cnbc_sitemap" in src:
            return "cnbc_sitemap"
        if "newsapi" in src:
            return "newsapi"
        return "unknown"

    df["source_type"] = df.apply(infer_source, axis=1)

    # Pilih dan urutkan kolom output
    keep_cols = ["date", "year", "month", "title", "url", "source_type"]
    if "source" in df.columns:
        keep_cols.insert(5, "source")
    df = df[keep_cols].sort_values("date").reset_index(drop=True)

    log.info("\nTotal artikel bersih: %d", len(df))
    return df


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pipeline preprocessing berita geopolitik")
    parser.add_argument("--no-filter", action="store_true",
                        help="Simpan semua artikel tanpa filter keyword")
    args = parser.parse_args()

    log.info("Membaca semua file CSV dari: %s", RAW_DIR)
    df_raw = load_all_raw(RAW_DIR)
    if df_raw.empty:
        log.error("Tidak ada data untuk diproses.")
        return

    df_clean = run_pipeline(df_raw, apply_filter=not args.no_filter)

    if df_clean.empty:
        log.warning("Tidak ada data setelah preprocessing.")
        return

    # Statistik distribusi per tahun
    log.info("\nDistribusi artikel per tahun:")
    year_dist = df_clean["year"].value_counts().sort_index()
    for yr, cnt in year_dist.items():
        log.info("  %s: %d artikel", yr, cnt)

    log.info("\nDistribusi per sumber:")
    src_dist = df_clean["source_type"].value_counts()
    for src, cnt in src_dist.items():
        log.info("  %s: %d artikel", src, cnt)

    df_clean.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    log.info("\nDisimpan: %s", OUTPUT_FILE)
    log.info("\nSample 5 baris:\n%s", df_clean[["date","title","source_type"]].head(5).to_string())


if __name__ == "__main__":
    main()
