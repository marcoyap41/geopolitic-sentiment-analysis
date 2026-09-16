"""
=====================================================================
preprocessing_pipeline.py  -  Pipeline Pembersihan Data Berita
=====================================================================
Membaca file CSV hasil merge CNBC, membersihkan teks, dan memfilter
artikel yang relevan dengan geopolitik/USD.

Pipeline (urutan optimal):
  1. Baca file CSV spesifik (cnbc_real_titles_merged_*.csv terbaru)
  2. Filter rentang tanggal (1 Sep 2021 - 1 Sep 2026)   <- dipindah ke awal
     -> memangkas data sejak dini agar langkah berikutnya lebih ringan
  3. Deduplikasi berdasarkan URL
  4. Filter URL non-artikel (halaman kategori/index)
  5. Bersihkan teks judul
  6. Hapus baris judul kosong
  7. Filter keyword geopolitik (regex vectorized, 8 kategori)
  8. Tambah kolom bantu: year, month, keyword_category
  9. Simpan ke data/cleaned/news_geopolitik_clean.csv

Cara pakai:
    python preprocessing_pipeline.py
    python preprocessing_pipeline.py --no-filter
    python preprocessing_pipeline.py --input path/to/file.csv
"""

import pandas as pd
import re
import logging
import argparse
import html
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

ROOT_DIR   = Path(__file__).parent.parent
RAW_DIR    = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "data" / "cleaned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "news_geopolitik_clean.csv"

DATE_START = pd.Timestamp("2021-09-01")
DATE_END   = pd.Timestamp("2026-09-01")


# === Keyword Geopolitik & USD (8 Kategori) ===================================

CURRENCY_KEYWORDS = [
    "dollar", "USD", "exchange rate", "currency", "forex", "FX",
    "dollar index", "DXY", "dollar strength", "dollar weakness",
    "dollar rally", "dollar surge", "dollar falls", "dollar slips",
    "greenback", "currency war", "currency crisis", "devaluation",
    "depreciation", "appreciation", "monetary policy", "currency swap",
    "dollar dominance", "dedollarization", "de-dollarization",
    "petrodollar", "eurodollar",
]

CENTRAL_BANK_KEYWORDS = [
    "Federal Reserve", "Fed", "FOMC", "Jerome Powell",
    "interest rate", "rate hike", "rate cut", "rate decision",
    "basis points", "bps", "tightening", "easing", "tapering",
    "quantitative easing", "QE", "quantitative tightening", "QT",
    "inflation", "CPI", "PCE", "deflation", "stagflation",
    "ECB", "Bank of England", "Bank of Japan", "BOJ", "PBoC",
    "central bank", "monetary stimulus", "hawkish", "dovish",
]

CONFLICT_KEYWORDS = [
    "war", "conflict", "invasion", "military",
    "Russia Ukraine", "Ukraine war", "NATO",
    "Israel Gaza", "Middle East", "Iran", "Israel",
    "North Korea", "Taiwan", "South China Sea",
    "missile", "nuclear", "troops", "offensive",
    "ceasefire", "peace talks", "escalation",
    "coup", "insurgency", "civil war",
]

DIPLOMACY_KEYWORDS = [
    "sanctions", "sanction", "embargo",
    "US China", "Sino-American", "trade war", "tariff",
    "diplomatic", "summit", "bilateral", "multilateral",
    "alliance", "treaty", "agreement", "deal",
    "G7", "G20", "UN", "United Nations", "IMF", "World Bank",
    "WTO", "ASEAN", "geopolitical", "geopolitics",
    "foreign policy", "State Department", "Pentagon",
    "White House", "Congress", "Senate",
]

ENERGY_KEYWORDS = [
    "oil", "crude", "Brent", "WTI", "OPEC", "OPEC+",
    "oil price", "gas price", "natural gas", "LNG",
    "energy crisis", "fuel", "pipeline", "refinery",
    "energy sanctions", "oil embargo", "oil supply",
    "gold", "commodity", "commodity price",
]

MACRO_KEYWORDS = [
    "recession", "GDP", "economic growth", "slowdown",
    "debt", "deficit", "surplus", "fiscal", "budget",
    "treasury", "bond", "yield", "10-year yield",
    "stock market", "Wall Street", "S&P 500",
    "market crash", "financial crisis", "banking crisis",
    "default", "debt ceiling", "credit rating",
    "Moody s", "Fitch", "S&P downgrade",
    "global economy", "emerging market", "developing countries",
]

DEDOLLAR_KEYWORDS = [
    "BRICS", "de-dollarization", "yuan", "renminbi", "RMB",
    "digital currency", "CBDC", "digital yuan",
    "gold reserve", "reserve currency",
    "dollar alternative", "payment system", "SWIFT",
    "Russia payment", "China payment", "ruble",
    "petro-yuan", "oil-yuan",
]

US_POLITICS_KEYWORDS = [
    "Trump", "Biden", "election", "president",
    "US economy", "American economy", "US budget",
    "debt ceiling", "government shutdown",
    "US inflation", "US recession", "US GDP",
    "US jobs", "unemployment", "nonfarm payroll",
    "US trade deficit", "US exports",
]

KEYWORD_CATEGORIES = {
    "currency":     CURRENCY_KEYWORDS,
    "central_bank": CENTRAL_BANK_KEYWORDS,
    "conflict":     CONFLICT_KEYWORDS,
    "diplomacy":    DIPLOMACY_KEYWORDS,
    "energy":       ENERGY_KEYWORDS,
    "macro":        MACRO_KEYWORDS,
    "dedollar":     DEDOLLAR_KEYWORDS,
    "us_politics":  US_POLITICS_KEYWORDS,
}

CATEGORY_PATTERNS = {
    cat: re.compile(
        "|".join(r"\b" + re.escape(kw) + r"\b" for kw in kws),
        re.IGNORECASE,
    )
    for cat, kws in KEYWORD_CATEGORIES.items()
}

_all_keywords = list(dict.fromkeys(
    kw for kws in KEYWORD_CATEGORIES.values() for kw in kws
))
KEYWORD_PATTERN = re.compile(
    "|".join(r"\b" + re.escape(kw) + r"\b" for kw in _all_keywords),
    re.IGNORECASE,
)
log.info("Total keyword geopolitik unik: %d", len(_all_keywords))


# === Pattern URL non-artikel ==================================================

NON_ARTICLE_PATTERNS = re.compile(
    r"cnbc\.com/("
    r"us-top-news|world-top-news|europe-news|asia-news|"
    r"transportation|us-news|latest|cnbc-latest|market-insider|"
    r"financial-advisors|personal-finance|investing|markets|"
    r"business|technology|politics|entertainment|health|science|"
    r"select|pro|video|live-tv|digital-original|closing-bell|"
    r"mad-money|fast-money"
    r")/?$",
    re.IGNORECASE,
)


# === Helper functions =========================================================

def clean_title(title: str) -> str:
    if not isinstance(title, str) or not title.strip():
        return ""
    title = html.unescape(title)
    title = re.sub(
        r"\s*[-|]\s*(CNBC|Reuters|AP News|Bloomberg|FT|WSJ|BBC).*$",
        "", title, flags=re.IGNORECASE,
    )
    title = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", title)
    title = re.sub(r"<[^>]+>", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def is_article_url(url: str) -> bool:
    if not isinstance(url, str) or not url.strip():
        return False
    if re.search(r"/\d{4}/\d{2}/\d{2}/", url):
        return True
    if "cnbc.com" not in url:
        return True
    if NON_ARTICLE_PATTERNS.search(url):
        return False
    return True


def tag_keyword_category(title: str) -> str:
    matched = [cat for cat, pat in CATEGORY_PATTERNS.items() if pat.search(title)]
    return "|".join(matched) if matched else ""


# === Load CSV =================================================================

def load_csv(file_path: Path) -> pd.DataFrame:
    log.info("Membaca: %s  (%d KB)", file_path.name, file_path.stat().st_size // 1024)
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig", low_memory=False)
        df["_source_file"] = file_path.name
        log.info("  -> %d baris | kolom: %s", len(df), list(df.columns))
        return df
    except Exception as exc:
        log.error("Gagal membaca %s: %s", file_path.name, exc)
        return pd.DataFrame()


def find_latest_merged_csv(raw_dir: Path):
    candidates = sorted(raw_dir.glob("cnbc_real_titles_merged_*.csv"), reverse=True)
    return candidates[0] if candidates else None


# === Pipeline =================================================================

def run_pipeline(df: pd.DataFrame, apply_filter: bool = True) -> pd.DataFrame:
    log.info("\n=== PIPELINE PREPROCESSING ===")
    log.info("Input : %d baris", len(df))

    # Standardisasi nama kolom
    for candidate in ["date", "published", "seendate", "tanggal"]:
        if candidate in df.columns and candidate != "date":
            df = df.rename(columns={candidate: "date"})
            break

    for candidate in ["title", "Title", "headline"]:
        if candidate in df.columns and candidate != "title":
            df = df.rename(columns={candidate: "title"})
            break

    for col in ["url", "title", "date"]:
        if col not in df.columns:
            log.warning("Kolom '%s' tidak ditemukan -> diisi string kosong.", col)
            df[col] = ""

    # Step 2: Filter rentang tanggal (DIPINDAH KE AWAL)
    # Alasan: membuang artikel di luar rentang sejak dini (~60% data)
    # sehingga langkah-langkah mahal berikutnya bekerja pada data lebih kecil.
    before = len(df)
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    n_invalid = df["date_parsed"].isna().sum()
    mask_date = df["date_parsed"].between(DATE_START, DATE_END, inclusive="left")
    df = df[mask_date].copy()
    log.info(
        "Step 2 - Filter tanggal     : %d -> %d  (-%d di luar %s s/d %s | -%d tanggal tidak valid)",
        before, len(df), before - len(df) - n_invalid,
        DATE_START.date(), DATE_END.date(), n_invalid,
    )

    # Step 3: Deduplikasi URL
    before = len(df)
    df = df.drop_duplicates(subset=["url"], keep="first")
    log.info("Step 3 - Deduplikasi URL    : %d -> %d  (-%d duplikat)", before, len(df), before - len(df))

    # Step 4: Filter URL non-artikel
    before = len(df)
    df = df[df["url"].apply(is_article_url)].copy()
    log.info("Step 4 - Filter non-artikel : %d -> %d  (-%d halaman kategori/index)", before, len(df), before - len(df))

    # Step 5: Bersihkan teks judul
    n_empty_before = (df["title"].isna() | (df["title"].astype(str).str.strip() == "")).sum()
    df["title"] = df["title"].apply(clean_title)
    n_empty_after = (df["title"] == "").sum()
    log.info("Step 5 - Bersihkan judul    : %d judul kosong sebelum -> %d setelah cleaning", n_empty_before, n_empty_after)

    # Step 6: Hapus baris judul kosong
    before = len(df)
    df = df[df["title"] != ""].copy()
    log.info("Step 6 - Hapus judul kosong : %d -> %d  (-%d baris)", before, len(df), before - len(df))

    # Step 7: Filter keyword geopolitik
    if apply_filter:
        before = len(df)
        mask = df["title"].str.contains(KEYWORD_PATTERN, na=False)
        df   = df[mask].copy()
        log.info("Step 7 - Filter geopolitik  : %d -> %d  (-%d tidak relevan)", before, len(df), before - len(df))
    else:
        log.info("Step 7 - Filter geopolitik  : DILEWATI (--no-filter)")

    # Step 8: Tambah kolom bantu
    df["date"]  = df["date_parsed"].dt.strftime("%Y-%m-%d")
    df["year"]  = df["date_parsed"].dt.year.astype(str)
    df["month"] = df["date_parsed"].dt.month.astype(str).str.zfill(2)
    df = df.drop(columns=["date_parsed"])

    if apply_filter:
        df["keyword_category"] = df["title"].apply(tag_keyword_category)

    def infer_source(row):
        src = str(row.get("_source_file", "")).lower()
        url = str(row.get("url", "")).lower()
        if "gdelt"        in src: return "gdelt"
        if "rss"          in src:
            if "cnbc"    in url: return "rss_cnbc"
            if "reuters" in url: return "rss_reuters"
            if "ap"      in url: return "rss_ap"
            return "rss_other"
        if "cnbc_sitemap" in src: return "cnbc_sitemap"
        if "cnbc"         in src: return "cnbc_merged"
        if "newsapi"      in src: return "newsapi"
        return "unknown"

    df["source_type"] = df.apply(infer_source, axis=1)

    keep_cols = ["date", "year", "month", "title", "url", "source_type"]
    if apply_filter:
        keep_cols.append("keyword_category")
    if "source" in df.columns:
        keep_cols.insert(5, "source")
    df = df[[c for c in keep_cols if c in df.columns]]
    df = df.sort_values("date").reset_index(drop=True)

    log.info("\nTotal artikel bersih: %d", len(df))
    return df


# === Main =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Pipeline preprocessing berita geopolitik/USD")
    parser.add_argument("--no-filter", action="store_true",
                        help="Simpan semua artikel tanpa filter keyword geopolitik")
    parser.add_argument("--input", type=str, default=None,
                        help="Path ke file CSV input (default: file cnbc_real_titles_merged_*.csv terbaru)")
    args = parser.parse_args()

    if args.input:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = ROOT_DIR / input_path
    else:
        input_path = find_latest_merged_csv(RAW_DIR)
        if input_path is None:
            log.error("Tidak ditemukan file cnbc_real_titles_merged_*.csv di: %s", RAW_DIR)
            log.error("Gunakan --input <path> untuk menentukan file secara manual.")
            return

    if not input_path.exists():
        log.error("File tidak ditemukan: %s", input_path)
        return

    log.info("File input : %s", input_path)
    df_raw = load_csv(input_path)
    if df_raw.empty:
        log.error("Tidak ada data untuk diproses.")
        return

    df_clean = run_pipeline(df_raw, apply_filter=not args.no_filter)

    if df_clean.empty:
        log.warning("Tidak ada data setelah preprocessing.")
        return

    log.info("\nDistribusi artikel per tahun:")
    for yr, cnt in df_clean["year"].value_counts().sort_index().items():
        log.info("  %s : %d artikel", yr, cnt)

    log.info("\nDistribusi per sumber:")
    for src, cnt in df_clean["source_type"].value_counts().items():
        log.info("  %-15s : %d artikel", src, cnt)

    if "keyword_category" in df_clean.columns:
        log.info("\nDistribusi per kategori keyword (multi-label):")
        cat_counts = (
            df_clean["keyword_category"]
            .str.split("|")
            .explode()
            .value_counts()
        )
        for cat, cnt in cat_counts.items():
            log.info("  %-15s : %d artikel", cat, cnt)

    df_clean.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    log.info("\nDisimpan ke : %s", OUTPUT_FILE)
    log.info("\nSample 5 baris:\n%s",
             df_clean[["date", "title", "source_type"]].head(5).to_string())


if __name__ == "__main__":
    main()
