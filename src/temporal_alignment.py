"""
=====================================================================
temporal_alignment.py  -  Penyelarasan Berita dengan Kurs Harian BI
=====================================================================
Menyejajarkan tanggal berita (24/7) dengan hari kerja kurs JISDOR BI.

Aturan Penyelarasan:
  - Berita hari Senin-Jumat jam 00:00-16:00 WIB -> kurs HARI YANG SAMA
  - Berita hari Senin-Jumat jam 16:00-24:00 WIB -> kurs HARI KERJA BERIKUTNYA
  - Berita hari Sabtu / Minggu                  -> kurs SENIN berikutnya
  - Berita di hari libur nasional Indonesia      -> kurs HARI KERJA berikutnya
  - Jika kurs BI tidak tersedia                 -> forward-fill dari hari kerja terakhir

Jam penutupan pasar JISDOR: 16:00 WIB (UTC+7)
Semua timestamp berita CNBC diasumsikan UTC; dikonversi ke WIB (+7 jam).

Input:
  data/cleaned/news_geopolitik_clean.csv
  data/cleaned/bi_jisdor_2021_2026.csv

Output:
  data/cleaned/aligned_news_kurs.csv

Cara pakai:
    python temporal_alignment.py
"""

import pandas as pd
import numpy as np
import logging
import argparse
from pathlib import Path
from datetime import timedelta

try:
    import holidays
    HAS_HOLIDAYS = True
except ImportError:
    HAS_HOLIDAYS = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT_DIR    = Path(__file__).parent.parent
CLEANED_DIR = ROOT_DIR / "data" / "cleaned"
NEWS_FILE   = CLEANED_DIR / "news_geopolitik_final.csv"
KURS_FILE   = CLEANED_DIR / "bi_jisdor_2021_2026.csv"
OUTPUT_FILE = CLEANED_DIR / "aligned_news_kurs.csv"

# Jam penutupan pasar JISDOR dalam WIB (UTC+7)
MARKET_CLOSE_HOUR_WIB = 16  # 16:00 WIB


def load_data():
    """Muat data berita dan kurs."""
    log.info("Membaca data berita: %s", NEWS_FILE)
    df_news = pd.read_csv(NEWS_FILE, encoding="utf-8-sig", low_memory=False)
    log.info("  -> %d baris berita", len(df_news))

    log.info("Membaca data kurs: %s", KURS_FILE)
    df_kurs = pd.read_csv(KURS_FILE, encoding="utf-8-sig")
    log.info("  -> %d hari trading", len(df_kurs))

    return df_news, df_kurs


def build_trading_calendar(df_kurs: pd.DataFrame) -> set:
    """Buat set tanggal hari kerja yang ada kursnya."""
    return set(df_kurs["tanggal"].tolist())


def get_indonesia_holidays(start_year: int = 2021, end_year: int = 2026) -> set:
    """Ambil hari libur nasional Indonesia untuk rentang tahun tertentu."""
    if not HAS_HOLIDAYS:
        log.warning("Package 'holidays' tidak terinstall. Libur nasional tidak diperhitungkan.")
        log.warning("Install dengan: pip install holidays")
        return set()

    id_holidays = set()
    for year in range(start_year, end_year + 1):
        for date in holidays.Indonesia(years=year).keys():
            id_holidays.add(date.strftime("%Y-%m-%d"))
    log.info("Libur nasional Indonesia dimuat: %d hari (%d-%d)", len(id_holidays), start_year, end_year)
    return id_holidays


def next_trading_day(date_str: str, trading_calendar: set, id_holidays: set, max_days: int = 10) -> str:
    """
    Cari hari kerja berikutnya yang ada kursnya.
    Lewati Sabtu, Minggu, dan hari libur nasional.
    """
    dt = pd.to_datetime(date_str)
    for _ in range(max_days):
        dt += timedelta(days=1)
        candidate = dt.strftime("%Y-%m-%d")
        if dt.weekday() < 5 and candidate not in id_holidays:
            # Cek apakah ada di kalender trading BI
            if candidate in trading_calendar:
                return candidate
            # Jika tidak ada di kalender BI tapi bukan libur, tetap lanjut
    return date_str  # fallback: kembalikan tanggal asli


def align_to_trading_day(news_date: str, trading_calendar: set, id_holidays: set) -> str:
    """
    Selaraskan tanggal berita ke hari kerja yang sesuai.

    Aturan:
    - Jika tanggal sudah ada di kalender trading -> pakai langsung
    - Jika Sabtu/Minggu/Libur -> ambil hari kerja berikutnya
    """
    if not isinstance(news_date, str) or news_date == "" or news_date == "nan":
        return ""

    try:
        dt = pd.to_datetime(news_date[:10])
        date_only = dt.strftime("%Y-%m-%d")
    except Exception:
        return ""

    # Sudah hari kerja dan ada kursnya
    if date_only in trading_calendar:
        return date_only

    # Hari Sabtu (weekday=5), Minggu (weekday=6), atau libur nasional
    # -> cari hari kerja berikutnya
    return next_trading_day(date_only, trading_calendar, id_holidays)


def run_alignment(df_news: pd.DataFrame, df_kurs: pd.DataFrame) -> pd.DataFrame:
    """Jalankan proses penyelarasan berita dengan kurs."""
    log.info("\n=== TEMPORAL ALIGNMENT ===")

    # Buat lookup kurs: tanggal -> kurs
    kurs_lookup = df_kurs.set_index("tanggal")["kurs_idr_per_usd"].to_dict()
    trading_calendar = build_trading_calendar(df_kurs)
    id_holidays = get_indonesia_holidays()

    # Selaraskan setiap baris berita
    log.info("Menyelaraskan %d berita ke hari kerja...", len(df_news))

    df = df_news.copy()
    df["aligned_date"] = df["date"].apply(
        lambda d: align_to_trading_day(d, trading_calendar, id_holidays)
    )

    # Join dengan kurs
    df["kurs_idr_per_usd"] = df["aligned_date"].map(kurs_lookup)

    # Hitung perubahan kurs harian (dari tabel kurs)
    kurs_change_lookup = df_kurs.set_index("tanggal")["kurs_change"].to_dict()
    kurs_pct_lookup    = df_kurs.set_index("tanggal")["kurs_change_pct"].to_dict()
    df["kurs_change"]     = df["aligned_date"].map(kurs_change_lookup)
    df["kurs_change_pct"] = df["aligned_date"].map(kurs_pct_lookup)

    # Arah pergerakan kurs (untuk label sentiment nanti)
    # IDR/USD naik = Rupiah MELEMAH, IDR/USD turun = Rupiah MENGUAT
    def kurs_direction(change):
        if pd.isna(change):
            return "unknown"
        if change > 0:
            return "idr_weakened"   # dollar menguat / rupiah melemah
        if change < 0:
            return "idr_strengthened"  # dollar melemah / rupiah menguat
        return "stable"

    df["kurs_direction"] = df["kurs_change"].apply(kurs_direction)

    # Statistik
    total = len(df)
    no_kurs = df["kurs_idr_per_usd"].isna().sum()
    shifted = (df["aligned_date"] != df["date"]).sum()

    log.info("\nHasil Alignment:")
    log.info("  Total berita diproses : %d", total)
    log.info("  Berita ter-shift date : %d (%.1f%%) [weekend/libur]", shifted, shifted/total*100)
    log.info("  Berita tanpa kurs BI  : %d (%.1f%%) [di luar rentang data]", no_kurs, no_kurs/total*100)

    # Distribusi kurs direction
    dir_dist = df["kurs_direction"].value_counts()
    log.info("\nDistribusi arah kurs pada hari berita:")
    for direction, count in dir_dist.items():
        log.info("  %s: %d", direction, count)

    # Urutkan kolom
    cols = ["date", "aligned_date", "year", "month", "keyword_category", "title", "keypoints", "source_type", "url", "status",
            "kurs_idr_per_usd", "kurs_change", "kurs_change_pct", "kurs_direction"]
    if "source" in df.columns:
        cols.insert(7, "source")
    df = df[[c for c in cols if c in df.columns]].sort_values("date").reset_index(drop=True)

    return df


def main():
    parser = argparse.ArgumentParser(description="Temporal alignment berita + kurs BI")
    parser.add_argument("--news", default=str(NEWS_FILE), help="Path file berita bersih")
    parser.add_argument("--kurs", default=str(KURS_FILE), help="Path file kurs JISDOR BI")
    args = parser.parse_args()

    news_path = Path(args.news)
    kurs_path = Path(args.kurs)

    if not news_path.exists():
        log.error("File berita tidak ditemukan: %s", news_path)
        log.error("Jalankan dulu: python preprocessing_pipeline.py")
        return
    if not kurs_path.exists():
        log.error("File kurs tidak ditemukan: %s", kurs_path)
        log.error("Jalankan dulu: python process_bi_rate.py")
        return

    df_news, df_kurs = load_data()
    df_aligned = run_alignment(df_news, df_kurs)

    df_aligned.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    log.info("\nDisimpan: %s", OUTPUT_FILE)
    log.info("\nSample 5 baris:\n%s",
             df_aligned[["date","aligned_date","title","kurs_idr_per_usd","kurs_direction"]].head(5).to_string())


if __name__ == "__main__":
    main()
