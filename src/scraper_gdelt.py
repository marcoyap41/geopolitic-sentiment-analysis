"""
=====================================================================
scraper_gdelt.py  -  Scraper Berita Geopolitik via GDELT API
=====================================================================
Sumber  : GDELT 2.0 DOC API (https://api.gdeltproject.org/api/v2/doc/doc)
Periode : 5 tahun ke belakang (September 2021 - September 2026)
Topik   : Geopolitik global dan dampak terhadap nilai tukar USD

Mengapa GDELT?
- API publik GRATIS tanpa API key
- Data historis lengkap, terstruktur (JSON/CSV)
- Rate limit: 1 request per 5 detik (sudah ditangani oleh scraper ini)

CATATAN RATE LIMIT:
GDELT membatasi 1 request per 5 detik.
Script ini otomatis menunggu sesuai aturan tersebut.

Cara pakai:
    python scraper_gdelt.py              # mode test (cepat)
    python scraper_gdelt.py --mode full  # scraping 5 tahun penuh (~3-4 jam)
    python scraper_gdelt.py --mode full --chunk 180  # chunk lebih besar (lebih cepat)
"""

import requests
import pandas as pd
import json
import time
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# --- Konfigurasi ---
BASE_URL    = "https://api.gdeltproject.org/api/v2/doc/doc"
OUTPUT_DIR  = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2021-09-01"
END_DATE   = "2026-09-01"

# Delay antar request - 12 detik lebih aman dari rate limit GDELT
# (GDELT policy: 1 req/5 detik, 12 detik memberi buffer ekstra)
GDELT_MIN_DELAY = 12.0

GEOPOLITICAL_KEYWORDS = [
    "dollar geopolitics",
    "USD exchange rate",
    "Federal Reserve rate",
    "US China trade",
    "Russia Ukraine economy",
    "oil price dollar",
    "US sanctions dollar",
    "BRICS currency",
    "dollar index DXY",
    "emerging market currency",
    "US inflation dollar",
    "IMF dollar",
    "geopolitical risk currency",
    "US debt dollar",
    "global currency crisis",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


# Pelacak berapa kali berturut-turut kena 429 (global state)
_consecutive_failures = 0


def query_gdelt_articles(keyword, startdatetime, enddatetime,
                         max_records=250, retries=5):
    """
    Memanggil GDELT DOC API dengan retry cerdas dan exponential backoff.

    Strategi backoff saat 429:
      attempt 1 -> tunggu 60 detik
      attempt 2 -> tunggu 120 detik
      attempt 3 -> tunggu 180 detik
      attempt 4 -> tunggu 240 detik
      attempt 5 -> tunggu 300 detik

    Parameter
    ---------
    keyword       : Kata kunci pencarian
    startdatetime : Format YYYYMMDDHHMMSS
    enddatetime   : Format YYYYMMDDHHMMSS
    max_records   : Maksimum artikel per request (GDELT max=250)
    retries       : Jumlah percobaan ulang jika gagal (default: 5)
    """
    global _consecutive_failures

    params = {
        "query"         : keyword,
        "mode"          : "artlist",
        "startdatetime" : startdatetime,
        "enddatetime"   : enddatetime,
        "maxrecords"    : max_records,
        "sourcelang"    : "English",
        "format"        : "json",
        "sort"          : "DateDesc",
    }

    # Cooldown global jika sudah sering gagal berturut-turut
    if _consecutive_failures >= 3:
        cooldown = 120
        log.warning(f"  [{_consecutive_failures}x gagal berturut-turut] Cool down {cooldown}s...")
        time.sleep(cooldown)
        _consecutive_failures = 0

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(BASE_URL, params=params,
                                headers=HEADERS, timeout=40)

            if resp.status_code == 429:
                # Exponential backoff: 60, 120, 180, 240, 300 detik
                wait = min(60 * attempt, 300)
                log.warning(f"  Rate limited! Attempt {attempt}/{retries} - tunggu {wait}s...")
                _consecutive_failures += 1
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])
            _consecutive_failures = 0  # reset saat berhasil
            log.info(f"  '{keyword[:35]}' [{startdatetime[:8]}-{enddatetime[:8]}]: {len(articles)} artikel")
            return articles

        except requests.Timeout:
            log.warning(f"  Timeout (attempt {attempt}/{retries}), retry dalam 30s...")
            time.sleep(30)
        except requests.RequestException as e:
            log.error(f"  Error (attempt {attempt}/{retries}): {e}")
            time.sleep(GDELT_MIN_DELAY)

    _consecutive_failures += 1
    log.error(f"  Gagal setelah {retries} percobaan - chunk ini dilewati")
    return []


def scrape_all_keywords(keywords, start_date=START_DATE, end_date=END_DATE,
                        delay_seconds=GDELT_MIN_DELAY, chunk_days=90):
    """
    Scraping semua keyword dengan chunking waktu.

    GDELT membatasi 250 artikel per request, sehingga kita memecah
    rentang 5 tahun menjadi chunk (misal 90 hari) per keyword.

    Estimasi waktu untuk 15 keyword x 5 tahun x chunk 90 hari:
      = 15 x 20 chunk x 6 detik delay = ~1800 detik = ~30 menit
    """
    all_articles = []
    seen_urls    = set()

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt   = datetime.strptime(end_date,   "%Y-%m-%d")

    # Buat daftar chunk waktu
    time_chunks = []
    current = start_dt
    while current < end_dt:
        chunk_end = min(current + timedelta(days=chunk_days), end_dt)
        time_chunks.append((current, chunk_end))
        current = chunk_end

    total = len(keywords) * len(time_chunks)
    log.info(f"Total request yang direncanakan: {total}")
    log.info(f"Estimasi waktu: ~{total * delay_seconds / 60:.0f} menit")

    req_count = 0
    for keyword in keywords:
        log.info(f"\n[KEYWORD] '{keyword}'")
        for chunk_start, chunk_end in time_chunks:
            articles = query_gdelt_articles(
                keyword       = keyword,
                startdatetime = chunk_start.strftime("%Y%m%d%H%M%S"),
                enddatetime   = chunk_end.strftime("%Y%m%d%H%M%S"),
            )
            for article in articles:
                url = article.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    article["keyword"] = keyword
                    all_articles.append(article)

            req_count += 1
            log.info(f"  Progress: {req_count}/{total} | Unik: {len(all_articles)}")

            # Simpan checkpoint setiap 50 request
            if req_count % 50 == 0 and all_articles:
                _save_checkpoint(all_articles, req_count)

            # Rate limiting wajib GDELT
            time.sleep(delay_seconds)

    log.info(f"\nTotal artikel unik: {len(all_articles)}")
    if not all_articles:
        return pd.DataFrame()

    df = pd.DataFrame(all_articles)
    standard_cols = ["url","title","seendate","domain","language","sourcecountry","keyword"]
    for col in standard_cols:
        if col not in df.columns:
            df[col] = None
    return df[standard_cols + [c for c in df.columns if c not in standard_cols]]


def _save_checkpoint(articles, count):
    """Simpan checkpoint sementara agar data tidak hilang jika terjadi error."""
    checkpoint_path = OUTPUT_DIR / f"gdelt_checkpoint_{count}.json"
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    log.info(f"  [CHECKPOINT] Disimpan: {checkpoint_path}")


def save_raw_data(df, prefix="gdelt_news"):
    """Simpan data mentah ke CSV dan JSON."""
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"{prefix}_{ts}.csv"
    js_path  = OUTPUT_DIR / f"{prefix}_{ts}.json"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_json(js_path, orient="records", force_ascii=False, indent=2)
    log.info(f"\nData disimpan:")
    log.info(f"  CSV  -> {csv_path}")
    log.info(f"  JSON -> {js_path}")
    return csv_path, js_path


def quick_test():
    """Test cepat: 1 keyword, 1 bulan data historis (Jan 2024)."""
    log.info("=== MODE TEST: 1 keyword, Jan 2024 ===")
    log.info("Menunggu 6 detik sesuai rate limit GDELT...")
    time.sleep(6)

    df = scrape_all_keywords(
        keywords      = ["dollar geopolitics"],
        start_date    = "2024-01-01",
        end_date      = "2024-02-01",
        delay_seconds = GDELT_MIN_DELAY,
        chunk_days    = 31,
    )
    if not df.empty:
        log.info(f"\nSample data:")
        log.info(f"\n{df[['title','domain','seendate']].head(5).to_string()}")
        save_raw_data(df, "gdelt_test")
        log.info(f"\n[SUKSES] {len(df)} artikel berhasil diambil!")
    else:
        log.warning("Tidak ada data. Kemungkinan penyebab:")
        log.warning("  1. Rate limit masih aktif - tunggu beberapa menit lalu coba lagi")
        log.warning("  2. Koneksi internet bermasalah")
        log.warning("  3. GDELT server sedang down")
    return df


def main():
    parser = argparse.ArgumentParser(description="Scraper Berita Geopolitik via GDELT API")
    parser.add_argument("--mode", choices=["full","test"], default="test",
                        help="'test' untuk quick test, 'full' untuk 5 tahun penuh")
    parser.add_argument("--start", default=START_DATE, help="Tanggal mulai YYYY-MM-DD")
    parser.add_argument("--end",   default=END_DATE,   help="Tanggal selesai YYYY-MM-DD")
    parser.add_argument("--delay", type=float, default=GDELT_MIN_DELAY,
                        help="Jeda antar request (min 6 detik untuk GDELT)")
    parser.add_argument("--chunk", type=int, default=90,
                        help="Ukuran chunk waktu dalam hari (default: 90)")
    args = parser.parse_args()

    # Pastikan delay minimal 6 detik untuk GDELT
    delay = max(args.delay, GDELT_MIN_DELAY)
    if delay != args.delay:
        log.warning(f"Delay diset ke minimum GDELT: {GDELT_MIN_DELAY} detik")

    if args.mode == "test":
        quick_test()
    else:
        log.info(f"=== SCRAPING PENUH: {args.start} s/d {args.end} ===")
        log.info(f"Keyword: {len(GEOPOLITICAL_KEYWORDS)}, Delay: {delay}s, Chunk: {args.chunk} hari")
        df = scrape_all_keywords(
            keywords      = GEOPOLITICAL_KEYWORDS,
            start_date    = args.start,
            end_date      = args.end,
            delay_seconds = delay,
            chunk_days    = args.chunk,
        )
        if not df.empty:
            save_raw_data(df, "gdelt_geopolitics_5yr")
            log.info(f"Selesai! {len(df)} artikel tersimpan.")
        else:
            log.error("Scraping gagal atau tidak ada data.")

if __name__ == "__main__":
    main()
