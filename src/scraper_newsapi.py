"""
=====================================================================
scraper_newsapi.py  -  Scraper Berita via NewsAPI.org
=====================================================================
Sumber : NewsAPI.org (https://newsapi.org)
Plan   : Free (Developer) - 100 request/hari, berita 1 bulan terakhir
         Paid              - Berita hingga 5 tahun ke belakang

CARA MENDAPATKAN API KEY (GRATIS):
1. Buka https://newsapi.org/register
2. Daftar dengan email
3. Salin API key yang dikirim ke email
4. Masukkan di variabel API_KEY di bawah

CATATAN:
- Free tier: data hanya 1 bulan terakhir
- Untuk data 5 tahun: upgrade ke plan berbayar ATAU gunakan scraper_gdelt.py

KEUNGGULAN:
- Query mudah, hasil langsung berupa JSON terstruktur
- Tidak perlu scraping manual
- Bisa filter berdasarkan: keyword, tanggal, sumber, bahasa, country

Cara pakai:
    python scraper_newsapi.py --apikey YOUR_KEY_HERE
    python scraper_newsapi.py --apikey YOUR_KEY --start 2024-01-01 --end 2024-03-01
"""

import requests
import pandas as pd
import time
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://newsapi.org/v2/everything"

# Keyword pencarian geopolitik & USD
SEARCH_QUERIES = [
    "dollar geopolitics",
    "USD exchange rate",
    "Federal Reserve",
    "US China trade war",
    "Russia Ukraine economy dollar",
    "BRICS currency dollar",
    "oil price dollar",
    "emerging market currency crisis",
    "US sanctions dollar",
    "geopolitical risk economy",
]

# Sumber berita yang relevan
PREFERRED_SOURCES = "reuters.com,apnews.com,cnbc.com,bloomberg.com,ft.com,wsj.com"


def fetch_newsapi(query, from_date, to_date, api_key,
                  page_size=100, page=1, language="en"):
    """
    Memanggil NewsAPI /everything endpoint.

    Parameter
    ---------
    query      : Kata kunci pencarian
    from_date  : Format YYYY-MM-DD
    to_date    : Format YYYY-MM-DD
    api_key    : API key NewsAPI
    page_size  : Artikel per halaman (max 100)
    page       : Nomor halaman
    language   : Bahasa ('en' untuk Inggris)
    """
    params = {
        "q"        : query,
        "from"     : from_date,
        "to"       : to_date,
        "language" : language,
        "sortBy"   : "publishedAt",
        "pageSize" : page_size,
        "page"     : page,
        "apiKey"   : api_key,
    }
    try:
        resp = requests.get(BASE_URL, params=params, timeout=30)
        data = resp.json()

        if resp.status_code == 429:
            log.warning("Rate limited! Menunggu 60 detik...")
            time.sleep(60)
            return [], 0

        if data.get("status") != "ok":
            log.error(f"Error API: {data.get('message','Unknown error')}")
            return [], 0

        articles    = data.get("articles", [])
        total_count = data.get("totalResults", 0)
        log.info(f"  '{query[:30]}' [{from_date}~{to_date}] p{page}: "
                 f"{len(articles)} artikel (total: {total_count})")
        return articles, total_count

    except Exception as e:
        log.error(f"  Error: {e}")
        return [], 0


def scrape_newsapi(api_key, queries=None, start_date=None, end_date=None,
                   delay_seconds=1.0, max_pages=5):
    """
    Scraping artikel dari NewsAPI untuk semua query.

    Parameter
    ---------
    api_key        : API key NewsAPI
    queries        : List keyword (default: SEARCH_QUERIES)
    start_date     : Tanggal mulai (YYYY-MM-DD), default: 30 hari lalu
    end_date       : Tanggal selesai (YYYY-MM-DD), default: hari ini
    delay_seconds  : Jeda antar request
    max_pages      : Maks halaman per query (100 artikel/halaman)
    """
    if queries is None:
        queries = SEARCH_QUERIES

    if start_date is None:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    all_articles = []
    seen_urls    = set()

    log.info(f"Periode: {start_date} -> {end_date}")
    log.info(f"Query: {len(queries)}, Max halaman per query: {max_pages}")

    for query in queries:
        log.info(f"\n[QUERY] '{query}'")
        for page in range(1, max_pages + 1):
            articles, total = fetch_newsapi(
                query=query, from_date=start_date, to_date=end_date,
                api_key=api_key, page=page,
            )

            for art in articles:
                url = art.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_articles.append({
                        "url"          : url,
                        "title"        : art.get("title", ""),
                        "description"  : art.get("description", ""),
                        "content"      : art.get("content", ""),
                        "published_at" : art.get("publishedAt", ""),
                        "source"       : art.get("source", {}).get("name", ""),
                        "author"       : art.get("author", ""),
                        "query"        : query,
                    })

            time.sleep(delay_seconds)

            # Hentikan jika sudah tidak ada lagi artikel
            if not articles or (page * 100) >= min(total, max_pages * 100):
                break

    log.info(f"\nTotal artikel unik: {len(all_articles)}")
    return pd.DataFrame(all_articles) if all_articles else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Scraper Berita via NewsAPI.org")
    parser.add_argument("--apikey", required=True,
                        help="API key NewsAPI (daftar gratis di newsapi.org)")
    parser.add_argument("--start", default=None,
                        help="Tanggal mulai YYYY-MM-DD (default: 30 hari lalu)")
    parser.add_argument("--end", default=None,
                        help="Tanggal selesai YYYY-MM-DD (default: hari ini)")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--pages", type=int, default=5,
                        help="Max halaman per query (1 halaman = 100 artikel)")
    args = parser.parse_args()

    df = scrape_newsapi(
        api_key      = args.apikey,
        start_date   = args.start,
        end_date     = args.end,
        delay_seconds= args.delay,
        max_pages    = args.pages,
    )

    if not df.empty:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = OUTPUT_DIR / f"newsapi_{ts}.csv"
        js_path  = OUTPUT_DIR / f"newsapi_{ts}.json"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        df.to_json(js_path, orient="records", force_ascii=False, indent=2)
        log.info(f"\nDisimpan: {csv_path}")
        log.info(f"\nSample:\n{df[['title','source','published_at']].head().to_string()}")
    else:
        log.warning("Tidak ada data yang berhasil diambil.")

if __name__ == "__main__":
    main()
