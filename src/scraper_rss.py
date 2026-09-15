"""
=====================================================================
scraper_rss.py  -  Scraper Berita via RSS Feed (Reuters & AP News)
=====================================================================
Sumber alternatif selain GDELT menggunakan RSS Feed publik.
Reuters dan AP News menyediakan RSS yang bisa diakses langsung.

CATATAN: RSS feed hanya menyimpan berita beberapa hari/minggu terakhir.
Untuk data historis 5 tahun, GUNAKAN scraper_gdelt.py sebagai SUMBER UTAMA.
Scraper ini berguna untuk memperkaya dataset dengan berita real-time.

Cara pakai:
    python scraper_rss.py              # ambil semua feed
    python scraper_rss.py --source reuters
    python scraper_rss.py --source ap
"""

import feedparser
import requests
import pandas as pd
import time
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from bs4 import BeautifulSoup

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# RSS Feed URLs
RSS_FEEDS = {
    "reuters": {
        "world"   : "https://feeds.reuters.com/reuters/worldNews",
        "business": "https://feeds.reuters.com/reuters/businessNews",
        "us"      : "https://feeds.reuters.com/Reuters/domesticNews",
        "markets" : "https://feeds.reuters.com/reuters/UKFocus",
    },
    "ap_news": {
        "world"   : "https://rsshub.app/apnews/topics/world-news",
        "business": "https://rsshub.app/apnews/topics/business",
        "politics": "https://rsshub.app/apnews/topics/politics",
    },
    "cnbc": {
        "world"   : "https://www.cnbc.com/id/100727362/device/rss/rss.html",
        "economy" : "https://www.cnbc.com/id/20910258/device/rss/rss.html",
        "forex"   : "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    },
}

# Keyword filter relevan geopolitik & USD
GEOPOLITICAL_KEYWORDS_FILTER = [
    "dollar", "usd", "currency", "exchange rate", "fed", "federal reserve",
    "geopolitic", "sanction", "trade war", "inflation", "interest rate",
    "ukraine", "russia", "china", "opec", "oil", "brics", "imf", "world bank",
    "debt", "treasury", "forex", "monetary", "economic", "gdp",
]


def parse_rss_feed(url: str, source_name: str) -> list[dict]:
    """Parse satu RSS feed dan kembalikan list artikel."""
    log.info(f"  Fetching: {url}")
    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries:
            published = entry.get("published", entry.get("updated", ""))
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                published = dt.strftime("%Y-%m-%d %H:%M:%S")

            # Gabungkan title + summary untuk filter keyword
            content_lower = (
                entry.get("title", "") + " " + entry.get("summary", "")
            ).lower()

            articles.append({
                "url"       : entry.get("link", ""),
                "title"     : entry.get("title", ""),
                "summary"   : entry.get("summary", ""),
                "published" : published,
                "source"    : source_name,
            })
        log.info(f"    -> {len(articles)} artikel ditemukan")
        return articles
    except Exception as e:
        log.error(f"  Error fetch '{url}': {e}")
        return []


def filter_geopolitical(articles: list[dict]) -> list[dict]:
    """Filter artikel yang relevan dengan geopolitik & USD."""
    filtered = []
    for art in articles:
        text = (art.get("title","") + " " + art.get("summary","")).lower()
        if any(kw in text for kw in GEOPOLITICAL_KEYWORDS_FILTER):
            filtered.append(art)
    log.info(f"  Filter: {len(articles)} -> {len(filtered)} artikel relevan")
    return filtered


def scrape_rss(sources: list[str] = None, apply_filter: bool = True) -> pd.DataFrame:
    """
    Scraping semua RSS feed dari sumber yang dipilih.

    Parameter
    ---------
    sources      : List nama sumber ['reuters','ap_news','cnbc'] atau None (semua)
    apply_filter : Jika True, hanya ambil artikel geopolitik/USD
    """
    if sources is None:
        sources = list(RSS_FEEDS.keys())

    all_articles = []
    seen_urls    = set()

    for source in sources:
        if source not in RSS_FEEDS:
            log.warning(f"Sumber '{source}' tidak dikenali, skip.")
            continue

        log.info(f"\n[SOURCE] {source.upper()}")
        for category, url in RSS_FEEDS[source].items():
            articles = parse_rss_feed(url, f"{source}/{category}")

            if apply_filter:
                articles = filter_geopolitical(articles)

            for art in articles:
                if art["url"] and art["url"] not in seen_urls:
                    seen_urls.add(art["url"])
                    all_articles.append(art)

            time.sleep(1.0)  # rate limiting

    log.info(f"\nTotal artikel unik: {len(all_articles)}")
    return pd.DataFrame(all_articles) if all_articles else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None,
                        help="Sumber: reuters, ap_news, cnbc (default: semua)")
    parser.add_argument("--no-filter", action="store_true",
                        help="Nonaktifkan filter keyword geopolitik")
    args = parser.parse_args()

    sources = [args.source] if args.source else None
    df = scrape_rss(sources=sources, apply_filter=not args.no_filter)

    if not df.empty:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        src = args.source or "all"
        csv_path = OUTPUT_DIR / f"rss_{src}_{ts}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        log.info(f"\nDisimpan: {csv_path}")
        log.info(f"\nSample:\n{df[['title','source','published']].head().to_string()}")
    else:
        log.warning("Tidak ada data yang berhasil diambil.")

if __name__ == "__main__":
    main()
