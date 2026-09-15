"""
=====================================================================
scraper_cnbc_sitemap.py  -  Scraper Judul Artikel CNBC via Sitemap
=====================================================================
Strategi:
  1. Baca sitemapAll.xml -> dapat 13 sub-sitemap
  2. Parse setiap sub-sitemap -> dapat daftar URL + lastmod
  3. Filter berdasarkan rentang tanggal 2021-2026
  4. Ambil judul dari slug URL (cepat) atau dari HTML (lengkap)

Dua mode:
  --mode url_only   : Hanya kumpulkan URL + tanggal dari sitemap (CEPAT)
  --mode with_title : Tambahkan judul dari halaman HTML (lebih lambat)

Cara pakai:
    python scraper_cnbc_sitemap.py --mode url_only
    python scraper_cnbc_sitemap.py --mode with_title --workers 5
    python scraper_cnbc_sitemap.py --start 2021-01-01 --end 2026-09-01
"""

import requests
import pandas as pd
import time
import logging
import argparse
import re
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SITEMAP_INDEX = "https://www.cnbc.com/sitemapAll.xml"
DEFAULT_START = "2021-09-01"
DEFAULT_END   = "2026-09-15"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

GEOPOLITICAL_KEYWORDS = [
    "dollar", "usd", "currency", "exchange rate", "fed", "federal reserve",
    "geopolitic", "sanction", "trade war", "inflation", "interest rate",
    "ukraine", "russia", "china", "opec", "oil", "brics", "imf", "world bank",
    "debt", "treasury", "forex", "monetary", "economic", "gdp", "tariff",
    "recession", "war", "conflict", "iran", "north korea", "taiwan",
]

NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


def get_sub_sitemaps() -> list:
    """Ambil semua URL sub-sitemap dari sitemapAll.xml."""
    log.info("Fetching sitemap index: %s", SITEMAP_INDEX)
    r = requests.get(SITEMAP_INDEX, headers=HEADERS, timeout=15)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    sitemaps = [s.find("sm:loc", NS).text for s in root.findall("sm:sitemap", NS)]
    log.info("  -> %d sub-sitemap ditemukan", len(sitemaps))
    return sitemaps


def parse_sub_sitemap(url: str, start_date: str, end_date: str) -> list:
    """Parse satu sub-sitemap XML dan filter berdasarkan tanggal."""
    try:
        log.info("  Parsing: %s", url)
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        articles = []

        for entry in root.findall("sm:url", NS):
            loc  = entry.find("sm:loc", NS)
            lmod = entry.find("sm:lastmod", NS)
            if loc is None:
                continue

            article_url = loc.text.strip()
            date_str    = lmod.text[:10] if lmod is not None and lmod.text else ""

            if not date_str:
                m = re.search(r"/(\d{4}/\d{2}/\d{2})/", article_url)
                date_str = m.group(1).replace("/", "-") if m else ""

            if date_str and (start_date <= date_str <= end_date):
                articles.append({"url": article_url, "date": date_str})

        log.info("    -> %d artikel dalam rentang tanggal", len(articles))
        return articles

    except Exception as e:
        log.error("  Error parsing %s: %s", url, e)
        return []


def collect_all_urls(start_date: str, end_date: str) -> pd.DataFrame:
    """Kumpulkan semua URL CNBC dalam rentang tanggal dari semua sub-sitemap."""
    sitemaps = get_sub_sitemaps()
    all_articles = []
    seen_urls    = set()

    for sm_url in sitemaps:
        articles = parse_sub_sitemap(sm_url, start_date, end_date)
        for art in articles:
            if art["url"] not in seen_urls:
                seen_urls.add(art["url"])
                all_articles.append(art)
        time.sleep(0.5)

    log.info("\nTotal URL unik dalam rentang: %d", len(all_articles))
    return pd.DataFrame(all_articles) if all_articles else pd.DataFrame()


def fetch_title(url: str) -> str:
    """Ambil judul artikel dari meta og:title atau tag <title>."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")

        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            return og["content"].strip()

        h1 = soup.find("h1", class_=re.compile(r"headline|title", re.I))
        if h1:
            return h1.get_text(strip=True)

        title_tag = soup.find("title")
        if title_tag:
            t = title_tag.get_text(strip=True)
            t = re.sub(r"\s*[-|]\s*CNBC.*$", "", t, flags=re.I)
            return t.strip()

        return ""
    except Exception:
        return ""


def title_from_slug(url: str) -> str:
    """Ekstrak judul kasar dari slug URL (tanpa request HTTP)."""
    m = re.search(r"/\d{4}/\d{2}/\d{2}/([^/?#]+?)(?:\.html)?$", url)
    if m:
        slug = m.group(1).replace("-", " ")
        return slug.capitalize()
    return ""


def add_titles_parallel(df: pd.DataFrame, max_workers: int = 5) -> pd.DataFrame:
    """Tambahkan kolom title ke DataFrame secara paralel."""
    log.info("\nMengambil judul untuk %d artikel (workers=%d)...", len(df), max_workers)
    titles = [""] * len(df)
    urls   = df["url"].tolist()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(fetch_title, url): i for i, url in enumerate(urls)}
        done = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                titles[idx] = future.result()
            except Exception:
                titles[idx] = ""
            done += 1
            if done % 100 == 0:
                log.info("  Progress: %d/%d", done, len(df))

    df = df.copy()
    df["title"] = titles
    # Fallback ke slug jika judul kosong
    mask = df["title"] == ""
    df.loc[mask, "title"] = df.loc[mask, "url"].apply(title_from_slug)
    return df


def filter_geopolitical(df: pd.DataFrame, col: str = "title") -> pd.DataFrame:
    """Filter baris yang relevan dengan topik geopolitik & USD."""
    if col not in df.columns or df.empty:
        return df
    mask = df[col].str.lower().apply(
        lambda t: any(kw in t for kw in GEOPOLITICAL_KEYWORDS)
    )
    filtered = df[mask].copy()
    log.info("Filter geopolitik: %d -> %d artikel relevan", len(df), len(filtered))
    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Scraper judul artikel CNBC via sitemap (2021-2026)"
    )
    parser.add_argument(
        "--mode",
        choices=["url_only", "with_title"],
        default="url_only",
        help="url_only: URL+tanggal dari sitemap (CEPAT) | with_title: ambil judul dari HTML"
    )
    parser.add_argument("--start",     default=DEFAULT_START, help="Tanggal mulai YYYY-MM-DD")
    parser.add_argument("--end",       default=DEFAULT_END,   help="Tanggal selesai YYYY-MM-DD")
    parser.add_argument("--workers",   type=int, default=5,   help="Jumlah thread paralel (default:5)")
    parser.add_argument("--no-filter", action="store_true",   help="Nonaktifkan filter keyword geopolitik")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("SCRAPER CNBC SITEMAP")
    log.info("Rentang : %s s/d %s", args.start, args.end)
    log.info("Mode    : %s", args.mode)
    log.info("=" * 60)

    df = collect_all_urls(args.start, args.end)
    if df.empty:
        log.warning("Tidak ada URL ditemukan dalam rentang tanggal ini.")
        return

    log.info("\nURL terkumpul: %d", len(df))

    if args.mode == "with_title":
        df = add_titles_parallel(df, max_workers=args.workers)
    else:
        log.info("Mode url_only: ekstrak judul dari URL slug...")
        df["title"] = df["url"].apply(title_from_slug)

    if not args.no_filter:
        df = filter_geopolitical(df, col="title")

    if not df.empty:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_tag = "full" if args.mode == "with_title" else "urlonly"
        csv_path = OUTPUT_DIR / ("cnbc_sitemap_" + mode_tag + "_" + ts + ".csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        log.info("\nDisimpan: %s", csv_path)
        log.info("Total artikel: %d", len(df))
        log.info("\nSample:\n%s", df[["date", "title", "url"]].head(10).to_string())
    else:
        log.warning("Tidak ada data setelah filter.")


if __name__ == "__main__":
    main()
