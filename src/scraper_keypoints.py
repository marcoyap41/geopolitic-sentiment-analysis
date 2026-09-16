"""
=====================================================================
scraper_keypoints.py  -  Scrape Key Points dari artikel CNBC
=====================================================================
Membaca URL dari data/cleaned/news_geopolitik_clean.csv,
mengambil keypoints tiap artikel, lalu menyimpan hasilnya.

Fitur:
  - Async scraping (aiohttp) dengan N concurrent workers
  - Checkpoint otomatis setiap CHECKPOINT_EVERY artikel
    -> resume otomatis jika script dihentikan/crash
  - Retry dengan exponential backoff untuk error transient
  - Rate limiting adaptif (delay acak antar request)
  - Rotasi User-Agent untuk hindari blokir
  - Progress bar real-time (tqdm)

Output:
  data/raw/cnbc_keypoints.csv
    kolom: url | keypoints | n_keypoints | status | scraped_at

Checkpoint:
  data/raw/keypoints_checkpoint.parquet
    -> disimpan setiap CHECKPOINT_EVERY artikel

Cara pakai:
  python scraper_keypoints.py                    # scrape semua
  python scraper_keypoints.py --workers 15       # lebih agresif
  python scraper_keypoints.py --limit 500        # test 500 URL saja
  python scraper_keypoints.py --reset            # mulai dari awal (hapus checkpoint)
"""

import asyncio
import aiohttp
import pandas as pd
import logging
import random
import argparse
import sys
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm

# ── Konfigurasi ───────────────────────────────────────────────────────────────
ROOT_DIR         = Path(__file__).parent.parent
INPUT_FILE       = ROOT_DIR / "data" / "cleaned" / "news_geopolitik_clean.csv"
CHECKPOINT_FILE  = ROOT_DIR / "data" / "raw"    / "keypoints_checkpoint.parquet"
OUTPUT_FILE      = ROOT_DIR / "data" / "raw"    / "cnbc_keypoints.csv"

MAX_WORKERS      = 10      # concurrent HTTP requests
REQUEST_TIMEOUT  = 20      # detik per request
MAX_RETRIES      = 3       # max percobaan ulang per URL
CHECKPOINT_EVERY = 300     # simpan checkpoint setiap N artikel
DELAY_MIN        = 0.4     # delay minimum antar request (detik)
DELAY_MAX        = 1.2     # delay maksimum antar request (detik)

# CSS selectors CNBC keypoints (urutan prioritas berdasarkan probe)
KEYPOINTS_SELECTORS = [
    "div.RenderKeyPoints-list li",          # paling umum (terverifikasi)
    "div.RenderKeyPoints-keyPoints li",
    ".KeyPoints-list li",
    "[data-testid='KeyPoints'] li",
    "[class*='keyPoint'] li",
    "[class*='KeyPoint'] li",
]

# Rotasi User-Agent agar tidak diblokir
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT_DIR / "data" / "raw" / "keypoints_scraper.log",
                            encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ── Fungsi Parsing ────────────────────────────────────────────────────────────

def extract_keypoints(html: str) -> list[str]:
    """
    Parse HTML artikel CNBC dan kembalikan list keypoints.
    Coba semua selector secara berurutan.
    """
    soup = BeautifulSoup(html, "html.parser")
    for selector in KEYPOINTS_SELECTORS:
        items = soup.select(selector)
        if items:
            kps = [li.get_text(separator=" ", strip=True) for li in items]
            kps = [kp for kp in kps if len(kp) > 10]  # buang noise pendek
            if kps:
                return kps
    return []


# ── Fungsi Scraping Satu URL ──────────────────────────────────────────────────

async def fetch_keypoints(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    url: str,
) -> dict:
    """
    Fetch satu URL dan ekstrak keypoints.
    Retry dengan exponential backoff untuk error 429/5xx.
    """
    result = {
        "url":         url,
        "keypoints":   "",
        "n_keypoints": 0,
        "status":      "error",
        "scraped_at":  datetime.now().isoformat(timespec="seconds"),
    }

    headers = {
        "User-Agent":      random.choice(USER_AGENTS),
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Cache-Control":   "no-cache",
        "Referer":         "https://www.cnbc.com/",
    }

    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            # Delay acak sebelum request
            await asyncio.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
            try:
                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
                    allow_redirects=True,
                    ssl=False,
                ) as resp:

                    if resp.status == 200:
                        html = await resp.text(errors="replace")
                        kps  = extract_keypoints(html)
                        result["keypoints"]   = " | ".join(kps)
                        result["n_keypoints"] = len(kps)
                        result["status"]      = "ok" if kps else "no_keypoints"
                        return result

                    elif resp.status == 404:
                        result["status"] = "error_404"
                        return result

                    elif resp.status == 429:
                        # Rate limited - tunggu lebih lama
                        wait = 5 * attempt
                        log.warning("Rate limited (%s), tunggu %ds...", url[:60], wait)
                        await asyncio.sleep(wait)

                    elif resp.status in (503, 502, 500):
                        # Server error - retry
                        wait = 2 ** attempt
                        await asyncio.sleep(wait)

                    else:
                        result["status"] = f"error_{resp.status}"
                        return result

            except asyncio.TimeoutError:
                result["status"] = "error_timeout"
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(2 ** attempt)

            except aiohttp.ClientError as e:
                result["status"] = f"error_client"
                log.debug("Client error %s: %s", url[:60], e)
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(2 ** attempt)

            except Exception as e:
                result["status"] = "error_other"
                log.debug("Unexpected error %s: %s", url[:60], e)
                break

        # Semua retry habis
        if result["status"] == "error":
            result["status"] = "error_max_retry"
        return result


# ── Checkpoint ────────────────────────────────────────────────────────────────

def load_checkpoint() -> set[str]:
    """Baca checkpoint dan kembalikan set URL yang sudah di-scrape."""
    if CHECKPOINT_FILE.exists():
        try:
            df = pd.read_parquet(CHECKPOINT_FILE)
            done = set(df["url"].tolist())
            log.info("Checkpoint ditemukan: %d URL sudah diproses", len(done))
            return done
        except Exception as e:
            log.warning("Gagal baca checkpoint: %s. Mulai dari awal.", e)
    return set()


def save_checkpoint(results: list[dict]) -> None:
    """Simpan semua hasil ke file checkpoint (parquet)."""
    if not results:
        return
    df = pd.DataFrame(results)
    df.to_parquet(CHECKPOINT_FILE, index=False)
    log.info("  Checkpoint disimpan: %d baris -> %s", len(df), CHECKPOINT_FILE.name)


# ── Pipeline Utama ────────────────────────────────────────────────────────────

async def run_scraper(urls: list[str], max_workers: int) -> list[dict]:
    """
    Scrape semua URL secara async dengan max_workers concurrent connections.
    Simpan checkpoint setiap CHECKPOINT_EVERY artikel.
    """
    semaphore = asyncio.Semaphore(max_workers)
    connector = aiohttp.TCPConnector(
        limit=max_workers + 5,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
    )

    all_results: list[dict] = []
    # Load hasil checkpoint yang sudah ada (bisa berupa existing results)
    if CHECKPOINT_FILE.exists():
        try:
            existing = pd.read_parquet(CHECKPOINT_FILE).to_dict("records")
            all_results.extend(existing)
        except Exception:
            pass

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            fetch_keypoints(session, semaphore, url)
            for url in urls
        ]

        batch_buffer: list[dict] = []
        completed = 0

        with tqdm(total=len(urls), desc="Scraping keypoints", unit="artikel",
                  ncols=90, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:

            for coro in asyncio.as_completed(tasks):
                result = await coro
                all_results.append(result)
                batch_buffer.append(result)
                completed += 1

                pbar.update(1)
                pbar.set_postfix({
                    "ok":      sum(1 for r in batch_buffer if r["status"] == "ok"),
                    "no_kp":   sum(1 for r in batch_buffer if r["status"] == "no_keypoints"),
                    "err":     sum(1 for r in batch_buffer if r["status"].startswith("error")),
                })

                # Simpan checkpoint setiap CHECKPOINT_EVERY artikel
                if completed % CHECKPOINT_EVERY == 0:
                    save_checkpoint(all_results)
                    ok_count  = sum(1 for r in all_results if r["status"] == "ok")
                    log.info(
                        "Progress: %d/%d selesai | %d dengan keypoints (%.1f%%)",
                        completed, len(urls), ok_count,
                        ok_count / completed * 100,
                    )

    return all_results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scrape CNBC keypoints dari artikel geopolitik"
    )
    parser.add_argument(
        "--workers", type=int, default=MAX_WORKERS,
        help=f"Jumlah concurrent workers (default: {MAX_WORKERS})",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Batasi jumlah URL untuk testing (default: semua)",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Hapus checkpoint dan mulai dari awal",
    )
    parser.add_argument(
        "--input", type=str, default=str(INPUT_FILE),
        help="Path file CSV input (default: data/cleaned/news_geopolitik_clean.csv)",
    )
    args = parser.parse_args()

    # Reset checkpoint jika diminta
    if args.reset and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        log.info("Checkpoint dihapus. Mulai dari awal.")

    # Baca file input
    input_path = Path(args.input)
    if not input_path.exists():
        log.error("File input tidak ditemukan: %s", input_path)
        return

    log.info("Membaca URL dari: %s", input_path.name)
    df_input = pd.read_csv(input_path, encoding="utf-8-sig")
    all_urls = df_input["url"].dropna().unique().tolist()
    log.info("Total URL unik: %d", len(all_urls))

    # Resume: skip URL yang sudah ada di checkpoint
    done_urls = load_checkpoint()
    remaining = [u for u in all_urls if u not in done_urls]
    log.info("URL yang perlu diproses: %d  (sudah selesai: %d)", len(remaining), len(done_urls))

    if args.limit:
        remaining = remaining[: args.limit]
        log.info("Mode test: dibatasi %d URL", args.limit)

    if not remaining:
        log.info("Semua URL sudah diproses! Langsung ke tahap ekspor.")
    else:
        log.info(
            "Mulai scraping %d URL dengan %d workers...",
            len(remaining), args.workers,
        )
        log.info("Estimasi waktu: ~%.0f menit (asumsi ~0.8 detik/artikel, %d workers)",
                 len(remaining) * 0.8 / args.workers / 60, args.workers)

        results = asyncio.run(run_scraper(remaining, args.workers))

        # Simpan checkpoint akhir
        save_checkpoint(results)

    # ── Ekspor hasil akhir ────────────────────────────────────────────────────
    if CHECKPOINT_FILE.exists():
        df_out = pd.read_parquet(CHECKPOINT_FILE)

        # Statistik
        log.info("\n=== HASIL SCRAPING ===")
        log.info("Total diproses : %d", len(df_out))
        status_dist = df_out["status"].value_counts()
        for status, cnt in status_dist.items():
            pct = cnt / len(df_out) * 100
            log.info("  %-20s: %d (%.1f%%)", status, cnt, pct)

        ok = df_out[df_out["status"] == "ok"]
        if len(ok):
            log.info("\nArtikel DENGAN keypoints: %d", len(ok))
            log.info("Rata-rata jumlah keypoints: %.1f", ok["n_keypoints"].mean())
            log.info("Sample keypoints:")
            for _, row in ok.head(3).iterrows():
                log.info("  [%s]", row["url"].split("/")[-1][:50])
                for kp in row["keypoints"].split(" | ")[:2]:
                    log.info("    • %s", kp[:100])

        # Simpan output CSV final
        df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        log.info("\nOutput disimpan: %s", OUTPUT_FILE)
    else:
        log.warning("Tidak ada hasil untuk diekspor.")


if __name__ == "__main__":
    main()
