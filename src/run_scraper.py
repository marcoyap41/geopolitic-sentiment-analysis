"""
=====================================================================
run_scraper.py  -  Script Utama: Pipeline Scraping Berita Geopolitik
=====================================================================
Koordinator semua scraper dalam satu entry point.

REKOMENDASI URUTAN PENGGUNAAN:
1. Test dulu koneksi GDELT:
       python run_scraper.py --mode gdelt_test

2. Jika GDELT berhasil, jalankan scraping penuh:
       python run_scraper.py --mode gdelt_full

3. Sambil/setelah GDELT, tambahkan data dari RSS:
       python run_scraper.py --mode rss

4. Jika punya API key NewsAPI, tambahkan:
       python run_scraper.py --mode newsapi --apikey YOUR_KEY

Cara pakai:
    python run_scraper.py --mode gdelt_test
    python run_scraper.py --mode gdelt_full
    python run_scraper.py --mode rss
    python run_scraper.py --mode newsapi --apikey YOUR_KEY_HERE
    python run_scraper.py --mode all --apikey YOUR_KEY_HERE
"""

import subprocess
import sys
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SRC_DIR = Path(__file__).parent


def run_script(script, args=None):
    """Jalankan script Python dan tampilkan output secara real-time."""
    cmd = [sys.executable, str(SRC_DIR / script)]
    if args:
        cmd.extend(args)
    log.info(f"\n{'='*60}")
    log.info(f"Menjalankan: {' '.join(cmd)}")
    log.info(f"{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log.error(f"Script {script} selesai dengan error (code {result.returncode})")
    return result.returncode


def print_banner():
    banner = """
    ================================================================
    SCRAPER BERITA GEOPOLITIK & USD - NLP PROJECT TUGAS 1
    ================================================================
    Sumber Data:
      1. GDELT API   - Data historis 5 tahun (UTAMA)
      2. RSS Feeds   - Reuters, AP News, CNBC (Real-time)
      3. NewsAPI     - Terstruktur (butuh API key gratis)
    ================================================================
    """
    print(banner)


def main():
    print_banner()
    parser = argparse.ArgumentParser(description="Pipeline Scraping Berita Geopolitik")
    parser.add_argument(
        "--mode",
        choices=["gdelt_test", "gdelt_full", "rss", "newsapi", "all"],
        default="gdelt_test",
        help=(
            "Mode scraping:\n"
            "  gdelt_test  : Test koneksi GDELT (1 keyword, 1 bulan) - MULAI DARI SINI\n"
            "  gdelt_full  : Scraping GDELT 5 tahun penuh (~30-60 menit)\n"
            "  rss         : Scraping RSS (Reuters, AP News, CNBC)\n"
            "  newsapi     : Scraping via NewsAPI (butuh API key)\n"
            "  all         : Semua sumber (gdelt_test + rss + newsapi)"
        )
    )
    parser.add_argument("--apikey", default=None,
                        help="API key NewsAPI (hanya untuk mode newsapi/all)")
    parser.add_argument("--start", default="2021-09-01",
                        help="Tanggal mulai untuk GDELT full (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-09-01",
                        help="Tanggal selesai untuk GDELT full (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.mode == "gdelt_test":
        log.info("MODE: Test GDELT (cepat, ~30 detik)")
        run_script("scraper_gdelt.py", ["--mode", "test"])

    elif args.mode == "gdelt_full":
        log.info("MODE: GDELT Full 5 tahun")
        log.info("PERKIRAAN WAKTU: 30-60 menit")
        log.info("Data disimpan secara berkala (checkpoint setiap 50 request)")
        run_script("scraper_gdelt.py", [
            "--mode", "full",
            "--start", args.start,
            "--end", args.end,
            "--delay", "6",
            "--chunk", "90",
        ])

    elif args.mode == "rss":
        log.info("MODE: RSS Feeds (Reuters, AP News, CNBC)")
        run_script("scraper_rss.py")

    elif args.mode == "newsapi":
        if not args.apikey:
            log.error("Harap sertakan --apikey YOUR_KEY untuk mode newsapi")
            log.error("Daftar gratis di: https://newsapi.org/register")
            return
        log.info("MODE: NewsAPI")
        run_script("scraper_newsapi.py", [
            "--apikey", args.apikey,
            "--start", args.start,
            "--end", args.end,
        ])

    elif args.mode == "all":
        log.info("MODE: Semua sumber")
        run_script("scraper_gdelt.py", ["--mode", "test"])
        run_script("scraper_rss.py")
        if args.apikey:
            run_script("scraper_newsapi.py", ["--apikey", args.apikey])
        else:
            log.info("Lewati NewsAPI (tidak ada --apikey)")

    log.info("\nScraping selesai! Cek folder data/raw/ untuk hasilnya.")
    log.info("Langkah berikutnya: jalankan preprocessing_pipeline.py")

if __name__ == "__main__":
    main()
