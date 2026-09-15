"""
=====================================================================
fetch_real_titles.py  -  Ambil Judul Asli Artikel dari HTML (bukan slug)
=====================================================================
Baca langsung dari raw CSV hasil scraper_cnbc_sitemap.py --mode url_only
(kolom: url, date, title). Mendukung pembagian kerja ke beberapa mesin
sekaligus lewat --chunk-id / --num-chunks, supaya bisa dijalankan
paralel di beberapa laptop tanpa tabrakan/duplikat.

Cara pakai (SATU laptop, semua data):
    python fetch_real_titles.py --input ../data/raw/cnbc_sitemap_urlonly_20260915_114841.csv

Cara pakai (4 LAPTOP paralel, tiap laptop 1/4 data):
    # Laptop 1:
    python fetch_real_titles.py --input ../data/raw/cnbc_sitemap_urlonly_20260915_114841.csv --num-chunks 4 --chunk-id 0
    # Laptop 2:
    python fetch_real_titles.py --input ../data/raw/cnbc_sitemap_urlonly_20260915_114841.csv --num-chunks 4 --chunk-id 1
    # Laptop 3:
    python fetch_real_titles.py --input ../data/raw/cnbc_sitemap_urlonly_20260915_114841.csv --num-chunks 4 --chunk-id 2
    # Laptop 4:
    python fetch_real_titles.py --input ../data/raw/cnbc_sitemap_urlonly_20260915_114841.csv --num-chunks 4 --chunk-id 3

Tiap laptop harus pakai FILE INPUT YANG SAMA PERSIS (jangan dipotong manual
duluan) -- script ini yang membagi berdasarkan urutan URL yang sudah
di-sort, supaya pembagiannya konsisten dan tidak ada baris yang
kelewat/dobel antar laptop.

Lanjutkan proses yang terhenti di tengah jalan (per chunk):
    python fetch_real_titles.py --input ... --num-chunks 4 --chunk-id 0 --resume

Setelah SEMUA chunk selesai, gabungkan hasilnya:
    python fetch_real_titles.py --merge --num-chunks 4 --input ../data/raw/cnbc_sitemap_urlonly_20260915_114841.csv
"""

import argparse
import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = ROOT_DIR / "data" / "raw"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def chunk_paths(num_chunks: int, chunk_id: int, sample: int = None):
    tag = f"chunk{chunk_id}of{num_chunks}"
    if sample:
        tag += f".sample{sample}"
    checkpoint = OUTPUT_DIR / f"fetch_real_titles.{tag}.checkpoint.csv"
    output = OUTPUT_DIR / f"cnbc_real_titles.{tag}.csv"
    return checkpoint, output


def get_chunk(df: pd.DataFrame, num_chunks: int, chunk_id: int) -> pd.DataFrame:
    """Bagi dataframe jadi num_chunks bagian yang konsisten (urut berdasarkan url),
    supaya laptop manapun yang jalanin chunk-id yang sama selalu dapat baris yang sama."""
    df_sorted = df.sort_values("url").reset_index(drop=True)
    return df_sorted.iloc[chunk_id::num_chunks].reset_index(drop=True)


def load_checkpoint(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["url", "title_real"])


def fetch_title(url: str, session: requests.Session, retries: int = 2) -> str:
    """Ambil judul asli dari meta og:title / <title>, dengan retry ringan."""
    for attempt in range(retries + 1):
        try:
            r = session.get(url, headers=HEADERS, timeout=10)
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))  # backoff kalau kena rate-limit
                continue
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
                t = re.sub(r"\s*[-|]\s*(CNBC|Reuters|AP News|Bloomberg|FT|WSJ|BBC).*$",
                            "", t, flags=re.IGNORECASE)
                return t.strip()
            return ""
        except requests.RequestException:
            time.sleep(1)
    return ""


def run_chunk(args):
    df = pd.read_csv(args.input)
    df = df.dropna(subset=["url"]).drop_duplicates(subset=["url"])
    log.info("Total baris di file input: %d", len(df))

    my_chunk = get_chunk(df, args.num_chunks, args.chunk_id)
    log.info("Chunk %d/%d -> %d baris jadi tanggung jawab laptop ini", args.chunk_id, args.num_chunks, len(my_chunk))

    if args.sample:
        n = min(args.sample, len(my_chunk))
        my_chunk = my_chunk.sample(n=n, random_state=42).reset_index(drop=True)
        log.info("Mode SAMPLE aktif: ambil %d baris acak dari chunk ini (buat tes dulu)", n)

    checkpoint_path, output_path = chunk_paths(args.num_chunks, args.chunk_id, args.sample)

    checkpoint = load_checkpoint(checkpoint_path) if args.resume else pd.DataFrame(columns=["url", "title_real"])
    done_map = dict(zip(checkpoint["url"], checkpoint["title_real"]))

    remaining = my_chunk[~my_chunk["url"].isin(done_map.keys())]
    log.info("Sudah selesai sebelumnya (chunk ini): %d | sisa: %d", len(done_map), len(remaining))

    session = requests.Session()
    new_results = []

    if not remaining.empty:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            future_to_url = {ex.submit(fetch_title, url, session): url for url in remaining["url"]}
            n_done = 0
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                real_title = future.result()
                new_results.append({"url": url, "title_real": real_title})
                n_done += 1
                if n_done % 100 == 0:
                    log.info("Progress chunk %d: %d/%d", args.chunk_id, n_done, len(remaining))
                    pd.concat([checkpoint, pd.DataFrame(new_results)], ignore_index=True) \
                        .to_csv(checkpoint_path, index=False, encoding="utf-8-sig")

    all_titles = pd.concat([checkpoint, pd.DataFrame(new_results)], ignore_index=True)
    all_titles.to_csv(checkpoint_path, index=False, encoding="utf-8-sig")

    merged = my_chunk.merge(all_titles, on="url", how="left")
    merged = merged.rename(columns={"title": "title_slug"})
    merged["title"] = merged["title_real"].fillna("").replace("", pd.NA)
    merged["title"] = merged["title"].fillna(merged["title_slug"])
    merged = merged.drop(columns=["title_real"])

    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    log.info("Chunk %d selesai. Disimpan ke: %s (%d baris)", args.chunk_id, output_path, len(merged))


def run_merge(args):
    """Gabungkan semua file cnbc_real_titles.chunk*of{num_chunks}[.sampleN].csv jadi satu."""
    frames = []
    for i in range(args.num_chunks):
        _, output_path = chunk_paths(args.num_chunks, i, args.sample)
        if not output_path.exists():
            log.warning("Chunk %d belum ada hasilnya (%s) -- pastikan semua laptop sudah selesai.", i, output_path)
            continue
        frames.append(pd.read_csv(output_path))

    if not frames:
        log.error("Tidak ada chunk yang ditemukan untuk digabung.")
        return

    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["url"])
    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = f"sample{args.sample}_" if args.sample else ""
    final_path = OUTPUT_DIR / f"cnbc_real_titles_merged_{tag}{ts}.csv"
    combined.to_csv(final_path, index=False, encoding="utf-8-sig")
    log.info("Gabungan selesai: %d baris -> %s", len(combined), final_path)


def main():
    parser = argparse.ArgumentParser(description="Fetch judul asli CNBC, mendukung pembagian kerja multi-laptop")
    parser.add_argument("--input", required=True, help="Path ke CSV raw (url_only) hasil scraper_cnbc_sitemap.py")
    parser.add_argument("--workers", type=int, default=8, help="Jumlah thread paralel per laptop (default: 8)")
    parser.add_argument("--resume", action="store_true", help="Lanjutkan chunk ini dari checkpoint sebelumnya")
    parser.add_argument("--num-chunks", type=int, default=1, help="Total jumlah laptop/bagian (default: 1, tidak dibagi)")
    parser.add_argument("--chunk-id", type=int, default=0, help="Nomor bagian untuk laptop ini, mulai dari 0 (0, 1, 2, 3 untuk 4 laptop)")
    parser.add_argument("--merge", action="store_true", help="Mode gabung: satukan semua hasil chunk jadi 1 file (jalankan setelah semua laptop selesai)")
    parser.add_argument("--sample", type=int, default=None, help="Mode tes: cuma ambil N baris acak dari chunk ini (misal 1000), tidak proses semua dulu")
    args = parser.parse_args()

    if args.chunk_id >= args.num_chunks:
        parser.error(f"--chunk-id ({args.chunk_id}) harus lebih kecil dari --num-chunks ({args.num_chunks})")

    if args.merge:
        run_merge(args)
    else:
        run_chunk(args)


if __name__ == "__main__":
    main()
