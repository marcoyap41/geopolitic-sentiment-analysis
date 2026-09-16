"""
=====================================================================
filter_negative.py  -  Filtering Akhir menggunakan Negative Keywords
=====================================================================
Script ini menggabungkan data berita bersih dengan hasil scraping 
keypoints. Kemudian membuang artikel false positive berdasarkan 
Negative Keywords dan URL Blacklist.

Aturan Filtering:
1. Jika artikel TIDAK memiliki keypoints -> LOLOS (bypass filter)
2. Jika artikel MEMILIKI keypoints:
   a. URL masuk Blacklist -> BUANG
   b. URL masuk Whitelist -> LOLOS (bypass negative keyword)
   c. Mengandung Negative Keyword -> BUANG
   d. Sisanya -> LOLOS
"""

import pandas as pd
import re
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent
CLEAN_FILE = ROOT_DIR / "data" / "cleaned" / "news_geopolitik_clean.csv"
KEYPOINTS_FILE = ROOT_DIR / "data" / "raw" / "cnbc_keypoints.csv"
FINAL_OUTPUT = ROOT_DIR / "data" / "cleaned" / "news_geopolitik_final.csv"

# ─── 1. DAFTAR NEGATIVE KEYWORDS (False Positives) ───────────────────────────
EXCLUDE_EDITORIAL = [
    "Jim Cramer's top", "Jim Cramer's Mad Money", "Jim Cramer says buy",
    "Jim Cramer says sell", "Jim Cramer names", "Jim Cramer reveals",
    "things to watch in the stock market", "watch list for",
    "watch live", "speak live", "watch the full interview", "tune in live",
    "Fast Money picks", "Options Action", "Halftime Report",
    "Power Lunch picks", "Mad Money recap", "Closing Bell recap",
]

EXCLUDE_ENTERTAINMENT = [
    "Taylor Swift", "Eras Tour", "concert film", "concert tour",
    "album release", "Grammy Award", "Grammy nomination",
    "Academy Award", "Oscar nomination", "Emmy Award",
    "box office", "movie premiere", "film premiere", "streaming series",
    "Netflix show", "Disney+ series", "Shark Tank deal",
    "Shark Tank founders", "Shark Tank pitch", "got a deal on Shark Tank",
]

EXCLUDE_SPORTS = [
    "Super Bowl", "NFL draft", "NBA Finals", "NBA playoffs",
    "World Cup final", "Olympics medal", "sports betting odds",
    "fantasy football", "quarterback", "touchdown", "slam dunk",
    "playoff roster", "sports drink", "energy drink war",
]

EXCLUDE_LIFESTYLE = [
    "weight loss", "diet plan", "calorie", "workout routine",
    "fitness challenge", "skincare routine", "anti-aging",
    "wellness tips", "mental health tips", "sleep tips",
    "recipe", "cooking tips", "restaurant review",
]

EXCLUDE_OTHER = [
    "interior design tips", "renovation tips", "DIY home",
    "resume tips", "job interview tips", "how to negotiate salary",
    "side hustle ideas", "passive income ideas",
    "credit card rewards", "best credit card", "shopping deals",
    "best laptop", "best smartphone", "iPhone review",
    "fool's gold", "turf war", "civil war reenact",
]

ALL_NEGATIVE = (
    EXCLUDE_EDITORIAL + EXCLUDE_ENTERTAINMENT + EXCLUDE_SPORTS +
    EXCLUDE_LIFESTYLE + EXCLUDE_OTHER
)

# Regex Matcher (Gunakan word boundary jika memungkinkan, tapi untuk phrase biarkan case-insensitive)
NEG_REGEX = re.compile("|".join(re.escape(kw) for kw in ALL_NEGATIVE), re.IGNORECASE)

# ─── 2. URL BLACKLIST & WHITELIST ──────────────────────────────────────────
# URL yang PASTI BUKAN geopolitik
URL_BLACKLIST = re.compile(
    r"cnbc\.com/(entertainment|lifestyle|travel|sports|make-it|select|personal-finance|health-and-science|real-estate)/", 
    re.IGNORECASE
)

# URL yang PASTI geopolitik/ekonomi makro (kebal terhadap negative keyword)
URL_WHITELIST = re.compile(
    r"cnbc\.com/(world-economy|politics|global-investor|energy|defense)/", 
    re.IGNORECASE
)

# ─── FUNGSI UTAMA ──────────────────────────────────────────────────────────
def main():
    if not CLEAN_FILE.exists():
        log.error("File berita bersih tidak ditemukan: %s", CLEAN_FILE)
        return
        
    log.info("Membaca data artikel: %s", CLEAN_FILE.name)
    df_clean = pd.read_csv(CLEAN_FILE, encoding="utf-8-sig")
    
    if KEYPOINTS_FILE.exists():
        log.info("Membaca data keypoints: %s", KEYPOINTS_FILE.name)
        df_kp = pd.read_csv(KEYPOINTS_FILE, encoding="utf-8-sig")
        # Hanya ambil kolom yang dibutuhkan
        df_kp = df_kp[['url', 'keypoints', 'status']]
        
        # Gabungkan data
        df = pd.merge(df_clean, df_kp, on='url', how='left')
    else:
        log.warning("File keypoints tidak ditemukan! Anggap semua tidak punya keypoints.")
        df = df_clean.copy()
        df['keypoints'] = None
        df['status'] = 'no_keypoints'

    # Isi status kosong untuk data yang belum ter-scrape
    df['status'] = df['status'].fillna('no_keypoints')
    df['keypoints'] = df['keypoints'].fillna('')

    log.info("Total artikel awal: %d", len(df))

    # --- PROSES FILTERING ---
    # 1. Identifikasi URL Blacklist & Whitelist (Berlaku untuk SEMUA artikel)
    mask_blacklist = df['url'].str.contains(URL_BLACKLIST, na=False)
    mask_whitelist = df['url'].str.contains(URL_WHITELIST, na=False)

    # 2. Identifikasi mana yang Punya Keypoints
    mask_has_kp = (df['status'] == 'ok') & (df['keypoints'].str.strip() != '')

    # 3. Identifikasi Negative Keywords pada Judul + Keypoints
    combined_text = df['title'] + " " + df['keypoints']
    mask_negative = combined_text.str.contains(NEG_REGEX, na=False)

    # --- LOGIKA KEPUTUSAN ---
    # Syarat DIBUANG: 
    # A. URL masuk Blacklist (Berlaku GLOBAL baik punya keypoint atau tidak)
    # ATAU
    # B. Punya Keypoint AND Punya Negative Keyword AND URL BUKAN Whitelist
    mask_drop = mask_blacklist | (mask_has_kp & mask_negative & ~mask_whitelist)

    # Statistik untuk log
    dropped_by_blacklist = mask_blacklist.sum()
    dropped_by_negative = (mask_has_kp & mask_negative & ~mask_whitelist & ~mask_blacklist).sum()
    kept_by_whitelist = (mask_has_kp & mask_negative & mask_whitelist).sum()

    log.info("--- Statistik Filtering ---")
    log.info("Artikel DENGAN keypoints       : %d", mask_has_kp.sum())
    log.info("Artikel TANPA keypoints        : %d (Tidak dicek neg keyword)", (~mask_has_kp).sum())
    log.info("Dibuang karena URL Blacklist   : %d (Berlaku Global)", dropped_by_blacklist)
    log.info("Dibuang karena Negative Keyword: %d (Khusus ber-Keypoint)", dropped_by_negative)
    log.info("Diselamatkan oleh Whitelist    : %d (Punya neg keyword tapi URL aman)", kept_by_whitelist)
    
    # Filter DataFrame akhir
    df_final = df[~mask_drop].copy()
    
    log.info("Total artikel akhir (Final)    : %d (Berkurang %d)", len(df_final), mask_drop.sum())

    # Simpan
    df_final.to_csv(FINAL_OUTPUT, index=False, encoding="utf-8-sig")
    log.info("Disimpan ke: %s", FINAL_OUTPUT)

if __name__ == "__main__":
    main()
