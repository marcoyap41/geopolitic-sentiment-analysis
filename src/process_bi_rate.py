"""
=====================================================================
process_bi_rate.py  -  Bersihkan dan Standardisasi Data Kurs JISDOR BI
=====================================================================
Input : Informasi Kurs Jisdor BI.xlsx  (format mentah dari website BI)
Output: data/cleaned/bi_jisdor_2021_2026.csv

Kolom output:
  tanggal          : YYYY-MM-DD (hari kerja trading)
  kurs_idr_per_usd : Kurs tengah JISDOR (Rp per 1 USD)
  kurs_change      : Selisih kurs vs hari kerja sebelumnya
  kurs_change_pct  : Perubahan persen vs hari kerja sebelumnya

Cara pakai:
    python process_bi_rate.py
    python process_bi_rate.py --input path/ke/file.xlsx
"""

import pandas as pd
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT_DIR    = Path(__file__).parent.parent
INPUT_FILE  = ROOT_DIR / "Informasi Kurs Jisdor BI.xlsx"
OUTPUT_DIR  = ROOT_DIR / "data" / "cleaned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "bi_jisdor_2021_2026.csv"


def load_and_clean(input_path: Path) -> pd.DataFrame:
    """
    Membaca Excel JISDOR BI yang memiliki header baris tidak standar.
    Baris 0-3 adalah metadata/header, data dimulai dari baris 4.
    """
    log.info("Membaca file: %s", input_path)
    df_raw = pd.read_excel(input_path, header=None)
    log.info("  Raw shape: %s", df_raw.shape)

    # Skip baris header (0-3), ambil dari baris 4 ke bawah
    df = df_raw.iloc[4:].copy()
    df.columns = ["no", "tanggal", "kurs_idr_per_usd", "_extra"]
    df = df[["tanggal", "kurs_idr_per_usd"]].copy()

    # Bersihkan dan konversi tipe data
    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    df["kurs_idr_per_usd"] = pd.to_numeric(df["kurs_idr_per_usd"], errors="coerce")

    # Hapus baris dengan nilai null
    before = len(df)
    df = df.dropna(subset=["tanggal", "kurs_idr_per_usd"])
    log.info("  Dropped %d baris null", before - len(df))

    # Format tanggal ke YYYY-MM-DD
    df["tanggal"] = df["tanggal"].dt.strftime("%Y-%m-%d")

    # Sort ascending (terlama dulu)
    df = df.sort_values("tanggal").reset_index(drop=True)

    # Hapus duplikat tanggal (jika ada)
    before = len(df)
    df = df.drop_duplicates(subset=["tanggal"])
    if before != len(df):
        log.warning("  Removed %d duplikat tanggal", before - len(df))

    return df


def add_change_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan kolom perubahan kurs harian."""
    df = df.copy()
    df["kurs_change"]     = df["kurs_idr_per_usd"].diff().round(0)
    df["kurs_change_pct"] = (df["kurs_idr_per_usd"].pct_change() * 100).round(4)
    return df


def main():
    parser = argparse.ArgumentParser(description="Bersihkan data kurs JISDOR BI")
    parser.add_argument("--input", default=str(INPUT_FILE),
                        help="Path ke file Excel JISDOR BI")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        log.error("File tidak ditemukan: %s", input_path)
        return

    df = load_and_clean(input_path)
    df = add_change_columns(df)

    log.info("\nRentang data    : %s s/d %s", df["tanggal"].iloc[0], df["tanggal"].iloc[-1])
    log.info("Total hari kerja: %d", len(df))
    log.info("Kurs min        : Rp %s (tanggal %s)",
             int(df["kurs_idr_per_usd"].min()),
             df.loc[df["kurs_idr_per_usd"].idxmin(), "tanggal"])
    log.info("Kurs maks       : Rp %s (tanggal %s)",
             int(df["kurs_idr_per_usd"].max()),
             df.loc[df["kurs_idr_per_usd"].idxmax(), "tanggal"])

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    log.info("\nDisimpan: %s", OUTPUT_FILE)
    log.info("\nSample:\n%s", df.head(5).to_string())


if __name__ == "__main__":
    main()
