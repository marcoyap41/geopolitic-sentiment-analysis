"""
Langkah 0 - Formulasi target & split kronologis.

Formulasi (anti-leakage):
  * Unit prediksi = hari kerja target T (hari kurs JISDOR BI).
  * Target y_T = 1 jika kurs_T > kurs_{T-1} (IDR melemah), selain itu 0.
  * Berita berikut dipakai untuk T: artikel dengan tanggal terbit d yang
    memenuhi  T = hari kerja PERTAMA yang STRICTLY setelah d.
    -> berita Jumat/Sabtu/Minggu -> Senin; berita libur -> hari kerja berikut.
    Alasan: tanggal CNBC memakai waktu AS, sehingga berita tanggal d baru
    tercermin di pasar Asia pada hari kerja berikutnya; tanpa jam terbit,
    pergeseran 1 hari ini adalah pilihan paling aman.
  * Fitur kurs untuk T hanya memakai data <= T-1 (lag), dibuat di baseline.py.
"""
import numpy as np
import pandas as pd

from config import (ARTICLES_FILE, ARTICLES_TARGETED, DAILY_BASE, KURS_FILE,
                    PROCESSED, SPLIT_FRACS, SPLITS)


def build_daily_base(kurs_file=KURS_FILE) -> pd.DataFrame:
    """Kurs harian + target biner. Baris pertama (tanpa return) dibuang."""
    k = pd.read_csv(kurs_file, encoding="utf-8-sig")
    k = k.rename(columns={"tanggal": "date", "kurs_idr_per_usd": "kurs"})
    k["date"] = pd.to_datetime(k["date"])
    k = k.sort_values("date").reset_index(drop=True)
    k["ret"] = k["kurs"].diff()                      # perubahan absolut T vs T-1
    k["ret_pct"] = k["kurs"].pct_change() * 100      # perubahan persen
    k = k.dropna(subset=["ret"]).reset_index(drop=True)
    k["y"] = (k["ret"] > 0).astype(int)              # 1 = IDR melemah (kurs naik)
    return k[["date", "kurs", "ret", "ret_pct", "y"]]


def assign_target_date(articles: pd.DataFrame, trading_days: pd.Series) -> pd.DataFrame:
    """Pasangkan tiap artikel ke hari kerja pertama strictly setelah tanggal terbitnya."""
    td = np.sort(pd.to_datetime(trading_days).values)
    d = pd.to_datetime(articles["date"], errors="coerce")
    pos = np.searchsorted(td, d.values, side="right")   # indeks hari kerja pertama > d
    valid = d.notna().values & (pos < len(td))
    out = articles.loc[valid].copy()
    out["pub_date"] = d[valid].values
    out["target_date"] = td[pos[valid]]
    return out


def chronological_split(dates: pd.Series, fracs=SPLIT_FRACS) -> pd.Series:
    """Label 'train'/'val'/'test' berdasarkan urutan waktu (tanpa shuffle)."""
    n = len(dates)
    n_tr, n_va = int(n * fracs[0]), int(n * fracs[1])
    lab = np.array(["train"] * n_tr + ["val"] * n_va + ["test"] * (n - n_tr - n_va))
    return pd.Series(lab, index=dates.index)


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    SPLITS.mkdir(parents=True, exist_ok=True)

    base = build_daily_base()
    arts = pd.read_csv(ARTICLES_FILE, encoding="utf-8-sig", low_memory=False)
    n0 = len(arts)
    arts = arts.drop_duplicates(subset="url")
    arts = assign_target_date(arts, base["date"])

    # Hari target tanpa berita tidak bisa dipakai (hanya hari pertama)
    has_news = base["date"].isin(arts["target_date"].unique())
    base = base.loc[has_news].reset_index(drop=True)
    arts = arts[arts["target_date"].isin(base["date"])]

    base["split"] = chronological_split(base["date"])
    arts = arts.merge(base[["date", "split"]], left_on="target_date", right_on="date",
                      how="left", suffixes=("", "_t")).drop(columns="date_t")
    base.to_csv(DAILY_BASE, index=False)
    arts.to_csv(ARTICLES_TARGETED, index=False, encoding="utf-8-sig")

    print(f"Artikel: {n0} -> {len(arts)} setelah dedup & penautan target_date")
    print(f"Hari target: {len(base)}")
    for s, g in base.groupby("split", sort=False):
        print(f"  {s:5s} {g['date'].min().date()} .. {g['date'].max().date()} "
              f"n={len(g):4d}  P(y=1)={g['y'].mean():.3f}")


if __name__ == "__main__":
    main()
