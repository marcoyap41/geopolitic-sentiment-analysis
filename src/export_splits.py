"""Ekspor dataset akhir per split (train/val/test) ke folder data/."""
import pandas as pd

from config import ARTICLES_PREPROCESSED, ROOT, SPLITS
from dataset import load_dataset


def main():
    SPLITS.mkdir(parents=True, exist_ok=True)
    df, _ = load_dataset()
    for s in ("train", "val", "test"):                        # matriks fitur harian
        df[df["split"] == s].to_csv(ROOT / "data" / f"{s}.csv", index=False)
    if ARTICLES_PREPROCESSED.exists():                        # data level artikel (dibuat preprocessing.py)
        arts = pd.read_csv(ARTICLES_PREPROCESSED, encoding="utf-8-sig", low_memory=False)
        for s in ("train", "val", "test"):
            arts[arts["split"] == s].to_csv(SPLITS / f"{s}_articles.csv", index=False, encoding="utf-8-sig")
    print(df.groupby("split").size().to_dict())


if __name__ == "__main__":
    main()
