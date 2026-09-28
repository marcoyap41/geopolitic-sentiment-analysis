"""Ekspor dataset akhir per split (train/val/test) ke data/splits/ dan salinan ringkas di data/."""
import pandas as pd

from config import ARTICLES_PREPROCESSED, ROOT, SPLITS
from dataset import load_dataset


def main():
    SPLITS.mkdir(parents=True, exist_ok=True)
    df, _ = load_dataset()
    id_cols = ["date", "kurs", "ret", "ret_pct", "y", "split"]
    for s in ("train", "val", "test"):
        df[df["split"] == s].to_csv(ROOT / "data" / f"{s}.csv", index=False)      # matriks fitur harian
    arts = pd.read_csv(ARTICLES_PREPROCESSED, encoding="utf-8-sig", low_memory=False)
    for s in ("train", "val", "test"):                                             # level artikel
        arts[arts["split"] == s].to_csv(SPLITS / f"{s}_articles.csv", index=False, encoding="utf-8-sig")
    print(df.groupby("split").size().to_dict())


if __name__ == "__main__":
    main()
