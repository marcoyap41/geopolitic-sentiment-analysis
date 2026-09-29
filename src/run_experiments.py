"""
Menjalankan semua eksperimen dan menyimpan hasilnya.
Output: results/metrics.csv dan results/predictions_test.csv

Model yang dibandingkan:
  baseline        : majority, persistence, ARIMA, XGBoost (kurs saja)
  model gabungan  : XGBoost (kurs + lexicon), (kurs + TF-IDF), (kurs + lexicon + TF-IDF)
"""
import json

import pandas as pd

from baseline import arima_baseline, majority_baseline, persistence_baseline
from config import ROOT
from dataset import load_dataset
from evaluate import metrics
from model import run_xgb

RES = ROOT / "results"

# nama eksperimen -> set fitur yang dipakai
XGB_EXPERIMENTS = {
    "xgb:kurs":                       ["kurs"],                                  # baseline
    "xgb:kurs+lexicon":                ["kurs", "lexicon"],                       # model gabungan
    "xgb:kurs+tfidf":                  ["kurs", "tfidf"],
    "xgb:kurs+category":               ["kurs", "category"],
    "xgb:kurs+lexicon+tfidf+category": ["kurs", "lexicon", "tfidf", "category"],  # semua fitur NLP
}


def main():
    RES.mkdir(exist_ok=True)
    df, feature_sets = load_dataset()
    masks = {"val": df["split"] == "val", "test": df["split"] == "test"}
    rows, test_preds = [], {}

    def catat(name, family, split, y, pred, note=""):
        rows.append({"model": name, "family": family, "split": split, "n": len(y),
                     **metrics(y, pred), "note": note})

    # --- baseline non-ML: majority, persistence, ARIMA
    for name, pred in {"majority": majority_baseline(df),
                       "persistence": persistence_baseline(df),
                       "arima": arima_baseline(df)}.items():
        for split, m in masks.items():
            catat(name, "baseline", split, df.loc[m, "y"].values, pred[m].values)
        test_preds[name] = pred[masks["test"]].values

    # --- XGBoost: baseline (kurs saja) dan model gabungan (kurs + NLP)
    for name, sets in XGB_EXPERIMENTS.items():
        cols = [c for s in sets for c in feature_sets[s]]
        hp, out = run_xgb(df, cols)
        family = "baseline" if name == "xgb:kurs" else "gabungan"
        for split in ("val", "test"):
            y, pred = out[split]
            catat(name, family, split, y, pred, note=json.dumps(hp))
        test_preds[name] = out["test"][1]
        r = rows[-2:]
        print(f"{name:25s} DA val={r[0]['DA']:.3f}  DA test={r[1]['DA']:.3f}  "
              f"F1 test={r[1]['F1_macro']:.3f}  ({len(cols)} fitur, {hp})")

    pd.DataFrame(rows).to_csv(RES / "metrics.csv", index=False)
    pd.DataFrame({"date": df.loc[masks["test"], "date"].values,
                  "y": df.loc[masks["test"], "y"].values, **test_preds}
                 ).to_csv(RES / "predictions_test.csv", index=False)
    print("Selesai. Hasil di results/")


if __name__ == "__main__":
    main()
