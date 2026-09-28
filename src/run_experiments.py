"""
Orkestrasi eksperimen: baseline + model gabungan + ablation.
Output: results/metrics.csv, results/predictions_test.csv, results/confusion_*.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline import arima_baseline, majority_baseline, persistence_baseline
from combined_model import run_experiment
from config import ROOT
from dataset import load_dataset
from evaluate import (binom_vs_majority, bootstrap_da_ci, cm_table, mcnemar_p,
                      metrics)

RES = ROOT / "results"

# (nama eksperimen, fitur yang dipakai selain kurs; None = hanya NLP saja)
EXPERIMENTS = {
    "kurs":               ["kurs"],
    "kurs+volcat":        ["kurs", "volcat"],
    "kurs+lex_title":     ["kurs", "lex_t"],
    "kurs+lex_title_kp":  ["kurs", "lex_tk"],
    "kurs+tfidf_title":   ["kurs", "tfidf_t"],
    "kurs+tfidf_title_kp": ["kurs", "tfidf_tk"],
    "kurs+bow_title":     ["kurs", "bow_t"],
    "kurs+lex+tfidf":     ["kurs", "lex_t", "tfidf_t"],
    "nlp_only(lex_title)": ["lex_t"],
}


def main():
    RES.mkdir(exist_ok=True)
    df, fs = load_dataset()
    val, test = df["split"] == "val", df["split"] == "test"
    rows, preds = [], {}
    maj_rate = {"val": df.loc[val, "y"].mean(), "test": df.loc[test, "y"].mean()}

    def log(name, family, y, p, split, proba=None, note=""):
        m = metrics(y, p, proba)
        m.update(model=name, family=family, split=split, n=len(y), share_pred1=float(np.mean(p)), note=note)
        if split == "test":
            lo, hi = bootstrap_da_ci(y, p)
            m.update(DA_lo=lo, DA_hi=hi, p_vs_majority=binom_vs_majority(y, p, maj_rate["test"]))
        rows.append(m)

    # ---- baseline non-model & ARIMA
    base_preds = {"majority": majority_baseline(df), "persistence": persistence_baseline(df)}
    arima_pred, order = arima_baseline(df)
    base_preds["arima"] = arima_pred
    for name, s in base_preds.items():
        for split, mask in (("val", val), ("test", test)):
            log(name, "baseline", df.loc[mask, "y"].values, s[mask].values, split,
                note=f"order={order}" if name == "arima" else "")
        preds[name] = s[test].values

    # ---- model ML (LogReg, XGB) untuk tiap set fitur
    for exp, sets in EXPERIMENTS.items():
        cols = [c for s in sets for c in fs[s]]
        family = "baseline_ml" if exp == "kurs" else "combined"
        for kind in ("logreg", "xgb"):
            r = run_experiment(df, cols, kind)
            name = f"{kind}:{exp}"
            for split in ("val", "test"):
                y, p, pr = r[split]
                log(name, family, y, p, split, pr, note=json.dumps(r["hp"]))
            preds[name] = r["test"][1]
            print(f"{name:35s} val DA={rows[-2]['DA']:.3f}  test DA={rows[-1]['DA']:.3f}  hp={r['hp']}")

    res = pd.DataFrame(rows)
    res.to_csv(RES / "metrics.csv", index=False)
    y_test = df.loc[test, "y"].values
    pd.DataFrame({"date": df.loc[test, "date"].values, "y": y_test, **preds}).to_csv(
        RES / "predictions_test.csv", index=False)

    # McNemar: tiap model gabungan vs baseline ML kurs-saja dengan learner sama
    mc = []
    for kind in ("logreg", "xgb"):
        for exp in EXPERIMENTS:
            if exp == "kurs":
                continue
            mc.append({"model": f"{kind}:{exp}", "vs": f"{kind}:kurs",
                       "p_mcnemar": mcnemar_p(y_test, preds[f"{kind}:{exp}"], preds[f"{kind}:kurs"])})
    pd.DataFrame(mc).to_csv(RES / "mcnemar_vs_kurs_only.csv", index=False)
    print(f"\nMajority rate val={maj_rate['val']:.3f} test={maj_rate['test']:.3f}")


if __name__ == "__main__":
    main()
