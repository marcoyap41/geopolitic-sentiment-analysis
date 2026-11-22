"""
Tahap 1 Tugas 3 - Eksperimen Perbandingan 3 Kategori Ekstraksi Fitur Teks pada Base Pipeline.

Protokol identik dengan Tugas 2 (`src/model.py` & `src/run_experiments.py`):
  - Fitur time-series `kurs`: 7 fitur lag/rolling mean/std dari `src/dataset.py`.
  - Klasifier: `XGBClassifier` (`GRID`: max_depth in {2, 3}, n_estimators in {100, 300})
    dipilih berdasarkan Directional Accuracy (DA) di split `val`.
  - Untuk setiap representasi embedding berdimensi tinggi (`fasttext`, `finbert_emb`,
    `bge_small`, `bge_large`), dimensi terbaik (`PCA k in {8, 16, 32, 50}` atau `raw`)
    dipilih HANYA berdasarkan skor `val` (DA tertinggi, tie-break F1_macro val),
    lalu dievaluasi satu kali di `test`.

Output:
  - results/stage1_feature_extraction_metrics.csv
  - results/stage1_pca_sweep.csv
  - results/stage1_predictions_test.csv
"""
import json

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from baseline import arima_baseline, majority_baseline, persistence_baseline
from config import PROCESSED, ROOT
from dataset import load_dataset
from evaluate import metrics
from model import run_xgb

RES = ROOT / "results"
EMB_DIR = PROCESSED / "embeddings"
FEATURES_EMB_DAILY = PROCESSED / "features_embeddings_daily.csv"
PCA_DIMS = (8, 16, 32, 50)

EMBEDDING_SPECS = [
    ("fasttext", "Kategori A: Static (FastText)", "ft", "daily_fasttext.parquet"),
    ("finbert_emb", "Kategori B: Domain (FinBERT Emb)", "fb_emb", "daily_finbert_emb.parquet"),
    ("bge_small", "Kategori C: General (BGE-Small)", "bge_s", "daily_bge_small.parquet"),
    ("bge_large", "Kategori C: General (BGE-Large)", "bge_l", "daily_bge_large.parquet"),
]


def exact_mcnemar_pvalue(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    """Uji McNemar eksak dua sisi antara akurasi dua model pada data test."""
    ca = np.asarray(pred_a) == np.asarray(y_true)
    cb = np.asarray(pred_b) == np.asarray(y_true)
    n00 = int(np.sum((~ca) & (~cb)))
    n01 = int(np.sum((~ca) & cb))
    n10 = int(np.sum(ca & (~cb)))
    n11 = int(np.sum(ca & cb))
    if (n01 + n10) == 0:
        return 1.0
    res = mcnemar([[n00, n01], [n10, n11]], exact=True)
    return float(res.pvalue)


def load_stage1_dataframe() -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, dict[str, list[str]]]]:
    """Gabungkan tabel base Tugas 2 dengan fitur PCA, sentimen FinBERT, dan vektor raw."""
    df_base, base_sets = load_dataset()
    emb_pca = pd.read_csv(FEATURES_EMB_DAILY, parse_dates=["date"])
    df = df_base.merge(emb_pca, on="date", how="left")

    emb_cols: dict[str, dict[str, list[str]]] = {}
    for name, _, prefix, parquet_name in EMBEDDING_SPECS:
        emb_cols[name] = {}
        for k in PCA_DIMS:
            cols = [c for c in df.columns if c.startswith(f"{prefix}_pca{k}_")]
            if cols:
                emb_cols[name][f"pca{k}"] = cols
        raw_path = EMB_DIR / parquet_name
        if raw_path.exists():
            raw_df = pd.read_parquet(raw_path)
            raw_df["date"] = pd.to_datetime(raw_df["date"])
            raw_feature_cols = [c for c in raw_df.columns if c != "date"]
            df = df.merge(raw_df, on="date", how="left")
            emb_cols[name]["raw"] = raw_feature_cols

    finbert_sent_cols = [
        c for c in ("fb_pos_mean", "fb_neg_mean", "fb_neu_mean", "fb_net_mean", "fb_share_neg", "fb_share_pos")
        if c in df.columns
    ]
    base_sets["finbert_sent"] = finbert_sent_cols
    return df, base_sets, emb_cols


def eval_xgb_variant(df: pd.DataFrame, cols: list[str]) -> dict:
    hp, out = run_xgb(df, cols)
    val_m = metrics(*out["val"])
    test_m = metrics(*out["test"])
    return {
        "hp": hp,
        "val_y": out["val"][0],
        "val_pred": out["val"][1],
        "val_metrics": val_m,
        "test_y": out["test"][0],
        "test_pred": out["test"][1],
        "test_metrics": test_m,
    }


def main():
    RES.mkdir(parents=True, exist_ok=True)
    df, feature_sets, emb_cols = load_stage1_dataframe()
    masks = {"val": df["split"] == "val", "test": df["split"] == "test"}
    kurs_cols = feature_sets["kurs"]

    metric_rows = []
    sweep_rows = []
    test_preds = {}

    def record_metric(model_name, category, split, y, pred, n_features, note=""):
        metric_rows.append({
            "model": model_name,
            "category": category,
            "split": split,
            "n": len(y),
            "n_features": n_features,
            **metrics(y, pred),
            "note": note,
        })

    # -----------------------------------------------------------------------
    # 1. Baseline Tugas 2: Model A (Time-Series saja) & Model B (NLP Dasar)
    # -----------------------------------------------------------------------
    for name, pred in {
        "majority": majority_baseline(df),
        "persistence": persistence_baseline(df),
        "arima": arima_baseline(df),
    }.items():
        for split, m in masks.items():
            record_metric(name, "Model A: Time-Series Baseline", split, df.loc[m, "y"].values, pred[m].values, 1)
        test_preds[name] = pred[masks["test"]].values

    baseline_xgb_specs = [
        ("xgb:kurs", "Model A: Time-Series Baseline", ["kurs"]),
        ("xgb:kurs+lexicon", "Model B: NLP Dasar (Tugas 2)", ["kurs", "lexicon"]),
        ("xgb:kurs+tfidf", "Model B: NLP Dasar (Tugas 2)", ["kurs", "tfidf"]),
        ("xgb:kurs+lexicon+tfidf", "Model B: NLP Dasar (Tugas 2)", ["kurs", "lexicon", "tfidf"]),
    ]
    for name, cat, sets in baseline_xgb_specs:
        cols = [c for s in sets for c in feature_sets[s]]
        res = eval_xgb_variant(df, cols)
        for split in ("val", "test"):
            record_metric(
                name, cat, split,
                res[f"{split}_y"], res[f"{split}_pred"], len(cols), json.dumps(res["hp"])
            )
        test_preds[name] = res["test_pred"]

    # -----------------------------------------------------------------------
    # 2. Sweep Dimensi (PCA k in {8, 16, 32, 50} & raw) untuk tiap Embedding
    # -----------------------------------------------------------------------
    best_dim_by_name = {}
    best_pca_by_name = {}
    for name, cat, _, _ in EMBEDDING_SPECS:
        dim_map = emb_cols[name]
        best_any = None
        best_pca = None
        for dim_label in [f"pca{k}" for k in PCA_DIMS] + ["raw"]:
            if dim_label not in dim_map:
                continue
            cols = kurs_cols + dim_map[dim_label]
            res = eval_xgb_variant(df, cols)
            vm, tm = res["val_metrics"], res["test_metrics"]
            sweep_rows.append({
                "embedding": name,
                "category": cat,
                "dim_setting": dim_label,
                "n_emb_features": len(dim_map[dim_label]),
                "n_total_features": len(cols),
                "val_DA": vm["DA"],
                "val_F1_macro": vm["F1_macro"],
                "val_share_pred1": vm["share_pred1"],
                "test_DA": tm["DA"],
                "test_F1_macro": tm["F1_macro"],
                "test_share_pred1": tm["share_pred1"],
                "hp": json.dumps(res["hp"]),
            })
            score = (vm["DA"], vm["F1_macro"])
            if best_any is None or score > best_any[0]:
                best_any = (score, dim_label, len(cols), res)
            if dim_label != "raw" and (best_pca is None or score > best_pca[0]):
                best_pca = (score, dim_label, len(cols), res)

        best_dim_by_name[name] = best_any
        best_pca_by_name[name] = best_pca

        # Catat varian terpilih berdasarkan val + varian raw untuk transparansi
        _, best_lbl, best_nfeat, best_res = best_any
        for split in ("val", "test"):
            record_metric(
                f"xgb:kurs+{name} ({best_lbl}_val_best)", cat, split,
                best_res[f"{split}_y"], best_res[f"{split}_pred"], best_nfeat,
                f"dim={best_lbl}, hp={json.dumps(best_res['hp'])}"
            )
        test_preds[f"xgb:kurs+{name}_best"] = best_res["test_pred"]

        if "raw" in dim_map:
            raw_cols = kurs_cols + dim_map["raw"]
            raw_res = eval_xgb_variant(df, raw_cols)
            for split in ("val", "test"):
                record_metric(
                    f"xgb:kurs+{name} (raw_{len(dim_map['raw'])}d)", cat, split,
                    raw_res[f"{split}_y"], raw_res[f"{split}_pred"], len(raw_cols),
                    f"dim=raw, hp={json.dumps(raw_res['hp'])}"
                )
            test_preds[f"xgb:kurs+{name}_raw"] = raw_res["test_pred"]

    # -----------------------------------------------------------------------
    # 3. FinBERT Sentimen (6D) & Gabungan FinBERT Emb (PCA terbaik di val) + Sent
    # -----------------------------------------------------------------------
    sent_cols = feature_sets["finbert_sent"]
    if sent_cols:
        res_sent = eval_xgb_variant(df, kurs_cols + sent_cols)
        for split in ("val", "test"):
            record_metric(
                "xgb:kurs+finbert_sent (6d)", "Kategori B: Domain (FinBERT Sent)", split,
                res_sent[f"{split}_y"], res_sent[f"{split}_pred"], len(kurs_cols) + len(sent_cols),
                json.dumps(res_sent["hp"])
            )
        test_preds["xgb:kurs+finbert_sent"] = res_sent["test_pred"]

        if "finbert_emb" in best_pca_by_name and best_pca_by_name["finbert_emb"] is not None:
            _, fb_pca_lbl, _, _ = best_pca_by_name["finbert_emb"]
            comb_cols = kurs_cols + emb_cols["finbert_emb"][fb_pca_lbl] + sent_cols
            res_comb = eval_xgb_variant(df, comb_cols)
            for split in ("val", "test"):
                record_metric(
                    f"xgb:kurs+finbert_{fb_pca_lbl}+sent", "Kategori B: Domain (FinBERT Emb+Sent)", split,
                    res_comb[f"{split}_y"], res_comb[f"{split}_pred"], len(comb_cols),
                    f"dim={fb_pca_lbl}+sent6, hp={json.dumps(res_comb['hp'])}"
                )
            test_preds["xgb:kurs+finbert_emb+sent"] = res_comb["test_pred"]

    # -----------------------------------------------------------------------
    # 4. Uji McNemar di test set terhadap Model A (xgb:kurs) & Model B (xgb:kurs+lexicon)
    # -----------------------------------------------------------------------
    y_test = df.loc[masks["test"], "y"].values
    ref_kurs = test_preds["xgb:kurs"]
    ref_lex = test_preds["xgb:kurs+lexicon"]

    pred_key_lookup = {}
    for row in metric_rows:
        mname = row["model"]
        if mname in test_preds:
            pred_key_lookup[mname] = mname
        elif "_val_best)" in mname:
            base_k = mname.split(" (")[0] + "_best"
            pred_key_lookup[mname] = base_k
        elif "(raw_" in mname:
            base_k = mname.split(" (")[0] + "_raw"
            pred_key_lookup[mname] = base_k
        elif mname == "xgb:kurs+finbert_sent (6d)":
            pred_key_lookup[mname] = "xgb:kurs+finbert_sent"
        elif mname.startswith("xgb:kurs+finbert_") and mname.endswith("+sent"):
            pred_key_lookup[mname] = "xgb:kurs+finbert_emb+sent"

    for row in metric_rows:
        if row["split"] == "test":
            pk = pred_key_lookup[row["model"]]
            p = test_preds[pk]
            row["p_mcnemar_vs_kurs"] = exact_mcnemar_pvalue(y_test, ref_kurs, p)
            row["p_mcnemar_vs_lexicon"] = exact_mcnemar_pvalue(y_test, ref_lex, p)
        else:
            row["p_mcnemar_vs_kurs"] = np.nan
            row["p_mcnemar_vs_lexicon"] = np.nan

    metrics_df = pd.DataFrame(metric_rows)
    sweep_df = pd.DataFrame(sweep_rows)
    preds_df = pd.DataFrame({"date": df.loc[masks["test"], "date"].values, "y": y_test, **test_preds})

    metrics_df.to_csv(RES / "stage1_feature_extraction_metrics.csv", index=False)
    sweep_df.to_csv(RES / "stage1_pca_sweep.csv", index=False)
    preds_df.to_csv(RES / "stage1_predictions_test.csv", index=False)

    print("\n=== RINGKASAN HASIL TAHAP 1 (TEST SET, DIPILIH VIA VAL) ===")
    test_view = metrics_df[metrics_df["split"] == "test"][
        ["model", "n_features", "DA", "F1_macro", "share_pred1", "p_mcnemar_vs_kurs"]
    ].copy()
    val_da = metrics_df[metrics_df["split"] == "val"].set_index("model")["DA"]
    test_view.insert(2, "DA_val", test_view["model"].map(val_da))
    print(test_view.round(4).to_string(index=False))
    print("\n=== DETAIL PCA SWEEP ===")
    print(sweep_df[["embedding", "dim_setting", "val_DA", "val_F1_macro", "test_DA", "test_F1_macro"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
