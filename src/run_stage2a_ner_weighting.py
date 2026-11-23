"""
Tahap 2A - Uji Sensitivitas Bobot NER Bertingkat & Perbandingan dengan Pipeline Sebelumnya.

Basis teks: `BAAI/bge-small-en-v1.5` (pemenang Tahap 1) + cache probabilitas FinBERT.
Pembobotan bertingkat (normalisasi total bobot hari T = 1.0):
  * Tier 1: Moneter & kurs langsung (bank sentral + istilah suku bunga/valas)
  * Tier 2: Makro & geopolitik penggerak (negara kunci GPE/NORP + indikator makro/komoditas)
  * Tier 0: Berita lainnya (bobot dasar 1.0)

Output:
  - results/stage2a_ner_sensitivity_sweep.csv
  - results/stage2a_ner_comparison_metrics.csv
  - results/stage2a_ner_predictions_test.csv
  - data/processed/embeddings/daily_bge_small_ner_best.parquet
  - data/processed/features_stage2a_ner_daily.csv
"""
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import DAILY_BASE, PROCESSED, ROOT, SEED
from dataset import load_dataset
from evaluate import metrics
from extract_embeddings import EMB_DIR, PCA_DIMS, load_articles_and_base, save_daily_parquet
from model import run_xgb
from ner_weighting import (
    aggregate_daily_weighted,
    compute_article_weights,
    compute_daily_ner_meta,
    extract_ner_tiers,
    print_ner_summary,
)
from run_stage1_comparison import exact_mcnemar_pvalue

RES = ROOT / "results"
FEATURES_NER_DAILY = PROCESSED / "features_stage2a_ner_daily.csv"

# Grid sensitivitas pengali Tier 1 vs Tier 2
WEIGHT_SCHEMES = [
    ("uniform", 0.0, 0.0, "hierarchical", 1.0, "1.0 / 1.0 / 1.0 (kontrol simple mean)"),
    ("tier_soft", 1.0, 0.5, "hierarchical", 1.0, "2.0 / 1.5 / 1.0"),
    ("tier_med", 2.0, 1.0, "hierarchical", 1.0, "3.0 / 2.0 / 1.0"),
    ("tier_strong", 3.0, 1.0, "hierarchical", 1.0, "4.0 / 2.0 / 1.0"),
    ("tier_sharp", 4.0, 1.5, "hierarchical", 1.0, "5.0 / 2.5 / 1.0"),
    ("additive_med", 2.0, 1.0, "additive", 1.0, "1 + 2*T1 + 1*T2"),
    ("additive_strong", 3.0, 1.0, "additive", 1.0, "1 + 3*T1 + 1*T2"),
    ("tier_filter_only", 2.0, 0.0, "hierarchical", 0.0, "3.0 / 1.0 / 0.0 (hard filter T0=0)"),
]


def fit_train_pca_blocks(daily_mat: np.ndarray, is_train: np.ndarray, prefix: str) -> pd.DataFrame:
    """Fit StandardScaler + PCA hanya pada 720 hari train."""
    scaler = StandardScaler().fit(daily_mat[is_train])
    X_scaled = scaler.transform(daily_mat)
    blocks = []
    for k in PCA_DIMS:
        pca = PCA(n_components=k, random_state=SEED).fit(X_scaled[is_train])
        Z = pca.transform(X_scaled)
        cols = [f"{prefix}_pca{k}_{i:02d}" for i in range(k)]
        blocks.append(pd.DataFrame(Z, columns=cols))
    return pd.concat(blocks, axis=1)


def compute_weighted_finbert_sent(
    art_probs: np.ndarray, target_dates: pd.Series, base_dates: pd.Series, norm_weights: np.ndarray
) -> pd.DataFrame:
    """Agregasi probabilitas FinBERT [pos, neg, neu] dengan bobot NER harian yang sama (total=1)."""
    weighted_probs, _ = aggregate_daily_weighted(art_probs, target_dates, base_dates, norm_weights)
    date_to_idx = {d: i for i, d in enumerate(pd.to_datetime(base_dates))}
    day_idx = pd.to_datetime(target_dates).map(date_to_idx).to_numpy(dtype=np.int64)
    n_days = len(base_dates)

    is_neg = (art_probs[:, 1] > 0.5).astype(np.float64) * norm_weights
    is_pos = (art_probs[:, 0] > 0.5).astype(np.float64) * norm_weights
    share_neg = np.bincount(day_idx, weights=is_neg, minlength=n_days)
    share_pos = np.bincount(day_idx, weights=is_pos, minlength=n_days)

    return pd.DataFrame({
        "fb_ner_pos_mean": weighted_probs[:, 0],
        "fb_ner_neg_mean": weighted_probs[:, 1],
        "fb_ner_neu_mean": weighted_probs[:, 2],
        "fb_ner_net_mean": weighted_probs[:, 0] - weighted_probs[:, 1],
        "fb_ner_share_neg": share_neg,
        "fb_ner_share_pos": share_pos,
    })


def eval_on_df(df_base: pd.DataFrame, feat_df: pd.DataFrame, feat_cols: list[str]) -> dict:
    """Gabungkan fitur sementara dan jalankan protokol run_xgb yang identik dengan Tugas 2."""
    kurs_cols = [c for c in df_base.columns if c.startswith("k_")]
    df_eval = pd.concat([df_base[["date", "y", "split"] + kurs_cols].reset_index(drop=True), feat_df[feat_cols].reset_index(drop=True)], axis=1)
    cols = kurs_cols + feat_cols
    hp, out = run_xgb(df_eval, cols)
    vm = metrics(*out["val"])
    tm = metrics(*out["test"])
    return {
        "hp": hp,
        "n_features": len(cols),
        "val_y": out["val"][0],
        "val_pred": out["val"][1],
        "val_metrics": vm,
        "test_y": out["test"][0],
        "test_pred": out["test"][1],
        "test_metrics": tm,
    }


def main():
    RES.mkdir(parents=True, exist_ok=True)
    arts, base = load_articles_and_base()
    ner_df = extract_ner_tiers(arts)
    print_ner_summary(arts, ner_df)

    art_bge = np.load(EMB_DIR / "articles_bge_small.npz")["embeddings"].astype(np.float32)
    art_fb_probs = np.load(EMB_DIR / "articles_finbert.npz")["sentiment_probs"].astype(np.float32)
    assert len(arts) == len(ner_df) == len(art_bge) == len(art_fb_probs), "Jumlah artikel tidak sinkron!"

    df_base, _ = load_dataset()
    is_train = df_base["split"].eq("train").to_numpy()

    # Sanity check: uniform (0, 0) harus cocok dengan daily_bge_small.parquet dari Tahap 1
    ref_bge_daily = pd.read_parquet(EMB_DIR / "daily_bge_small.parquet").drop(columns=["date"]).to_numpy(dtype=np.float32)

    sweep_rows = []
    scheme_cache = {}
    best_by_scheme = {}
    best_nonuniform = None

    print("\n=== UJI SENSITIVITAS PEMBOBOTAN NER (BGE-Small + XGBoost) ===")
    for name, b1, b2, mode, t0_w, desc in WEIGHT_SCHEMES:
        raw_w = compute_article_weights(ner_df, b1=b1, b2=b2, mode=mode, t0_weight=t0_w)
        daily_mat, norm_w = aggregate_daily_weighted(art_bge, ner_df["target_date"], base["date"], raw_w)

        if name == "uniform":
            max_diff = float(np.max(np.abs(daily_mat - ref_bge_daily)))
            print(f"[Sanity Check] Max diff uniform vs Tahap 1 daily_bge_small: {max_diff:.2e}")
            assert max_diff < 1e-4, f"Mismatch uniform vs Tahap 1: {max_diff}"

        pca_df = fit_train_pca_blocks(daily_mat, is_train, prefix="bge_ner")
        raw_cols = [f"bge_ner_raw_{i:04d}" for i in range(daily_mat.shape[1])]
        raw_df = pd.DataFrame(daily_mat, columns=raw_cols)
        full_feat_df = pd.concat([pca_df, raw_df], axis=1)

        scheme_cache[name] = {
            "desc": desc,
            "daily_mat": daily_mat,
            "norm_w": norm_w,
            "pca_df": pca_df,
            "full_feat_df": full_feat_df,
            "raw_cols": raw_cols,
        }

        best_this = None
        for dim_label in [f"pca{k}" for k in PCA_DIMS] + ["raw"]:
            fcols = raw_cols if dim_label == "raw" else [c for c in pca_df.columns if c.startswith(f"bge_ner_{dim_label}_")]
            res = eval_on_df(df_base, full_feat_df, fcols)
            vm, tm = res["val_metrics"], res["test_metrics"]
            sweep_rows.append({
                "scheme": name,
                "ratio_desc": desc,
                "b1": b1,
                "b2": b2,
                "mode": mode,
                "t0_weight": t0_w,
                "dim_setting": dim_label,
                "n_emb_features": len(fcols),
                "n_total_features": res["n_features"],
                "val_DA": vm["DA"],
                "val_F1_macro": vm["F1_macro"],
                "val_share_pred1": vm["share_pred1"],
                "test_DA": tm["DA"],
                "test_F1_macro": tm["F1_macro"],
                "test_share_pred1": tm["share_pred1"],
                "hp": json.dumps(res["hp"]),
            })

            score = (vm["DA"], vm["F1_macro"], -res["n_features"])
            if best_this is None or score > best_this[0]:
                best_this = (score, dim_label, fcols, res)
            if name != "uniform" and (best_nonuniform is None or score > best_nonuniform[0]):
                best_nonuniform = (score, name, dim_label, fcols, res)

        best_by_scheme[name] = best_this
        _, b_dim, _, b_res = best_this
        print(
            f"  {name:16s} ({desc:30s}) | val_best={b_dim:5s} -> "
            f"DA val={b_res['val_metrics']['DA']:.4f} (F1={b_res['val_metrics']['F1_macro']:.4f}) | "
            f"DA test={b_res['test_metrics']['DA']:.4f} (F1={b_res['test_metrics']['F1_macro']:.4f})"
        )

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(RES / "stage2a_ner_sensitivity_sweep.csv", index=False)

    # Pemenang non-uniform di val
    _, win_scheme, win_dim, win_fcols, win_res = best_nonuniform
    win_info = scheme_cache[win_scheme]
    print(f"\n[Terpilih di Val] scheme='{win_scheme}' ({win_info['desc']}), dim='{win_dim}'")

    # Simpan fitur harian pemenang NER untuk tahap berikutnya (2B / Fusion / Analisis)
    save_daily_parquet(win_info["daily_mat"], base, "bge_ner_raw", EMB_DIR / "daily_bge_small_ner_best.parquet")
    ner_meta_df = compute_daily_ner_meta(ner_df, base["date"], win_info["norm_w"])
    fb_sent_ner_df = compute_weighted_finbert_sent(art_fb_probs, ner_df["target_date"], base["date"], win_info["norm_w"])
    stage2a_daily = pd.concat(
        [pd.DataFrame({"date": base["date"].values}), win_info["pca_df"], ner_meta_df, fb_sent_ner_df],
        axis=1,
    )
    stage2a_daily.to_csv(FEATURES_NER_DAILY, index=False)

    # 3 Eksperimen tambahan dengan skema pemenang NER
    meta_cols = list(ner_meta_df.columns)
    sent_ner_cols = list(fb_sent_ner_df.columns)
    aug_df = pd.concat([win_info["full_feat_df"], ner_meta_df, fb_sent_ner_df], axis=1)

    extra_specs = [
        (f"xgb:kurs+bge_small_ner ({win_scheme}, {win_dim})+ner_meta", "Tahap 2A: NER Weighting + Meta", win_fcols + meta_cols),
        (f"xgb:kurs+finbert_sent_ner ({win_scheme})", "Tahap 2A: FinBERT Sent (NER-Weighted)", sent_ner_cols),
        (f"xgb:kurs+bge_small_ner ({win_scheme}, {win_dim})+finbert_sent_ner", "Tahap 2A: BGE+FinBERT (NER-Weighted)", win_fcols + sent_ner_cols),
    ]

    # Gabungkan metrik & prediksi sebelumnya (Tahap 1 memuat Model A, Model B, dan Tahap 1)
    prev_metrics = pd.read_csv(RES / "stage1_feature_extraction_metrics.csv")
    prev_preds = pd.read_csv(RES / "stage1_predictions_test.csv")
    y_test = prev_preds["y"].to_numpy()

    new_rows = []
    new_test_preds = {}

    def add_new_result(model_label: str, cat_label: str, res_dict: dict, pred_col_key: str, note: str):
        for split in ("val", "test"):
            m = res_dict[f"{split}_metrics"]
            new_rows.append({
                "model": model_label,
                "category": cat_label,
                "split": split,
                "n": len(res_dict[f"{split}_y"]),
                "n_features": res_dict["n_features"],
                "DA": m["DA"],
                "F1_macro": m["F1_macro"],
                "share_pred1": m["share_pred1"],
                "note": note,
            })
        new_test_preds[pred_col_key] = res_dict["test_pred"]

    # Catat pemenang di val + referensi eksplisit tier_med (+2/+1), tier_strong (+3/+1), dan tier_filter_only
    add_new_result(
        f"xgb:kurs+bge_small_ner ({win_scheme}, {win_dim}_val_best)",
        "Tahap 2A: NER Weighting (BGE-Small)",
        win_res,
        f"xgb:kurs+bge_small_ner_{win_scheme}_best",
        f"scheme={win_scheme}, dim={win_dim}, hp={json.dumps(win_res['hp'])}",
    )
    for ref_scheme in ("tier_med", "tier_strong", "tier_filter_only"):
        if ref_scheme == win_scheme:
            continue
        _, r_dim, _, r_res = best_by_scheme[ref_scheme]
        add_new_result(
            f"xgb:kurs+bge_small_ner ({ref_scheme}, {r_dim})",
            "Tahap 2A: NER Weighting (BGE-Small)",
            r_res,
            f"xgb:kurs+bge_small_ner_{ref_scheme}_{r_dim}",
            f"scheme={ref_scheme}, dim={r_dim}, hp={json.dumps(r_res['hp'])}",
        )

    for label, cat, cols in extra_specs:
        res_x = eval_on_df(df_base, aug_df, cols)
        key = label.split(" (")[0] + "_" + cat.split(":")[0].strip().lower().replace(" ", "_")
        if "finbert_sent_ner" in label and "bge_small" in label:
            key = "xgb:kurs+bge_small_ner+finbert_sent_ner"
        elif "finbert_sent_ner" in label:
            key = "xgb:kurs+finbert_sent_ner"
        else:
            key = "xgb:kurs+bge_small_ner+meta"
        add_new_result(label, cat, res_x, key, f"hp={json.dumps(res_x['hp'])}")

    # Gabungkan semua baris & hitung McNemar vsModel A (xgb:kurs), Model B (xgb:kurs+lexicon), & Tahap 1 (bge_small simple mean)
    prev_keep = prev_metrics[[
        "model", "category", "split", "n", "n_features", "DA", "F1_macro", "share_pred1", "note"
    ]].copy()
    all_metrics = pd.concat([prev_keep, pd.DataFrame(new_rows)], ignore_index=True)

    all_preds_df = pd.concat([prev_preds, pd.DataFrame(new_test_preds)], axis=1)
    ref_kurs = all_preds_df["xgb:kurs"].to_numpy()
    ref_lex = all_preds_df["xgb:kurs+lexicon"].to_numpy()
    ref_bge_s1 = all_preds_df["xgb:kurs+bge_small_best"].to_numpy()

    # Map nama baris metrik -> kolom prediksi test
    pred_lookup = {}
    for mname in all_metrics["model"].unique():
        if mname in all_preds_df.columns:
            pred_lookup[mname] = mname
        elif "_val_best)" in mname and "ner" not in mname:
            pred_lookup[mname] = mname.split(" (")[0] + "_best"
        elif "(raw_" in mname:
            pred_lookup[mname] = mname.split(" (")[0] + "_raw"
        elif mname == "xgb:kurs+finbert_sent (6d)":
            pred_lookup[mname] = "xgb:kurs+finbert_sent"
        elif mname.startswith("xgb:kurs+finbert_") and mname.endswith("+sent"):
            pred_lookup[mname] = "xgb:kurs+finbert_emb+sent"
        elif mname == f"xgb:kurs+bge_small_ner ({win_scheme}, {win_dim}_val_best)":
            pred_lookup[mname] = f"xgb:kurs+bge_small_ner_{win_scheme}_best"
        elif mname.endswith("+ner_meta"):
            pred_lookup[mname] = "xgb:kurs+bge_small_ner+meta"
        elif mname.endswith("+finbert_sent_ner"):
            pred_lookup[mname] = "xgb:kurs+bge_small_ner+finbert_sent_ner"
        elif mname.startswith("xgb:kurs+finbert_sent_ner"):
            pred_lookup[mname] = "xgb:kurs+finbert_sent_ner"
        else:
            for ref_s in ("tier_med", "tier_strong", "tier_filter_only"):
                if f"({ref_s}," in mname:
                    _, r_dim, _, _ = best_by_scheme[ref_s]
                    pred_lookup[mname] = f"xgb:kurs+bge_small_ner_{ref_s}_{r_dim}"

    p_kurs, p_lex, p_bge = [], [], []
    for _, row in all_metrics.iterrows():
        if row["split"] == "test":
            p = all_preds_df[pred_lookup[row["model"]]].to_numpy()
            p_kurs.append(exact_mcnemar_pvalue(y_test, ref_kurs, p))
            p_lex.append(exact_mcnemar_pvalue(y_test, ref_lex, p))
            p_bge.append(exact_mcnemar_pvalue(y_test, ref_bge_s1, p))
        else:
            p_kurs.append(np.nan)
            p_lex.append(np.nan)
            p_bge.append(np.nan)

    all_metrics["p_mcnemar_vs_kurs"] = p_kurs
    all_metrics["p_mcnemar_vs_lexicon"] = p_lex
    all_metrics["p_mcnemar_vs_bge_simple"] = p_bge

    all_metrics.to_csv(RES / "stage2a_ner_comparison_metrics.csv", index=False)
    all_preds_df.to_csv(RES / "stage2a_ner_predictions_test.csv", index=False)

    print("\n=== PERBANDINGAN LENGKAP (VAL & TEST) ===")
    val_map = all_metrics[all_metrics["split"] == "val"].set_index("model")["DA"]
    val_f1_map = all_metrics[all_metrics["split"] == "val"].set_index("model")["F1_macro"]
    tview = all_metrics[all_metrics["split"] == "test"][
        ["model", "n_features", "DA", "F1_macro", "share_pred1", "p_mcnemar_vs_kurs", "p_mcnemar_vs_bge_simple"]
    ].copy()
    tview.insert(2, "DA_val", tview["model"].map(val_map))
    tview.insert(3, "F1_val", tview["model"].map(val_f1_map))
    print(tview.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
