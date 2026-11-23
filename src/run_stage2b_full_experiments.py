"""
Master Experiment & Ablation Runner (Tugas 3: Tahap 1 -> Tahap 2A -> Tahap 2B -> Tahap 2C).

Menggabungkan:
  1. Hasil Tahap 1 (Static vs Domain vs General Embeddings) & Tahap 2A (NER Weighting)
     dari CSV hasil yang sudah ada (tanpa re-ekstraksi embedding).
  2. Tahap 2B: Ablasi Feature Engineering (Extended X_time, BGE Topic Modeling,
     EWMA Sliding Window + Shock, NER Moderat + FinBERT Sentimen) across LogReg, XGBoost, LightGBM.
  3. Tahap 2C: Multimodal Fusion (Early Fusion [LogReg, XGB, LGBM, LSTM],
     Late Fusion [Weighted Prob, Weighted Logit, Stacking OOF], Hybrid Gated Neural Net)
     + Threshold Optimization di `val`.

Output:
  - results/tugas3_stage2b_fe_ablation.csv
  - results/tugas3_stage2c_fusion_metrics.csv
  - results/tugas3_full_ablation_metrics.csv
  - results/tugas3_full_predictions_test.csv
  - results/tugas3_summary.json
"""
import json

import numpy as np
import pandas as pd

from config import PROCESSED, ROOT
from extract_embeddings import EMB_DIR
from finetune_transformer import train_text_projection_head
from multimodal_fusion import (
    eval_prob_metrics,
    fit_eval_gated_fusion,
    fit_eval_lgbm,
    fit_eval_logreg,
    fit_eval_lstm,
    fit_eval_tabular,
    fit_eval_xgb,
    run_late_stacking_fusion,
    run_late_weighted_fusion,
    tune_threshold,
)
from run_stage1_comparison import exact_mcnemar_pvalue

RES = ROOT / "results"
FEATURES_T3_FULL = PROCESSED / "features_tugas3_full.parquet"


def build_feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    cols = list(df.columns)
    g = {
        "time_basic": [f"k_ret_lag{i}" for i in range(1, 6)] + ["k_mean5", "k_std5"],
        "time_ext": [c for c in cols if c.startswith("k_")],
        "lexicon": [
            "t_compound_mean", "t_share_neg", "t_share_pos",
            "t_lm_negative_mean", "t_lm_positive_mean", "t_lm_uncertainty_mean", "t_lm_litigious_mean",
        ],
        "topic": [c for c in cols if c.startswith("topic_share_")] + ["topic_entropy"],
        "ner_meta": ["ner_share_tier1", "ner_share_tier2", "ner_count_tier1_log", "ner_eff_n_ratio"],
        "fb_soft": [c for c in cols if c.startswith("fb_soft_") and not c.endswith(("_ewma3", "_ewma5", "_diff5"))],
        "ewma_shock": (
            ["bge_cos_shift_1d", "bge_cos_shift_5d", "n_articles_z5", "fb_soft_net_diff5"]
            + [c for c in cols if c.endswith(("_ewma3", "_ewma5"))]
        ),
        "bge_raw384": [c for c in cols if c.startswith("bge_s_raw_")],
        "bge_pca16": [c for c in cols if c.startswith("bge_raw_pca16_")],
        "bge_pca32": [c for c in cols if c.startswith("bge_raw_pca32_")],
        "bge_ewma3_pca16": [c for c in cols if c.startswith("bge_ewma3_pca16_")],
        "bge_ewma3_pca32": [c for c in cols if c.startswith("bge_ewma3_pca32_")],
        "bge_ewma5_pca16": [c for c in cols if c.startswith("bge_ewma5_pca16_")],
        "bge_soft_pca32": [c for c in cols if c.startswith("bge_soft_pca32_")],
        "bge_soft_raw384": [c for c in cols if c.startswith("bge_soft_raw_")],
        "bge_ewma3_raw384": [c for c in cols if c.startswith("bge_ewma3_raw_")],
        "bge_add_pca32": [c for c in cols if c.startswith("bge_add_pca32_")],
    }
    g["full_fe_compact_text"] = (
        g["bge_pca16"] + g["bge_ewma3_pca16"] + g["topic"] + g["ner_meta"] + g["fb_soft"] + g["ewma_shock"]
    )
    g["full_fe_pca32_text"] = (
        g["bge_pca32"] + g["bge_ewma3_pca16"] + g["topic"] + g["ner_meta"] + g["fb_soft"] + g["ewma_shock"]
    )
    g["full_fe_raw_text"] = (
        g["bge_raw384"] + g["bge_ewma3_pca16"] + g["topic"] + g["ner_meta"] + g["fb_soft"] + g["ewma_shock"]
    )
    return g


def load_full_dataframe() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_T3_FULL)
    df["date"] = pd.to_datetime(df["date"])
    for pq in ("daily_bge_small.parquet", "daily_bge_ner_soft.parquet", "daily_bge_ewma3.parquet"):
        sub = pd.read_parquet(EMB_DIR / pq)
        sub["date"] = pd.to_datetime(sub["date"])
        df = df.merge(sub, on="date", how="left")
    return df


def _score_val(m: dict) -> tuple[float, float, float]:
    return (0.5 * (m["DA"] + m["F1_macro"]), m["DA"], m["ROC_AUC"])


def main():
    RES.mkdir(parents=True, exist_ok=True)
    df = load_full_dataframe()
    fg = build_feature_groups(df)

    m_tr = df["split"].eq("train").to_numpy()
    m_va = df["split"].eq("val").to_numpy()
    m_te = df["split"].eq("test").to_numpy()
    y_tr = df.loc[m_tr, "y"].to_numpy(dtype=int)
    y_va = df.loc[m_va, "y"].to_numpy(dtype=int)
    y_te = df.loc[m_te, "y"].to_numpy(dtype=int)

    # -----------------------------------------------------------------------
    # 1. Tahap 2B: Ablasi Feature Engineering across LogReg, XGB, LGBM
    # -----------------------------------------------------------------------
    fe_configs = [
        ("t2b:time_ext_only", "Tahap 2B: Extended Time-Series", fg["time_ext"]),
        ("t2b:time_ext+bge_raw384", "Tahap 2B: Ext Time + BGE Raw", fg["time_ext"] + fg["bge_raw384"]),
        ("t2b:time_ext+bge_pca16", "Tahap 2B: Ext Time + BGE PCA16", fg["time_ext"] + fg["bge_pca16"]),
        ("t2b:time_ext+bge_pca32", "Tahap 2B: Ext Time + BGE PCA32", fg["time_ext"] + fg["bge_pca32"]),
        ("t2b:time_ext+bge_pca32+topic", "Tahap 2B: +Topic Modeling", fg["time_ext"] + fg["bge_pca32"] + fg["topic"]),
        ("t2b:time_ext+bge_raw384+topic", "Tahap 2B: +Topic Modeling (Raw)", fg["time_ext"] + fg["bge_raw384"] + fg["topic"]),
        ("t2b:time_ext+bge_ewma3_pca32+shock", "Tahap 2B: +EWMA3 & Shock", fg["time_ext"] + fg["bge_ewma3_pca32"] + fg["ewma_shock"]),
        ("t2b:time_ext+bge_pca32+ewma_shock", "Tahap 2B: +BGE PCA32 & Shock", fg["time_ext"] + fg["bge_pca32"] + fg["ewma_shock"]),
        ("t2b:time_ext+bge_soft_pca32+ner_fb", "Tahap 2B: +NER Soft & FinBERT", fg["time_ext"] + fg["bge_soft_pca32"] + fg["ner_meta"] + fg["fb_soft"]),
        ("t2b:time_ext+bge_add_pca32+ner_fb", "Tahap 2B: +NER AddMed & FinBERT", fg["time_ext"] + fg["bge_add_pca32"] + fg["ner_meta"] + fg["fb_soft"]),
        ("t2b:time_ext+full_fe_compact", "Tahap 2B: Full FE Compact", fg["time_ext"] + fg["full_fe_compact_text"]),
        ("t2b:time_ext+full_fe_pca32", "Tahap 2B: Full FE PCA32", fg["time_ext"] + fg["full_fe_pca32_text"]),
        ("t2b:time_ext+full_fe_raw", "Tahap 2B: Full FE Raw384", fg["time_ext"] + fg["full_fe_raw_text"]),
    ]

    fe_rows = []
    fe_results = {}
    runners = {"logreg": fit_eval_logreg, "xgb": fit_eval_xgb, "lgbm": fit_eval_lgbm}

    print("=== TAHAP 2B: ABLASI FEATURE ENGINEERING (LogReg, XGB, LGBM) ===")
    for cname, cat, cols in fe_configs:
        for kind, fn in runners.items():
            res = fn(df, cols, compute_oof=False)
            vm, tm = res["val_metrics"], res["test_metrics"]
            tau, vm_tuned = tune_threshold(y_va, res["val_prob"])
            tm_tuned = eval_prob_metrics(y_te, res["test_prob"], tau)
            label = f"{kind}:{cname.split(':', 1)[1]}"
            fe_results[(cname, kind)] = (label, cat, res, tau, vm_tuned, tm_tuned)
            fe_rows.append({
                "model": label,
                "config": cname,
                "category": cat,
                "estimator": kind,
                "n_features": res["n_features"],
                "val_DA": vm["DA"],
                "val_F1_macro": vm["F1_macro"],
                "val_AUC": vm["ROC_AUC"],
                "val_share_pred1": vm["share_pred1"],
                "test_DA": tm["DA"],
                "test_F1_macro": tm["F1_macro"],
                "test_AUC": tm["ROC_AUC"],
                "test_share_pred1": tm["share_pred1"],
                "tau_val": tau,
                "val_DA_tuned": vm_tuned["DA"],
                "val_F1_tuned": vm_tuned["F1_macro"],
                "test_DA_tuned": tm_tuned["DA"],
                "test_F1_tuned": tm_tuned["F1_macro"],
                "test_share_pred1_tuned": tm_tuned["share_pred1"],
                "hp": json.dumps(res["hp"]),
            })
        best_k = max(("logreg", "xgb", "lgbm"), key=lambda k: _score_val(fe_results[(cname, k)][2]["val_metrics"]))
        _, _, br, _, _, _ = fe_results[(cname, best_k)]
        print(
            f"  {cname:36s} | best_val={best_k:6s} | "
            f"val DA={br['val_metrics']['DA']:.4f} F1={br['val_metrics']['F1_macro']:.4f} | "
            f"test DA={br['test_metrics']['DA']:.4f} F1={br['test_metrics']['F1_macro']:.4f}"
        )

    fe_df = pd.DataFrame(fe_rows)
    fe_df.to_csv(RES / "tugas3_stage2b_fe_ablation.csv", index=False)

    # -----------------------------------------------------------------------
    # 2. Tahap 2C: Cabang Unimodal & Multimodal Fusion
    # -----------------------------------------------------------------------
    print("\n=== TAHAP 2C: UNIMODAL BRANCHES & MULTIMODAL FUSION ===")
    fusion_rows = []
    fusion_test_probs = {}
    fusion_test_preds = {}
    fusion_val_probs = {}

    def record_fusion(name: str, family: str, res_dict: dict, note: str = ""):
        vm, tm = res_dict["val_metrics"], res_dict["test_metrics"]
        tau, vm_t = tune_threshold(y_va, res_dict["val_prob"])
        tm_t = eval_prob_metrics(y_te, res_dict["test_prob"], tau)
        row = {
            "model": name,
            "family": family,
            "n_features": res_dict.get("n_features", np.nan),
            "val_DA": vm["DA"],
            "val_F1_macro": vm["F1_macro"],
            "val_AUC": vm["ROC_AUC"],
            "val_share_pred1": vm["share_pred1"],
            "test_DA": tm["DA"],
            "test_F1_macro": tm["F1_macro"],
            "test_AUC": tm["ROC_AUC"],
            "test_share_pred1": tm["share_pred1"],
            "tau_val": tau,
            "val_DA_tuned": vm_t["DA"],
            "val_F1_tuned": vm_t["F1_macro"],
            "test_DA_tuned": tm_t["DA"],
            "test_F1_tuned": tm_t["F1_macro"],
            "test_share_pred1_tuned": tm_t["share_pred1"],
            "note": note,
        }
        fusion_rows.append(row)
        fusion_val_probs[name] = res_dict["val_prob"]
        fusion_test_probs[name] = res_dict["test_prob"]
        fusion_test_preds[name] = (res_dict["test_prob"] >= 0.5).astype(int)
        fusion_test_preds[f"{name} [tuned]"] = (res_dict["test_prob"] >= tau).astype(int)
        print(
            f"  {name:48s} | val DA={vm['DA']:.4f} F1={vm['F1_macro']:.4f} | "
            f"test DA={tm['DA']:.4f} F1={tm['F1_macro']:.4f} (tuned DA={tm_t['DA']:.4f} F1={tm_t['F1_macro']:.4f})"
        )
        return row

    # 2A. Cabang Time-Series (compute_oof=True untuk Stacking)
    time_branches = {}
    for kind in ("logreg", "xgb", "lgbm"):
        r = fit_eval_tabular(df, fg["time_ext"], kind, compute_oof=True)
        bname = f"branch_time:{kind}_ext"
        time_branches[bname] = r
        record_fusion(bname, "Unimodal Time-Series", r, json.dumps(r["hp"]))

    # 2B. Cabang Teks (tanpa fitur kurs!)
    text_specs = [
        ("branch_text:logreg_bge_raw384", "logreg", fg["bge_raw384"]),
        ("branch_text:logreg_bge_soft384", "logreg", fg["bge_soft_raw384"]),
        ("branch_text:logreg_bge_ewma3_384", "logreg", fg["bge_ewma3_raw384"]),
        ("branch_text:logreg_full_compact", "logreg", fg["full_fe_compact_text"]),
        ("branch_text:logreg_full_raw", "logreg", fg["full_fe_raw_text"]),
        ("branch_text:lgbm_full_compact", "lgbm", fg["full_fe_compact_text"]),
    ]
    text_branches = {}
    for bname, kind, cols in text_specs:
        r = fit_eval_tabular(df, cols, kind, compute_oof=True)
        text_branches[bname] = r
        record_fusion(bname, "Unimodal Text", r, json.dumps(r["hp"]))

    # Cabang Teks Fine-Tuned PyTorch ProjectionHead
    for hname, cols in [
        ("branch_text:nn_head_bge_raw384", fg["bge_raw384"]),
        ("branch_text:nn_head_full_compact", fg["full_fe_compact_text"]),
    ]:
        nn_res = train_text_projection_head(
            df.loc[m_tr, cols].to_numpy(dtype=np.float32), y_tr,
            df.loc[m_va, cols].to_numpy(dtype=np.float32), y_va,
            df.loc[m_te, cols].to_numpy(dtype=np.float32),
        )
        pack = {
            "n_features": len(cols),
            "oof_train": nn_res["oof_train"],
            "val_y": y_va,
            "val_prob": nn_res["val_prob"],
            "val_metrics": eval_prob_metrics(y_va, nn_res["val_prob"]),
            "test_y": y_te,
            "test_prob": nn_res["test_prob"],
            "test_metrics": eval_prob_metrics(y_te, nn_res["test_prob"]),
        }
        text_branches[hname] = pack
        record_fusion(hname, "Unimodal Text (PyTorch Head)", pack, "hidden=64, dropout=0.45")

    # 2C. Early Fusion (Tabular terbaik di val + LSTM sekuensial)
    best_early_rows = sorted(
        [(k, v) for k, v in fe_results.items() if k[0] != "t2b:time_ext_only"],
        key=lambda kv: _score_val(kv[1][2]["val_metrics"]),
        reverse=True,
    )
    seen_early = set()
    for (cname, kind), (label, _, res, _, _, _) in best_early_rows:
        if kind in seen_early:
            continue
        seen_early.add(kind)
        record_fusion(f"early:{label}", "Early Fusion (Tabular)", res, f"config={cname}, hp={json.dumps(res['hp'])}")

    for lname, cols in [
        ("early:lstm5_full_compact", fg["time_ext"] + fg["full_fe_compact_text"]),
        ("early:lstm5_bge16_topic_fb", fg["time_ext"] + fg["bge_pca16"] + fg["topic"] + fg["fb_soft"]),
    ]:
        r_lstm = fit_eval_lstm(df, cols, seq_len=5)
        record_fusion(lname, "Early Fusion (LSTM)", r_lstm, json.dumps(r_lstm["hp"]))

    # 2D. Late Fusion (Top-2 cabang teks x Top-2 cabang time-series di val)
    top_text_names = sorted(text_branches.keys(), key=lambda n: _score_val(text_branches[n]["val_metrics"]), reverse=True)[:2]
    # Pastikan nn_head terbaik juga diikutkan kalau belum masuk top-2
    best_nn_name = max(
        [n for n in text_branches if "nn_head" in n],
        key=lambda n: _score_val(text_branches[n]["val_metrics"]),
    )
    if best_nn_name not in top_text_names:
        top_text_names.append(best_nn_name)

    top_time_names = sorted(time_branches.keys(), key=lambda n: _score_val(time_branches[n]["val_metrics"]), reverse=True)[:2]
    ctx_cols = ["k_vol_ratio_5_20", "ner_share_tier1"]

    for tname in top_text_names:
        tb = text_branches[tname]
        short_t = tname.replace("branch_text:", "")
        for mname in top_time_names:
            mb = time_branches[mname]
            short_m = mname.replace("branch_time:", "")

            for space in ("prob", "logit"):
                lw = run_late_weighted_fusion(
                    y_va, tb["val_prob"], mb["val_prob"],
                    y_te, tb["test_prob"], mb["test_prob"],
                    space=space,
                )
                lw["n_features"] = tb["n_features"] + mb["n_features"]
                record_fusion(
                    f"late_{space}:{short_t}+{short_m}",
                    f"Late Fusion ({space.title()})",
                    lw,
                    f"alpha_text={lw['alpha']:.2f}",
                )

            ls = run_late_stacking_fusion(
                df, tb["oof_train"], mb["oof_train"],
                tb["val_prob"], mb["val_prob"],
                tb["test_prob"], mb["test_prob"],
                context_cols=None,
            )
            ls["n_features"] = tb["n_features"] + mb["n_features"]
            record_fusion(f"late_stack:{short_t}+{short_m}", "Late Fusion (Stacking)", ls, f"C={ls['C']}")

            ls_ctx = run_late_stacking_fusion(
                df, tb["oof_train"], mb["oof_train"],
                tb["val_prob"], mb["val_prob"],
                tb["test_prob"], mb["test_prob"],
                context_cols=ctx_cols,
            )
            ls_ctx["n_features"] = tb["n_features"] + mb["n_features"] + len(ctx_cols)
            record_fusion(f"late_stack_ctx:{short_t}+{short_m}", "Late Fusion (Stacking+Ctx)", ls_ctx, f"C={ls_ctx['C']}")

    # 2E. Hybrid / Gated Neural Fusion (PyTorch)
    for gname, tcols in [
        ("hybrid:gated_full_compact", fg["full_fe_compact_text"]),
        ("hybrid:gated_bge_raw384", fg["bge_raw384"]),
    ]:
        r_gmu = fit_eval_gated_fusion(df, tcols, fg["time_ext"])
        record_fusion(gname, "Hybrid / Gated Fusion", r_gmu, json.dumps(r_gmu["hp"]))

    fusion_df = pd.DataFrame(fusion_rows)
    fusion_df.to_csv(RES / "tugas3_stage2c_fusion_metrics.csv", index=False)

    # -----------------------------------------------------------------------
    # 3. Seleksi Pemenang Model C (murni via val) & Tabel Ablasi Menyeluruh
    # -----------------------------------------------------------------------
    multimodal_df = fusion_df[~fusion_df["family"].str.startswith("Unimodal")].copy()
    multimodal_df["val_score"] = 0.5 * (multimodal_df["val_DA"] + multimodal_df["val_F1_macro"])
    best_row = multimodal_df.sort_values(["val_score", "val_DA", "val_AUC"], ascending=False).iloc[0]
    best_model_name = best_row["model"]
    best_tau = float(best_row["tau_val"])

    best_early_tree_row = multimodal_df[
        multimodal_df["family"].eq("Early Fusion (Tabular)")
        & (multimodal_df["model"].str.startswith("early:lgbm") | multimodal_df["model"].str.startswith("early:xgb"))
    ].sort_values(["val_score", "val_DA"], ascending=False).iloc[0]
    best_early_tree_cfg = next(
        cname for (cname, kind), (lbl, _, _, _, _, _) in fe_results.items()
        if f"early:{lbl}" == best_early_tree_row["model"]
    )
    best_early_tree_cols = dict((c, cols) for c, _, cols in fe_configs)[best_early_tree_cfg]

    print(f"\n[PEMENANG MODEL C (via Val)] {best_model_name} | val DA={best_row['val_DA']:.4f} F1={best_row['val_F1_macro']:.4f} -> "
          f"test DA={best_row['test_DA']:.4f} F1={best_row['test_F1_macro']:.4f} | "
          f"tuned (tau={best_tau:.2f}): test DA={best_row['test_DA_tuned']:.4f} F1={best_row['test_F1_tuned']:.4f}")

    # Susun tabel ablasi menyeluruh (Model A -> Model B -> Tahap 1 -> Tahap 2A -> Tahap 2B -> Tahap 2C)
    s2a_metrics = pd.read_csv(RES / "stage2a_ner_comparison_metrics.csv")
    s2a_preds = pd.read_csv(RES / "stage2a_ner_predictions_test.csv")

    prev_keep = s2a_metrics[
        ~s2a_metrics["model"].isin([
            "xgb:kurs+finbert_emb (raw_768d)",
            "xgb:kurs+bge_small (raw_384d)",
            "xgb:kurs+bge_large (raw_1024d)",
        ])
    ][["model", "category", "split", "n", "n_features", "DA", "F1_macro", "share_pred1", "note"]].copy()

    t3_add_rows = []
    # Tambahkan pemenang Tahap 2B per kategori modul (dipilih berdasar skor val terbaik)
    selected_2b_configs = [
        "t2b:time_ext_only",
        "t2b:time_ext+bge_raw384",
        "t2b:time_ext+bge_pca32+topic",
        "t2b:time_ext+bge_raw384+topic",
        "t2b:time_ext+bge_pca32+ewma_shock",
        "t2b:time_ext+bge_ewma3_pca32+shock",
        "t2b:time_ext+bge_soft_pca32+ner_fb",
        "t2b:time_ext+bge_add_pca32+ner_fb",
        "t2b:time_ext+full_fe_compact",
        "t2b:time_ext+full_fe_raw",
    ]
    selected_2b = [
        (cname, max(("logreg", "xgb", "lgbm"), key=lambda k: _score_val(fe_results[(cname, k)][2]["val_metrics"])))
        for cname in selected_2b_configs
    ]
    extra_pred_cols = {}
    for cname, kind in selected_2b:
        lbl, cat, r, tau, _, _ = fe_results[(cname, kind)]
        for sp in ("val", "test"):
            m = r[f"{sp}_metrics"]
            t3_add_rows.append({
                "model": lbl,
                "category": cat,
                "split": sp,
                "n": len(r[f"{sp}_y"]),
                "n_features": r["n_features"],
                "DA": m["DA"],
                "F1_macro": m["F1_macro"],
                "share_pred1": m["share_pred1"],
                "note": json.dumps(r["hp"]),
            })
        extra_pred_cols[lbl] = (r["test_prob"] >= 0.5).astype(int)

    # Tambahkan semua wakil Tahap 2C (Unimodal text terbaik, Early terbaik per tipe, Late terbaik, Hybrid terbaik, & Pemenang Model C)
    top_2c_names = [
        top_text_names[0],
        "early:lstm5_full_compact",
        "hybrid:gated_full_compact",
        best_early_tree_row["model"],
        best_model_name,
    ]
    # Pastikan late terbaik & early logreg terbaik juga masuk jika belum ada
    for fam in ("Early Fusion (Tabular)", "Late Fusion (Prob)", "Late Fusion (Logit)", "Late Fusion (Stacking)", "Late Fusion (Stacking+Ctx)"):
        sub = multimodal_df[multimodal_df["family"] == fam]
        if len(sub) > 0:
            n_top = sub.sort_values(["val_score", "val_DA"], ascending=False).iloc[0]["model"]
            if n_top not in top_2c_names:
                top_2c_names.append(n_top)

    for mname in dict.fromkeys(top_2c_names):
        r = fusion_df[fusion_df["model"] == mname].iloc[0]
        for sp, da_c, f1_c, sp1_c, n_len in [
            ("val", "val_DA", "val_F1_macro", "val_share_pred1", len(y_va)),
            ("test", "test_DA", "test_F1_macro", "test_share_pred1", len(y_te)),
        ]:
            t3_add_rows.append({
                "model": mname,
                "category": f"Tahap 2C: {r['family']}",
                "split": sp,
                "n": n_len,
                "n_features": int(r["n_features"]),
                "DA": float(r[da_c]),
                "F1_macro": float(r[f1_c]),
                "share_pred1": float(r[sp1_c]),
                "note": str(r["note"]),
            })
        extra_pred_cols[mname] = fusion_test_preds[mname]

    # Tambahkan baris versi Tuned Threshold untuk Pemenang Model C
    tuned_label = f"{best_model_name} [tau={best_tau:.2f}]"
    for sp, da_c, f1_c, sp1_c, n_len in [
        ("val", "val_DA_tuned", "val_F1_tuned", "val_share_pred1", len(y_va)),
        ("test", "test_DA_tuned", "test_F1_tuned", "test_share_pred1_tuned", len(y_te)),
    ]:
        sp1_val = float(eval_prob_metrics(y_va, fusion_val_probs[best_model_name], best_tau)["share_pred1"]) if sp == "val" else float(best_row[sp1_c])
        t3_add_rows.append({
            "model": tuned_label,
            "category": "Tahap 2C: Model C Winner (Threshold Tuned)",
            "split": sp,
            "n": n_len,
            "n_features": int(best_row["n_features"]),
            "DA": float(best_row[da_c]),
            "F1_macro": float(best_row[f1_c]),
            "share_pred1": sp1_val,
            "note": f"tau={best_tau:.2f}, {best_row['note']}",
        })
    extra_pred_cols[tuned_label] = fusion_test_preds[f"{best_model_name} [tuned]"]

    full_ablation = pd.concat([prev_keep, pd.DataFrame(t3_add_rows)], ignore_index=True)

    # Gabungkan prediksi test
    all_preds = s2a_preds.copy()
    for k, arr in extra_pred_cols.items():
        all_preds[k] = arr
    for name, prob in fusion_test_probs.items():
        all_preds[f"prob::{name}"] = prob

    ref_kurs = all_preds["xgb:kurs"].to_numpy()
    ref_lex = all_preds["xgb:kurs+lexicon"].to_numpy()
    ref_bge = all_preds["xgb:kurs+bge_small_best"].to_numpy()

    def _resolve_pred_col(mname: str) -> str:
        if mname in all_preds.columns:
            return mname
        if "_val_best)" in mname and "ner" not in mname:
            return mname.split(" (")[0] + "_best"
        if "(raw_" in mname:
            return mname.split(" (")[0] + "_raw"
        if mname == "xgb:kurs+finbert_sent (6d)":
            return "xgb:kurs+finbert_sent"
        if mname.startswith("xgb:kurs+finbert_") and mname.endswith("+sent"):
            return "xgb:kurs+finbert_emb+sent"
        if "bge_small_ner" in mname and "_val_best)" in mname:
            return "xgb:kurs+bge_small_ner_tier_filter_only_best"
        if mname.endswith("+ner_meta"):
            return "xgb:kurs+bge_small_ner+meta"
        if mname.endswith("+finbert_sent_ner"):
            return "xgb:kurs+bge_small_ner+finbert_sent_ner"
        if mname.startswith("xgb:kurs+finbert_sent_ner"):
            return "xgb:kurs+finbert_sent_ner"
        if "(tier_med," in mname:
            return "xgb:kurs+bge_small_ner_tier_med_pca8"
        if "(tier_strong," in mname:
            return "xgb:kurs+bge_small_ner_tier_strong_pca16"
        raise KeyError(f"Kolom prediksi tidak ditemukan untuk {mname}")

    p_k, p_l, p_b = [], [], []
    for _, row in full_ablation.iterrows():
        if row["split"] == "test":
            pr = all_preds[_resolve_pred_col(row["model"])].to_numpy()
            p_k.append(exact_mcnemar_pvalue(y_te, ref_kurs, pr))
            p_l.append(exact_mcnemar_pvalue(y_te, ref_lex, pr))
            p_b.append(exact_mcnemar_pvalue(y_te, ref_bge, pr))
        else:
            p_k.append(np.nan)
            p_l.append(np.nan)
            p_b.append(np.nan)

    full_ablation["p_mcnemar_vs_kurs"] = p_k
    full_ablation["p_mcnemar_vs_lexicon"] = p_l
    full_ablation["p_mcnemar_vs_bge_simple"] = p_b

    full_ablation.to_csv(RES / "tugas3_full_ablation_metrics.csv", index=False)
    all_preds.to_csv(RES / "tugas3_full_predictions_test.csv", index=False)

    summary = {
        "best_model_c": best_model_name,
        "best_model_c_family": str(best_row["family"]),
        "best_model_c_tau": best_tau,
        "best_model_c_val_DA": float(best_row["val_DA"]),
        "best_model_c_val_F1": float(best_row["val_F1_macro"]),
        "best_model_c_test_DA": float(best_row["test_DA"]),
        "best_model_c_test_F1": float(best_row["test_F1_macro"]),
        "best_model_c_test_DA_tuned": float(best_row["test_DA_tuned"]),
        "best_model_c_test_F1_tuned": float(best_row["test_F1_tuned"]),
        "best_early_tree_model": str(best_early_tree_row["model"]),
        "best_early_tree_config": str(best_early_tree_row["note"]),
        "best_early_tree_cols": best_early_tree_cols,
        "top_text_branches": top_text_names,
        "top_time_branches": top_time_names,
        "feature_groups_sizes": {k: len(v) for k, v in fg.items()},
    }
    (RES / "tugas3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSelesai menyimpan seluruh hasil eksperimen ke results/tugas3_*")


if __name__ == "__main__":
    main()
