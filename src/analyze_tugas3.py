"""
Modul Analisis Eksploratif Wajib Tugas 3:
  1. Visualisasi Ruang Embedding (UMAP & t-SNE):
     - Hari volatilitas pasar tinggi (>= P75) vs rendah (<= P25) pada daily_bge_small
     - Sampel 4.000 artikel pada 10 Topik Geopolitik (c-TF-IDF)
  2. Feature Importance & Interpretabilitas:
     - SHAP values (TreeExplainer / pred_contribs) pada model Early Fusion terbaik
       dan LinearExplainer / koefisien pada cabang teks terbaik, plus agregasi per grup modalitas
  3. Analisis Korelasi, Distribusi Error & Case Study Tanggal Spesifik:
     - Tanggal di mana NLP membantu (Model A salah, Model C benar) vs menambah noise
     - Korelasi fitur teks terhadap return & volatilitas, serta distribusi error per kuartil volatilitas & hari.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from config import PROCESSED, ROOT, SEED, SPLITS
from extract_embeddings import EMB_DIR
from multimodal_fusion import fit_eval_lgbm, fit_eval_logreg, fit_eval_xgb
from run_stage2b_full_experiments import build_feature_groups, load_full_dataframe

RES = ROOT / "results"
FIG_DIR = RES / "figures"

plt.rcParams["figure.dpi"] = 160
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25


def _assign_modality_group(col: str) -> str:
    if col.startswith("k_"):
        return "Time-Series (Kurs)"
    if col.startswith(("bge_s_raw_", "bge_raw_pca", "bge_soft_pca", "bge_add_pca")):
        return "BGE Semantic"
    if col.startswith("topic_"):
        return "Topic Geopolitik"
    if col.startswith(("bge_ewma", "bge_cos_shift", "n_articles_z5")) or col.endswith(("_ewma3", "_ewma5", "_diff5")):
        return "EWMA & News Shock"
    if col.startswith(("ner_", "fb_")):
        return "NER & FinBERT Sentimen"
    return "Lainnya"


# ---------------------------------------------------------------------------
# 1. Visualisasi Ruang Embedding: UMAP & t-SNE
# ---------------------------------------------------------------------------
def run_embedding_visualizations(df: pd.DataFrame) -> dict:
    print("[Analisis 1] Visualisasi UMAP & t-SNE ruang embedding ...")
    bge_cols = [c for c in df.columns if c.startswith("bge_s_raw_")]
    X_daily = df[bge_cols].to_numpy(dtype=np.float32)
    abs_ret = df["ret_pct"].abs().to_numpy()
    q25, q75 = np.quantile(abs_ret, [0.25, 0.75])

    regime = np.where(
        abs_ret >= q75, "Volatilitas Tinggi (>=P75)",
        np.where(abs_ret <= q25, "Volatilitas Rendah (<=P25)", "Sedang")
    )

    X_pca50 = PCA(n_components=50, random_state=SEED).fit_transform(StandardScaler().fit_transform(X_daily))
    umap_2d = UMAP(n_components=2, n_neighbors=20, min_dist=0.15, metric="cosine", random_state=SEED).fit_transform(X_daily)
    tsne_2d = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=SEED).fit_transform(X_pca50)

    # Hitung jarak kosinus ke sentroid global untuk menguji apakah hari volatil lebih tersebar/bergeser
    centroid = X_daily.mean(axis=0, keepdims=True)
    cos_dist = 1.0 - (X_daily @ centroid.T).ravel() / (
        np.linalg.norm(X_daily, axis=1) * np.linalg.norm(centroid) + 1e-9
    )
    high_d = cos_dist[abs_ret >= q75]
    low_d = cos_dist[abs_ret <= q25]
    u_stat, u_pval = mannwhitneyu(high_d, low_d, alternative="two-sided")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    palette = {"Volatilitas Rendah (<=P25)": "#3B82F6", "Sedang": "#CBD5E1", "Volatilitas Tinggi (>=P75)": "#EF4444"}
    order = ["Sedang", "Volatilitas Rendah (<=P25)", "Volatilitas Tinggi (>=P75)"]
    sizes = {"Sedang": 14, "Volatilitas Rendah (<=P25)": 26, "Volatilitas Tinggi (>=P75)": 30}

    for ax, coords, title in [
        (axes[0], umap_2d, "UMAP 2D — Embedding Berita Harian (Cosine)"),
        (axes[1], tsne_2d, "t-SNE 2D — Embedding Berita Harian (PCA-50)"),
    ]:
        for lbl in order:
            m = regime == lbl
            ax.scatter(
                coords[m, 0], coords[m, 1],
                c=palette[lbl], s=sizes[lbl], alpha=0.8 if lbl != "Sedang" else 0.45,
                label=f"{lbl} (n={int(m.sum())})", edgecolors="none",
            )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"Ruang Embedding BGE-Small (1.201 Hari) vs Rezim Volatilitas USD/IDR | "
        f"Mean Cosine Dist: High={high_d.mean():.4f} vs Low={low_d.mean():.4f} (Mann-Whitney p={u_pval:.4f})",
        fontsize=10.5, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_embedding_volatility_umap_tsne.png", bbox_inches="tight")
    plt.close(fig)

    # UMAP 4.000 artikel terstratifikasi berdasarkan 10 Topik Geopolitik
    topics_art = pd.read_parquet(EMB_DIR / "articles_bge_topics.parquet")
    topic_desc = pd.read_csv(RES / "stage2b_topic_descriptions.csv").set_index("topic_id")
    art_bge = np.load(EMB_DIR / "articles_bge_small.npz")["embeddings"].astype(np.float32)

    rng = np.random.RandomState(SEED)
    sample_idx = []
    for tid in range(len(topic_desc)):
        t_idx = np.where(topics_art["topic_id"].to_numpy() == tid)[0]
        take = min(400, len(t_idx))
        sample_idx.append(rng.choice(t_idx, size=take, replace=False))
    sample_idx = np.concatenate(sample_idx)

    art_umap = UMAP(n_components=2, n_neighbors=25, min_dist=0.1, metric="cosine", random_state=SEED).fit_transform(
        art_bge[sample_idx]
    )
    s_topics = topics_art["topic_id"].to_numpy()[sample_idx]

    fig, ax = plt.subplots(figsize=(11, 6.2))
    cmap = plt.get_cmap("tab10")
    for tid in range(len(topic_desc)):
        m = s_topics == tid
        kws = ", ".join(str(topic_desc.loc[tid, "top_keywords"]).split(", ")[:4])
        ax.scatter(
            art_umap[m, 0], art_umap[m, 1],
            color=cmap(tid), s=10, alpha=0.68,
            label=f"T{tid:02d}: {kws}",
        )
    ax.set_title("UMAP 2D Sampel 4.000 Artikel Berita CNBC — 10 Klaster Topik Geopolitik (c-TF-IDF)", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=True, title="Topik & Top Keywords")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_articles_topics_umap.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "q25_abs_ret": float(q25),
        "q75_abs_ret": float(q75),
        "high_vol_cos_dist": float(high_d.mean()),
        "low_vol_cos_dist": float(low_d.mean()),
        "mannwhitney_p": float(u_pval),
    }


# ---------------------------------------------------------------------------
# 2. Feature Importance & Interpretabilitas (SHAP)
# ---------------------------------------------------------------------------
def run_shap_analysis(df: pd.DataFrame, fg: dict, summary: dict) -> pd.DataFrame:
    print("[Analisis 2] Menghitung SHAP Feature Importance & kontribusi modalitas ...")
    m_eval = df["split"].isin(["val", "test"]).to_numpy()

    # Model tree Early Fusion terbaik (dari summary)
    early_name = summary.get("best_early_tree_model", "early:lgbm:time_ext+full_fe_compact")
    kind = "lgbm" if "lgbm" in early_name else "xgb"
    tree_cols = summary.get("best_early_tree_cols", fg["time_ext"] + fg["full_fe_compact_text"])

    tree_res = (fit_eval_lgbm if kind == "lgbm" else fit_eval_xgb)(df, tree_cols, compute_oof=False)
    est = tree_res["model"]
    X_eval = df.loc[m_eval, tree_cols]

    explainer = shap.TreeExplainer(est)
    shap_vals = explainer.shap_values(X_eval)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    elif shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]

    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    imp_df = pd.DataFrame({
        "feature": tree_cols,
        "modality_group": [_assign_modality_group(c) for c in tree_cols],
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp_df["share_pct"] = 100.0 * imp_df["mean_abs_shap"] / max(imp_df["mean_abs_shap"].sum(), 1e-12)
    imp_df.to_csv(RES / "tugas3_feature_importance.csv", index=False)

    # Kontribusi cabang teks LogReg terbaik untuk interpretabilitas fitur ringkas
    text_lr = fit_eval_logreg(df, fg["full_fe_compact_text"], compute_oof=False)
    coefs = np.abs(text_lr["model"].named_steps["logisticregression"].coef_[0])
    lr_imp = pd.DataFrame({
        "feature": fg["full_fe_compact_text"],
        "modality_group": [_assign_modality_group(c) for c in fg["full_fe_compact_text"]],
        "abs_logreg_coef": coefs,
    }).sort_values("abs_logreg_coef", ascending=False)
    lr_imp.to_csv(RES / "tugas3_text_branch_coef_importance.csv", index=False)

    # Plot SHAP Top 20 & Agregasi per Modalitas
    group_imp = imp_df.groupby("modality_group")["share_pct"].sum().sort_values(ascending=True)
    top20 = imp_df.head(20).iloc[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), gridspec_kw={"width_ratios": [1.45, 1.0]})
    group_palette = {
        "Time-Series (Kurs)": "#1E3A8A",
        "BGE Semantic": "#2563EB",
        "EWMA & News Shock": "#0D9488",
        "Topic Geopolitik": "#D97706",
        "NER & FinBERT Sentimen": "#DC2626",
    }
    bar_colors = [group_palette.get(g, "#64748B") for g in top20["modality_group"]]
    axes[0].barh(top20["feature"], top20["mean_abs_shap"], color=bar_colors)
    axes[0].set_title(f"Top 20 Fitur ({early_name}) — Mean |SHAP| (Val+Test)", fontsize=10.5, fontweight="bold")
    axes[0].set_xlabel("Mean |SHAP value|")

    g_colors = [group_palette.get(g, "#64748B") for g in group_imp.index]
    bars = axes[1].barh(group_imp.index, group_imp.values, color=g_colors)
    axes[1].set_title("Kontribusi Total per Grup Modalitas (%)", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("Share of Total |SHAP| (%)")
    axes[1].set_xlim(0, max(group_imp.values) * 1.18)
    for b in bars:
        w = b.get_width()
        axes[1].text(w + 0.6, b.get_y() + b.get_height() / 2, f"{w:.1f}%", va="center", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_shap_importance.png", bbox_inches="tight")
    plt.close(fig)
    return imp_df


# ---------------------------------------------------------------------------
# 3. Analisis Korelasi, Distribusi Error & Case Study Tanggal Spesifik
# ---------------------------------------------------------------------------
def run_error_and_correlation_analysis(df: pd.DataFrame, fg: dict, summary: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("[Analisis 3] Korelasi fitur, distribusi error, dan case study tanggal test ...")
    preds_df = pd.read_csv(RES / "tugas3_full_predictions_test.csv", parse_dates=["date"])
    topics_art = pd.read_parquet(EMB_DIR / "articles_bge_topics.parquet")
    topics_art["target_date"] = pd.to_datetime(topics_art["target_date"])
    topic_desc = pd.read_csv(RES / "stage2b_topic_descriptions.csv").set_index("topic_id")
    ner_art = pd.read_parquet(EMB_DIR / "articles_ner_tiers.parquet")
    test_arts = pd.read_csv(SPLITS / "test_articles.csv", encoding="utf-8-sig", low_memory=False)
    test_arts["target_date"] = pd.to_datetime(test_arts["target_date"])

    # Sinkronkan tier NER pada split test
    ner_test = ner_art[ner_art["split"] == "test"].reset_index(drop=True)
    if len(ner_test) == len(test_arts):
        test_arts["tier"] = ner_test["tier"].to_numpy()
    else:
        test_arts["tier"] = 0

    best_c = summary["best_model_c"]
    prob_col = f"prob::{best_c}"
    test_base = df[df["split"] == "test"].reset_index(drop=True)
    assert (test_base["date"] == preds_df["date"]).all()

    y = preds_df["y"].to_numpy(dtype=int)
    p_a = preds_df["xgb:kurs"].to_numpy(dtype=int)
    p_b = preds_df["xgb:kurs+lexicon"].to_numpy(dtype=int)
    p_c = preds_df[best_c].to_numpy(dtype=int)
    prob_c = preds_df[prob_col].to_numpy(dtype=float)

    # Topik dominan per hari di test
    topic_cols = [c for c in fg["topic"] if c.startswith("topic_share_")]
    dom_topic_idx = np.argmax(test_base[topic_cols].to_numpy(), axis=1)

    case_df = pd.DataFrame({
        "date": test_base["date"],
        "kurs": test_base["kurs"],
        "ret_pct": test_base["ret_pct"],
        "abs_ret_pct": test_base["ret_pct"].abs(),
        "y": y,
        "pred_model_a_kurs": p_a,
        "pred_model_b_lex": p_b,
        "pred_model_c": p_c,
        "prob_model_c": prob_c,
        "a_correct": (p_a == y),
        "c_correct": (p_c == y),
        "n_articles": test_base["n_articles"],
        "fb_soft_net_mean": test_base["fb_soft_net_mean"],
        "ner_share_tier1": test_base["ner_share_tier1"],
        "dom_topic_id": dom_topic_idx,
        "dom_topic_keywords": [topic_desc.loc[t, "top_keywords"] for t in dom_topic_idx],
    })

    def _top_headlines(d):
        sub = test_arts[test_arts["target_date"] == d].sort_values("tier", ascending=False).head(3)
        return " || ".join([f"[T{int(r.tier)}] {r.text_title}" for r in sub.itertuples()])

    helped = case_df[(~case_df["a_correct"]) & case_df["c_correct"]].sort_values("abs_ret_pct", ascending=False).head(8).copy()
    helped["case_type"] = "NLP_Menyelamatkan (A salah, C benar)"
    hurt = case_df[case_df["a_correct"] & (~case_df["c_correct"])].sort_values("abs_ret_pct", ascending=False).head(8).copy()
    hurt["case_type"] = "NLP_Menambah_Noise (A benar, C salah)"

    cases_out = pd.concat([helped, hurt], ignore_index=True)
    cases_out["top_headlines"] = cases_out["date"].map(_top_headlines)
    cases_out.to_csv(RES / "tugas3_error_case_studies.csv", index=False)

    # Distribusi error berdasarkan kuartil volatilitas, hari (Senin vs lainnya), & arah aktual
    case_df["vol_quartile"] = pd.qcut(case_df["abs_ret_pct"], q=4, labels=["Q1 (Tenang)", "Q2", "Q3", "Q4 (Volatil)"])
    case_df["day_group"] = np.where(case_df["date"].dt.dayofweek == 0, "Senin (Akumulasi Wknd)", "Selasa-Jumat")
    case_df["actual_dir"] = np.where(case_df["y"] == 1, "y=1 (IDR Melemah)", "y=0 (IDR Menguat/Tetap)")

    bd_rows = []
    for dim_col in ("vol_quartile", "day_group", "actual_dir"):
        for val_lbl, grp in case_df.groupby(dim_col, observed=False):
            bd_rows.append({
                "dimension": dim_col,
                "segment": str(val_lbl),
                "n": len(grp),
                "DA_ModelA_kurs": float(grp["a_correct"].mean()),
                "DA_ModelB_lex": float((grp["pred_model_b_lex"] == grp["y"]).mean()),
                "DA_ModelC": float(grp["c_correct"].mean()),
                "delta_C_vs_A": float(grp["c_correct"].mean() - grp["a_correct"].mean()),
            })
    bd_df = pd.DataFrame(bd_rows)
    bd_df.to_csv(RES / "tugas3_error_breakdown.csv", index=False)

    # Korelasi fitur ringkas NLP terhadap ret_pct dan |ret_pct| di seluruh 1.201 hari
    corr_features = (
        ["t_compound_mean", "t_lm_uncertainty_mean", "fb_soft_net_mean", "fb_soft_share_neg",
         "ner_share_tier1", "ner_count_tier1_log", "bge_cos_shift_1d", "bge_cos_shift_5d",
         "n_articles_z5", "topic_entropy"]
        + topic_cols[:5]
    )
    corr_rows = []
    for fc in corr_features:
        s = df[fc].astype(float)
        r_ret, p_ret = pearsonr(s, df["ret_pct"])
        r_vol, p_vol = pearsonr(s, df["ret_pct"].abs())
        sr_vol, _ = spearmanr(s, df["ret_pct"].abs())
        corr_rows.append({
            "feature": fc,
            "pearson_ret": r_ret,
            "pval_ret": p_ret,
            "pearson_abs_ret": r_vol,
            "pval_abs_ret": p_vol,
            "spearman_abs_ret": sr_vol,
        })
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(RES / "tugas3_nlp_kurs_correlations.csv", index=False)

    # Plot 3-panel analisis error & korelasi
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # Panel 1: Korelasi dengan |ret_pct| (volatilitas) & ret_pct
    top_c = corr_df.reindex(corr_df["pearson_abs_ret"].abs().sort_values(ascending=False).index).head(10).iloc[::-1]
    y_pos = np.arange(len(top_c))
    axes[0].barh(y_pos - 0.18, top_c["pearson_abs_ret"], height=0.35, color="#EF4444", label="Korelasi |ret_pct| (Volatilitas)")
    axes[0].barh(y_pos + 0.18, top_c["pearson_ret"], height=0.35, color="#2563EB", label="Korelasi ret_pct (Arah)")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(top_c["feature"], fontsize=8)
    axes[0].axvline(0, color="black", lw=0.7)
    axes[0].set_title("Korelasi Fitur NLP vs Kurs & Volatilitas", fontsize=10, fontweight="bold")
    axes[0].legend(fontsize=7.5)

    # Panel 2: DA per Kuartil Volatilitas
    vq = bd_df[bd_df["dimension"] == "vol_quartile"]
    x_q = np.arange(len(vq))
    w = 0.25
    axes[1].bar(x_q - w, vq["DA_ModelA_kurs"], width=w, color="#94A3B8", label="Model A (xgb:kurs)")
    axes[1].bar(x_q, vq["DA_ModelB_lex"], width=w, color="#F59E0B", label="Model B (lexicon)")
    axes[1].bar(x_q + w, vq["DA_ModelC"], width=w, color="#10B981", label="Model C (Winner)")
    axes[1].set_xticks(x_q)
    axes[1].set_xticklabels(vq["segment"], fontsize=8)
    axes[1].set_ylim(0.3, 0.85)
    axes[1].axhline(0.5, color="red", ls="--", lw=0.8)
    axes[1].set_title("Akurasi per Kuartil Volatilitas (Test)", fontsize=10, fontweight="bold")
    axes[1].legend(fontsize=7.5)

    # Panel 3: Keunggulan kumulatif (Model C - Model A) sepanjang 241 hari test
    cum_diff = np.cumsum(case_df["c_correct"].astype(int) - case_df["a_correct"].astype(int))
    axes[2].plot(case_df["date"], cum_diff, color="#10B981", lw=2.0)
    axes[2].fill_between(case_df["date"], 0, cum_diff, where=(cum_diff >= 0), color="#10B981", alpha=0.18)
    axes[2].fill_between(case_df["date"], 0, cum_diff, where=(cum_diff < 0), color="#EF4444", alpha=0.18)
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_title(f"Selisih Prediksi Benar Kumulatif (C - A = +{int(cum_diff.iloc[-1])})", fontsize=10, fontweight="bold")
    axes[2].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_error_and_correlation.png", bbox_inches="tight")
    plt.close(fig)
    return cases_out, bd_df, corr_df


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads((RES / "tugas3_summary.json").read_text(encoding="utf-8"))
    df = load_full_dataframe()
    fg = build_feature_groups(df)

    umap_stats = run_embedding_visualizations(df)
    imp_df = run_shap_analysis(df, fg, summary)
    cases_out, bd_df, corr_df = run_error_and_correlation_analysis(df, fg, summary)

    print("\n=== RINGKASAN ANALISIS WAJIB ===")
    print("UMAP/Cosine Volatilitas:", umap_stats)
    print("\nTop 8 Fitur SHAP:")
    print(imp_df.head(8).round(4).to_string(index=False))
    print("\nBreakdown Error (Test):")
    print(bd_df.round(4).to_string(index=False))
    print(f"\nCase Studies tersimpan: {len(cases_out)} tanggal di results/tugas3_error_case_studies.csv")


if __name__ == "__main__":
    main()
