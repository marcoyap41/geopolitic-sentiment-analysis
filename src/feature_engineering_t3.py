"""
Tahap 2B Tugas 3 - Feature Engineering Lanjutan di atas Cache `BAAI/bge-small-en-v1.5`.

Memakai langsung cache yang sudah diekstrak (tanpa menjalankan ulang inferensi Transformer/NER):
  * data/processed/embeddings/articles_bge_small.npz  (41.406 x 384)
  * data/processed/embeddings/daily_bge_small.parquet   (1.201 x 384 float32)
  * data/processed/embeddings/articles_finbert.npz    (sentiment_probs 41.406 x 3)
  * data/processed/embeddings/articles_ner_tiers.parquet

Membangun 4 blok fitur anti-leakage (semua fitting HANYA pada split `train`):
  1. Topic Modeling BERTopic-style (MiniBatchKMeans 10 klaster + c-TF-IDF) -> 11 fitur topik.
  2. Extended Time-Series Features (`X_time_ext`) -> 22 fitur teknikal & kalender (semua <= T-1).
  3. NER-Weighted Agregasi Moderat (`tier_soft` & `additive_med`) + FinBERT Sentimen NER-Weighted.
  4. EWMA Sliding Window (`halflife=3, 5`), PCA (k=16, 32), dan News Shock / Temporal Dynamics.

Output: data/processed/features_tugas3_full.parquet (+ cache per blok di embeddings/).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

from config import DAILY_BASE, FEATURES_DAILY, PROCESSED, ROOT, SEED
from dataset import kurs_features
from extract_embeddings import EMB_DIR, load_articles_and_base, save_daily_parquet
from ner_weighting import aggregate_daily_weighted, compute_article_weights, compute_daily_ner_meta

RES = ROOT / "results"
TOPICS_ART_CACHE = EMB_DIR / "articles_bge_topics.parquet"
TOPICS_DESC_CSV = RES / "stage2b_topic_descriptions.csv"
FEATURES_T3_FULL = PROCESSED / "features_tugas3_full.parquet"
N_TOPICS = 10


# ---------------------------------------------------------------------------
# 1. Topic Modeling (MiniBatchKMeans pada bge-small + c-TF-IDF, fit di train)
# ---------------------------------------------------------------------------
def build_topic_features(arts: pd.DataFrame, base: pd.DataFrame, art_bge: np.ndarray) -> pd.DataFrame:
    print("[FE-1] Topic Modeling (MiniBatchKMeans k=10 + c-TF-IDF, fit di train) ...")
    norms = np.linalg.norm(art_bge, axis=1, keepdims=True)
    emb_unit = (art_bge / np.maximum(norms, 1e-9)).astype(np.float32)

    is_train_art = arts["split"].eq("train").to_numpy()
    km = MiniBatchKMeans(
        n_clusters=N_TOPICS,
        random_state=SEED,
        n_init=20,
        batch_size=2048,
        max_iter=300,
    ).fit(emb_unit[is_train_art])

    topic_ids = km.predict(emb_unit)

    # c-TF-IDF di atas clean_title split train untuk label kata kunci tiap topik
    train_texts = arts.loc[is_train_art, "clean_title"].fillna("")
    train_topics = topic_ids[is_train_art]
    class_docs = [
        " ".join(train_texts[train_topics == c].tolist())
        for c in range(N_TOPICS)
    ]
    vec = CountVectorizer(ngram_range=(1, 2), min_df=2, max_features=8000)
    tf_mat = vec.fit_transform(class_docs).toarray().astype(np.float64)
    vocab = np.array(vec.get_feature_names_out())

    tf_per_word = np.maximum(tf_mat.sum(axis=0), 1.0)
    avg_words_per_class = float(tf_mat.sum() / N_TOPICS)
    idf = np.log(1.0 + avg_words_per_class / tf_per_word)
    ctfidf = (tf_mat / np.maximum(tf_mat.sum(axis=1, keepdims=True), 1.0)) * idf

    desc_rows = []
    for c in range(N_TOPICS):
        top_idx = np.argsort(ctfidf[c])[::-1][:8]
        kws = ", ".join(vocab[top_idx])
        desc_rows.append({
            "topic_id": c,
            "top_keywords": kws,
            "train_articles": int(np.sum(train_topics == c)),
            "train_share": float(np.mean(train_topics == c)),
            "val_share": float(np.mean(topic_ids[arts["split"].eq("val").to_numpy()] == c)),
            "test_share": float(np.mean(topic_ids[arts["split"].eq("test").to_numpy()] == c)),
        })
        print(f"  Topic {c:02d} ({desc_rows[-1]['train_share']:.1%}): {kws}")

    RES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(desc_rows).to_csv(TOPICS_DESC_CSV, index=False)

    pd.DataFrame({
        "url": arts["url"].values,
        "target_date": arts["target_date"].values,
        "split": arts["split"].values,
        "topic_id": topic_ids,
    }).to_parquet(TOPICS_ART_CACHE, index=False)

    # Agregasi harian proporsi topik + entropi topik
    date_to_idx = {d: i for i, d in enumerate(base["date"])}
    day_idx = arts["target_date"].map(date_to_idx).to_numpy(dtype=np.int64)
    n_days = len(base)
    day_counts = np.bincount(day_idx, minlength=n_days).astype(np.float64)

    topic_df = pd.DataFrame(index=base.index)
    probs = np.zeros((n_days, N_TOPICS), dtype=np.float64)
    for c in range(N_TOPICS):
        c_cnt = np.bincount(day_idx, weights=(topic_ids == c).astype(np.float64), minlength=n_days)
        probs[:, c] = c_cnt / np.maximum(day_counts, 1.0)
        topic_df[f"topic_share_{c:02d}"] = probs[:, c]

    entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
    topic_df["topic_entropy"] = entropy
    return topic_df


# ---------------------------------------------------------------------------
# 2. Extended Time-Series Features (X_time_ext, semua <= T-1 anti-leakage)
# ---------------------------------------------------------------------------
def build_extended_time_features(base: pd.DataFrame) -> pd.DataFrame:
    print("[FE-2] Membangun 22 fitur Time-Series anti-leakage (<= T-1) ...")
    f = kurs_features(base).copy()  # 7 fitur dasar: k_ret_lag1..5, k_mean5, k_std5

    k_prev = base["kurs"].shift(1)
    r_prev = base["ret_pct"].shift(1)

    for span in (5, 10, 20):
        ema = k_prev.ewm(span=span, adjust=False, min_periods=span).mean()
        f[f"k_ema{span}_ratio"] = (k_prev / ema) - 1.0

    ema5 = k_prev.ewm(span=5, adjust=False, min_periods=5).mean()
    ema20 = k_prev.ewm(span=20, adjust=False, min_periods=20).mean()
    f["k_ema_cross_5_20"] = (ema5 - ema20) / (ema20 + 1e-9)

    # RSI(14) di atas kurs_{T-1}
    delta = k_prev.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    loss = (-delta).clip(lower=0.0).ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(loss != 0.0, np.where(gain > 0.0, 100.0, 50.0))
    f["k_rsi14"] = rsi

    # MACD histogram dinormalisasi oleh k_prev
    ema12 = k_prev.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = k_prev.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    f["k_macd_hist"] = (macd - signal) / (k_prev + 1e-9)

    f["k_std10"] = r_prev.rolling(10, min_periods=5).std()
    f["k_std20"] = r_prev.rolling(20, min_periods=10).std()
    f["k_vol_ratio_5_20"] = f["k_std5"] / (f["k_std20"] + 1e-6)

    f["k_mom5"] = (k_prev / base["kurs"].shift(6)) - 1.0
    f["k_mom10"] = (k_prev / base["kurs"].shift(11)) - 1.0

    # ARIMA(1,0,1) one-step-ahead forecast (fit HANYA pada train, anti-leakage)
    ret = base["ret_pct"].to_numpy()
    is_train = base["split"].eq("train").to_numpy()
    arima_fit = ARIMA(ret[is_train], order=(1, 0, 1)).fit()
    arima_applied = arima_fit.apply(ret, refit=False)
    f["k_arima_pred"] = arima_applied.get_prediction(start=0).predicted_mean

    # Fitur kalender (hari Senin menampung akumulasi berita Jumat-Minggu)
    dow = base["date"].dt.dayofweek
    f["k_is_monday"] = (dow == 0).astype(float)
    f["k_dow_sin"] = np.sin(2.0 * np.pi * dow / 5.0)
    f["k_dow_cos"] = np.cos(2.0 * np.pi * dow / 5.0)
    return f


# ---------------------------------------------------------------------------
# 3. PCA Anti-Leakage Helper & Weighted FinBERT
# ---------------------------------------------------------------------------
def fit_train_pca(mat: np.ndarray, is_train: np.ndarray, prefix: str, dims=(16, 32)) -> pd.DataFrame:
    scaler = StandardScaler().fit(mat[is_train])
    Xs = scaler.transform(mat)
    blocks = []
    for k in dims:
        pca = PCA(n_components=k, random_state=SEED).fit(Xs[is_train])
        Z = pca.transform(Xs)
        cols = [f"{prefix}_pca{k}_{i:02d}" for i in range(k)]
        blocks.append(pd.DataFrame(Z, columns=cols))
    return pd.concat(blocks, axis=1)


def compute_weighted_finbert_sent(
    art_probs: np.ndarray, target_dates: pd.Series, base_dates: pd.Series, norm_weights: np.ndarray, prefix: str = "fb_soft"
) -> pd.DataFrame:
    w_probs, _ = aggregate_daily_weighted(art_probs, target_dates, base_dates, norm_weights)
    date_to_idx = {d: i for i, d in enumerate(pd.to_datetime(base_dates))}
    day_idx = pd.to_datetime(target_dates).map(date_to_idx).to_numpy(dtype=np.int64)
    n_days = len(base_dates)

    share_neg = np.bincount(day_idx, weights=(art_probs[:, 1] > 0.5).astype(np.float64) * norm_weights, minlength=n_days)
    share_pos = np.bincount(day_idx, weights=(art_probs[:, 0] > 0.5).astype(np.float64) * norm_weights, minlength=n_days)

    return pd.DataFrame({
        f"{prefix}_pos_mean": w_probs[:, 0],
        f"{prefix}_neg_mean": w_probs[:, 1],
        f"{prefix}_neu_mean": w_probs[:, 2],
        f"{prefix}_net_mean": w_probs[:, 0] - w_probs[:, 1],
        f"{prefix}_share_neg": share_neg,
        f"{prefix}_share_pos": share_pos,
    })


def row_cosine_shift(mat_a: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(mat_a, axis=1)
    nb = np.linalg.norm(mat_b, axis=1)
    dot = np.sum(mat_a * mat_b, axis=1)
    cos = dot / np.maximum(na * nb, 1e-9)
    return np.clip(1.0 - cos, 0.0, 2.0)


def main():
    t0 = time.time()
    arts, base = load_articles_and_base()
    is_train = base["split"].eq("train").to_numpy()

    art_bge = np.load(EMB_DIR / "articles_bge_small.npz")["embeddings"].astype(np.float32)
    art_fb_probs = np.load(EMB_DIR / "articles_finbert.npz")["sentiment_probs"].astype(np.float32)
    ner_df = pd.read_parquet(EMB_DIR / "articles_ner_tiers.parquet")
    ner_df["target_date"] = pd.to_datetime(ner_df["target_date"])
    daily_bge_df = pd.read_parquet(EMB_DIR / "daily_bge_small.parquet")
    bge_raw_mat = daily_bge_df.drop(columns=["date"]).to_numpy(dtype=np.float32)

    daily_nlp = pd.read_csv(FEATURES_DAILY, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    assert (daily_nlp["date"] == base["date"]).all(), "Urutan tanggal daily_nlp tidak sama dengan base!"

    # 1. Topic Modeling
    topic_df = build_topic_features(arts, base, art_bge)

    # 2. Extended Time-Series Features
    time_ext_df = build_extended_time_features(base)

    # 3. NER Weighting Moderat (tier_soft & additive_med) + FinBERT Sentimen NER
    print("[FE-3] NER-Weighted Agregasi Moderat (tier_soft & additive_med) + FinBERT Sentimen ...")
    w_soft = compute_article_weights(ner_df, b1=1.0, b2=0.5, mode="hierarchical", t0_weight=1.0)
    bge_soft_mat, norm_w_soft = aggregate_daily_weighted(art_bge, ner_df["target_date"], base["date"], w_soft)
    save_daily_parquet(bge_soft_mat, base, "bge_soft_raw", EMB_DIR / "daily_bge_ner_soft.parquet")

    w_add = compute_article_weights(ner_df, b1=2.0, b2=1.0, mode="additive", t0_weight=1.0)
    bge_add_mat, _ = aggregate_daily_weighted(art_bge, ner_df["target_date"], base["date"], w_add)
    save_daily_parquet(bge_add_mat, base, "bge_add_raw", EMB_DIR / "daily_bge_ner_addmed.parquet")

    ner_meta_df = compute_daily_ner_meta(ner_df, base["date"], norm_w_soft)
    fb_soft_df = compute_weighted_finbert_sent(art_fb_probs, ner_df["target_date"], base["date"], norm_w_soft, prefix="fb_soft")

    # 4. EWMA Sliding Window & News Shock
    print("[FE-4] EWMA Sliding Window (halflife=3, 5) + News Shock & PCA anti-leakage ...")
    bge_df_num = pd.DataFrame(bge_raw_mat)
    bge_ewma3_mat = bge_df_num.ewm(halflife=3, adjust=True).mean().to_numpy(dtype=np.float32)
    bge_ewma5_mat = bge_df_num.ewm(halflife=5, adjust=True).mean().to_numpy(dtype=np.float32)
    save_daily_parquet(bge_ewma3_mat, base, "bge_ewma3_raw", EMB_DIR / "daily_bge_ewma3.parquet")
    save_daily_parquet(bge_ewma5_mat, base, "bge_ewma5_raw", EMB_DIR / "daily_bge_ewma5.parquet")

    pca_blocks = [
        fit_train_pca(bge_raw_mat, is_train, "bge_raw", dims=(16, 32)),
        fit_train_pca(bge_ewma3_mat, is_train, "bge_ewma3", dims=(16, 32)),
        fit_train_pca(bge_ewma5_mat, is_train, "bge_ewma5", dims=(16, 32)),
        fit_train_pca(bge_soft_mat, is_train, "bge_soft", dims=(16, 32)),
        fit_train_pca(bge_add_mat, is_train, "bge_add", dims=(16, 32)),
    ]

    # News Shock & Temporal Dynamics
    shock_df = pd.DataFrame(index=base.index)
    prev_1d = np.vstack([bge_raw_mat[:1], bge_raw_mat[:-1]])
    prev_ewma5 = np.vstack([bge_ewma5_mat[:1], bge_ewma5_mat[:-1]])
    shock_df["bge_cos_shift_1d"] = row_cosine_shift(bge_raw_mat, prev_1d)
    shock_df["bge_cos_shift_5d"] = row_cosine_shift(bge_raw_mat, prev_ewma5)
    shock_df.loc[0, ["bge_cos_shift_1d", "bge_cos_shift_5d"]] = 0.0

    n_art = daily_nlp["n_articles"].astype(float)
    n_mean5 = n_art.shift(1).rolling(5, min_periods=1).mean().fillna(n_art.iloc[0])
    n_std5 = n_art.shift(1).rolling(5, min_periods=2).std().fillna(1.0)
    shock_df["n_articles_z5"] = (n_art - n_mean5) / (n_std5 + 1.0)

    dyn_sources = {
        "fb_soft_net": fb_soft_df["fb_soft_net_mean"],
        "ner_share_t1": ner_meta_df["ner_share_tier1"],
        "vader_comp": daily_nlp["t_compound_mean"],
        "lm_uncert": daily_nlp["t_lm_uncertainty_mean"],
        "topic_ent": topic_df["topic_entropy"],
    }
    for sname, series in dyn_sources.items():
        shock_df[f"{sname}_ewma3"] = series.ewm(halflife=3, adjust=True).mean()
        shock_df[f"{sname}_ewma5"] = series.ewm(halflife=5, adjust=True).mean()

    fb_net_roll5 = fb_soft_df["fb_soft_net_mean"].shift(1).rolling(5, min_periods=1).mean().fillna(
        fb_soft_df["fb_soft_net_mean"].iloc[0]
    )
    shock_df["fb_soft_net_diff5"] = fb_soft_df["fb_soft_net_mean"] - fb_net_roll5

    # Gabungkan semua fitur ke satu tabel terstruktur
    full_df = pd.concat(
        [
            base[["date", "kurs", "ret", "ret_pct", "y", "split"]].reset_index(drop=True),
            time_ext_df.reset_index(drop=True),
            daily_nlp.drop(columns=["date"]).reset_index(drop=True),
            topic_df.reset_index(drop=True),
            ner_meta_df.reset_index(drop=True),
            fb_soft_df.reset_index(drop=True),
            shock_df.reset_index(drop=True),
            *[b.reset_index(drop=True) for b in pca_blocks],
        ],
        axis=1,
    )
    full_df.to_parquet(FEATURES_T3_FULL, index=False)

    non_lag_nans = int(full_df.drop(columns=[c for c in full_df.columns if c.startswith("k_")]).isna().sum().sum())
    val_test_nans = int(full_df[full_df["split"].isin(["val", "test"])].isna().sum().sum())
    print(f"\n[FE] Selesai dalam {time.time() - t0:.1f}s -> {FEATURES_T3_FULL}")
    print(f"  shape={full_df.shape} | non-lag NaNs={non_lag_nans} | val/test NaNs={val_test_nans}")


if __name__ == "__main__":
    main()
