"""
Tahap 1 Tugas 3 - Ekstraksi Fitur Teks (3 Kategori Embedding) + Reduksi Dimensi Anti-Leakage.

Kategori yang diekstrak dari judul artikel (`text_title` / `clean_title`):
  1. Static Embedding (Kategori A):
     - `fasttext`: Gensim FastText Skip-gram (100D) dilatih HANYA di split `train`,
       lalu simple mean pooling per artikel dan per hari target.
  2. Contextual / Transformer Domain-Specific (Kategori B):
     - `finbert` (`ProsusAI/finbert`): dalam 1 forward pass menghasilkan
       `finbert_emb` (768D attention-masked mean pooling + L2-norm) dan
       `finbert_sent` (6 fitur harian dari probabilitas pos/neg/neu).
  3. Contextual / Transformer General-Purpose (Kategori C):
     - `bge_small` (`BAAI/bge-small-en-v1.5`, 384D): inferensi lokal dengan CLS pooling + L2-norm.
     - `bge_large` (`baai/bge-large-en-v1.5`, 1024D): via OpenRouter API dengan thread pool,
       exponential backoff retry, dan incremental checkpointing.

Semua agregasi harian menggunakan simple mean pooling per `target_date` (tanpa weighted pooling).
Reduksi dimensi (`StandardScaler` + `PCA` untuk k in {8, 16, 32, 50}) di-fit HANYA pada 720 hari `train`.
"""
import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import DAILY_BASE, PROCESSED, ROOT, SEED, SPLITS

EMB_DIR = PROCESSED / "embeddings"
FEATURES_EMB_DAILY = PROCESSED / "features_embeddings_daily.csv"
PCA_DIMS = (8, 16, 32, 50)
OPENROUTER_EMBED_URL = "https://openrouter.ai/api/v1/embeddings"


def load_env_key(env_path: Path = ROOT / ".env", key_name: str = "OPENROUTER_API_KEY") -> str:
    """Ambil API key dari environment variable atau file `.env` tanpa hardcoding."""
    val = os.environ.get(key_name, "").strip()
    if val:
        return val
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == key_name:
                return v.strip().strip('"').strip("'")
    raise RuntimeError(f"{key_name} tidak ditemukan di environment maupun {env_path}")


def load_articles_and_base() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Muat seluruh artikel dari split train/val/test dan tabel hari kerja target."""
    base = pd.read_csv(DAILY_BASE, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    parts = []
    for split_name in ("train", "val", "test"):
        df = pd.read_csv(SPLITS / f"{split_name}_articles.csv", encoding="utf-8-sig", low_memory=False)
        parts.append(df)
    arts = pd.concat(parts, ignore_index=True)
    arts["target_date"] = pd.to_datetime(arts["target_date"])
    arts["text_title"] = arts["text_title"].fillna("").astype(str)
    arts["clean_title"] = arts["clean_title"].fillna("").astype(str)

    valid_dates = set(base["date"])
    missing = set(arts["target_date"]) - valid_dates
    if missing:
        raise ValueError(f"Terdapat {len(missing)} target_date artikel yang tidak ada di daily_base.csv")
    return arts, base


def aggregate_daily_mean(vectors: np.ndarray, target_dates: pd.Series, base_dates: pd.Series) -> np.ndarray:
    """Simple mean pooling per `target_date` sejajar urutan `base_dates`."""
    date_to_idx = {d: i for i, d in enumerate(base_dates)}
    day_idx = target_dates.map(date_to_idx).to_numpy(dtype=np.int64)
    n_days, dim = len(base_dates), vectors.shape[1]

    sums = np.zeros((n_days, dim), dtype=np.float64)
    counts = np.zeros(n_days, dtype=np.int64)
    np.add.at(sums, day_idx, vectors.astype(np.float64, copy=False))
    np.add.at(counts, day_idx, 1)

    if np.any(counts == 0):
        zero_days = int(np.sum(counts == 0))
        print(f"  [Peringatan] {zero_days} hari kerja memiliki 0 artikel; diisi vektor nol.")
    means = sums / np.maximum(counts[:, None], 1)
    return means.astype(np.float32)


def save_daily_parquet(mat: np.ndarray, base: pd.DataFrame, prefix: str, out_path: Path) -> None:
    cols = [f"{prefix}_{i:04d}" for i in range(mat.shape[1])]
    df = pd.DataFrame(mat, columns=cols)
    df.insert(0, "date", base["date"].values)
    df.to_parquet(out_path, index=False)


# ---------------------------------------------------------------------------
# 1. Kategori A: Static Embedding - FastText (latih HANYA di split train)
# ---------------------------------------------------------------------------
def extract_fasttext(arts: pd.DataFrame, base: pd.DataFrame, force: bool = False) -> np.ndarray:
    art_cache = EMB_DIR / "articles_fasttext.npz"
    daily_cache = EMB_DIR / "daily_fasttext.parquet"

    if daily_cache.exists() and art_cache.exists() and not force:
        print("[FastText] Memuat cache harian dari", daily_cache.name)
        df = pd.read_parquet(daily_cache)
        return df.drop(columns=["date"]).to_numpy(dtype=np.float32)

    if art_cache.exists() and not force:
        print("[FastText] Memuat cache artikel dari", art_cache.name)
        art_vecs = np.load(art_cache)["embeddings"].astype(np.float32)
    else:
        from gensim.models import FastText

        print("[FastText] Melatih FastText Skip-gram (100D) HANYA pada split train ...")
        t0 = time.time()
        train_mask = arts["split"].eq("train").to_numpy()
        train_tokens = [txt.split() for txt in arts.loc[train_mask, "clean_title"]]
        train_tokens = [toks for toks in train_tokens if toks]

        ft_model = FastText(
            sentences=train_tokens,
            vector_size=100,
            sg=1,
            window=5,
            min_count=2,
            min_n=3,
            max_n=6,
            epochs=15,
            workers=4,
            seed=SEED,
        )
        wv = ft_model.wv
        dim = ft_model.vector_size

        all_tokens = [txt.split() for txt in arts["clean_title"]]
        art_vecs = np.zeros((len(arts), dim), dtype=np.float32)
        empty_cnt = 0
        for i, toks in enumerate(all_tokens):
            vecs = [wv[w] for w in toks if w in wv]
            if vecs:
                art_vecs[i] = np.mean(vecs, axis=0)
            else:
                empty_cnt += 1
        print(f"  Selesai dalam {time.time() - t0:.1f}s | artikel kosong={empty_cnt}/{len(arts)}")
        np.savez_compressed(art_cache, embeddings=art_vecs.astype(np.float16))

    daily_mat = aggregate_daily_mean(art_vecs, arts["target_date"], base["date"])
    save_daily_parquet(daily_mat, base, "ft_raw", daily_cache)
    return daily_mat


# ---------------------------------------------------------------------------
# 2. Kategori B: FinBERT (Embedding 768D + Sentimen 3D dalam 1 forward pass)
# ---------------------------------------------------------------------------
def extract_finbert(
    arts: pd.DataFrame, base: pd.DataFrame, batch_size: int = 128, max_length: int = 48, force: bool = False
) -> tuple[np.ndarray, pd.DataFrame]:
    art_cache = EMB_DIR / "articles_finbert.npz"
    daily_emb_cache = EMB_DIR / "daily_finbert_emb.parquet"
    daily_sent_cache = EMB_DIR / "daily_finbert_sent.parquet"

    if daily_emb_cache.exists() and daily_sent_cache.exists() and art_cache.exists() and not force:
        print("[FinBERT] Memuat cache harian dari", daily_emb_cache.name, "&", daily_sent_cache.name)
        emb_df = pd.read_parquet(daily_emb_cache)
        sent_df = pd.read_parquet(daily_sent_cache)
        return emb_df.drop(columns=["date"]).to_numpy(dtype=np.float32), sent_df

    if art_cache.exists() and not force:
        print("[FinBERT] Memuat cache artikel dari", art_cache.name)
        npz = np.load(art_cache)
        art_emb = npz["embeddings"].astype(np.float32)
        art_probs = npz["sentiment_probs"].astype(np.float32)
    else:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(min(8, os.cpu_count() or 8))
        model_id = "ProsusAI/finbert"
        print(f"[FinBERT] Memuat model {model_id} (CPU length-sorted dynamic batching) ...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.eval()

        # Petakan indeks label secara dinamis agar urutan [positive, negative, neutral] pasti akurat
        id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
        label_order = [
            next(i for i, lbl in id2label.items() if "pos" in lbl),
            next(i for i, lbl in id2label.items() if "neg" in lbl),
            next(i for i, lbl in id2label.items() if "neu" in lbl),
        ]

        texts = arts["text_title"].tolist()
        n = len(texts)
        order = np.argsort([len(t) for t in texts], kind="mergesort")

        hidden_dim = model.config.hidden_size
        art_emb = np.zeros((n, hidden_dim), dtype=np.float32)
        art_probs = np.zeros((n, 3), dtype=np.float32)

        t0 = time.time()
        with torch.inference_mode():
            for start in tqdm(range(0, n, batch_size), desc="FinBERT"):
                idx = order[start : start + batch_size]
                batch_texts = [texts[i] if texts[i].strip() else " " for i in idx]
                enc = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                out = model(**enc, output_hidden_states=True)
                last_hidden = out.hidden_states[-1]
                mask = enc["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                probs = torch.softmax(out.logits, dim=-1)[:, label_order]

                art_emb[idx] = pooled.cpu().numpy()
                art_probs[idx] = probs.cpu().numpy()

        print(f"  FinBERT selesai dalam {time.time() - t0:.1f}s")
        np.savez_compressed(
            art_cache,
            embeddings=art_emb.astype(np.float16),
            sentiment_probs=art_probs.astype(np.float32),
        )

    daily_emb = aggregate_daily_mean(art_emb, arts["target_date"], base["date"])
    save_daily_parquet(daily_emb, base, "fb_emb_raw", daily_emb_cache)

    sent_art = pd.DataFrame({
        "target_date": arts["target_date"].values,
        "pos": art_probs[:, 0],
        "neg": art_probs[:, 1],
        "neu": art_probs[:, 2],
    })
    sent_art["net"] = sent_art["pos"] - sent_art["neg"]
    sent_art["is_neg"] = (sent_art["neg"] > 0.5).astype(np.float32)
    sent_art["is_pos"] = (sent_art["pos"] > 0.5).astype(np.float32)

    g = sent_art.groupby("target_date")
    sent_daily = pd.DataFrame({
        "fb_pos_mean": g["pos"].mean(),
        "fb_neg_mean": g["neg"].mean(),
        "fb_neu_mean": g["neu"].mean(),
        "fb_net_mean": g["net"].mean(),
        "fb_share_neg": g["is_neg"].mean(),
        "fb_share_pos": g["is_pos"].mean(),
    }).reindex(base["date"]).reset_index()
    sent_daily = sent_daily.rename(columns={sent_daily.columns[0]: "date"})
    sent_daily.to_parquet(daily_sent_cache, index=False)

    return daily_emb, sent_daily


# ---------------------------------------------------------------------------
# 3. Kategori C1: BAAI/bge-small-en-v1.5 (384D lokal, CLS pooling + L2 norm)
# ---------------------------------------------------------------------------
def extract_bge_small(
    arts: pd.DataFrame, base: pd.DataFrame, batch_size: int = 128, max_length: int = 48, force: bool = False
) -> np.ndarray:
    art_cache = EMB_DIR / "articles_bge_small.npz"
    daily_cache = EMB_DIR / "daily_bge_small.parquet"

    if daily_cache.exists() and art_cache.exists() and not force:
        print("[BGE-Small] Memuat cache harian dari", daily_cache.name)
        return pd.read_parquet(daily_cache).drop(columns=["date"]).to_numpy(dtype=np.float32)

    if art_cache.exists() and not force:
        print("[BGE-Small] Memuat cache artikel dari", art_cache.name)
        art_emb = np.load(art_cache)["embeddings"].astype(np.float32)
    else:
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(min(8, os.cpu_count() or 8))
        model_id = "BAAI/bge-small-en-v1.5"
        print(f"[BGE-Small] Memuat model {model_id} (CPU length-sorted dynamic batching) ...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        model.eval()

        texts = arts["text_title"].tolist()
        n = len(texts)
        order = np.argsort([len(t) for t in texts], kind="mergesort")
        dim = model.config.hidden_size
        art_emb = np.zeros((n, dim), dtype=np.float32)

        t0 = time.time()
        with torch.inference_mode():
            for start in tqdm(range(0, n, batch_size), desc="BGE-Small"):
                idx = order[start : start + batch_size]
                batch_texts = [texts[i] if texts[i].strip() else " " for i in idx]
                enc = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                out = model(**enc)
                cls_vec = out.last_hidden_state[:, 0]
                cls_vec = torch.nn.functional.normalize(cls_vec, p=2, dim=1)
                art_emb[idx] = cls_vec.cpu().numpy()

        print(f"  BGE-Small selesai dalam {time.time() - t0:.1f}s")
        np.savez_compressed(art_cache, embeddings=art_emb.astype(np.float16))

    daily_mat = aggregate_daily_mean(art_emb, arts["target_date"], base["date"])
    save_daily_parquet(daily_mat, base, "bge_s_raw", daily_cache)
    return daily_mat


# ---------------------------------------------------------------------------
# 4. Kategori C2: baai/bge-large-en-v1.5 (1024D via OpenRouter API)
# ---------------------------------------------------------------------------
def _fetch_openrouter_batch(
    batch_id: int, texts: list[str], api_key: str, model_id: str = "baai/bge-large-en-v1.5", max_retries: int = 5
) -> tuple[int, np.ndarray]:
    clean_texts = [t if t.strip() else " " for t in texts]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model_id, "input": clean_texts}

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(OPENROUTER_EMBED_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            data = resp.json()
            if "data" not in data:
                raise RuntimeError(f"Respons API tidak memiliki field 'data': {str(data)[:200]}")
            items = sorted(data["data"], key=lambda x: x["index"])
            if len(items) != len(clean_texts):
                raise RuntimeError(f"Jumlah embedding ({len(items)}) != input ({len(clean_texts)})")
            vecs = np.asarray([item["embedding"] for item in items], dtype=np.float32)
            return batch_id, vecs
        except Exception as exc:
            last_err = exc
            wait_s = min(2 ** attempt, 30)
            time.sleep(wait_s)
    raise RuntimeError(f"Batch {batch_id} gagal setelah {max_retries} kali percobaan: {last_err}")


def extract_bge_large_openrouter(
    arts: pd.DataFrame, base: pd.DataFrame, batch_size: int = 256, max_workers: int = 6, force: bool = False
) -> np.ndarray:
    art_cache = EMB_DIR / "articles_bge_large.npz"
    daily_cache = EMB_DIR / "daily_bge_large.parquet"
    ckpt_path = EMB_DIR / "ckpt_bge_large.npz"

    if daily_cache.exists() and art_cache.exists() and not force:
        print("[BGE-Large OpenRouter] Memuat cache harian dari", daily_cache.name)
        return pd.read_parquet(daily_cache).drop(columns=["date"]).to_numpy(dtype=np.float32)

    if art_cache.exists() and not force:
        print("[BGE-Large OpenRouter] Memuat cache artikel dari", art_cache.name)
        art_emb = np.load(art_cache)["embeddings"].astype(np.float32)
    else:
        api_key = load_env_key()
        texts = arts["text_title"].tolist()
        n = len(texts)
        dim = 1024
        batches = [(i, i * batch_size, min((i + 1) * batch_size, n)) for i in range((n + batch_size - 1) // batch_size)]

        art_emb_f16 = np.zeros((n, dim), dtype=np.float16)
        done_mask = np.zeros(len(batches), dtype=bool)
        if ckpt_path.exists() and not force:
            ckpt = np.load(ckpt_path)
            if ckpt["embeddings"].shape == (n, dim) and len(ckpt["done_batches"]) == len(batches):
                art_emb_f16 = ckpt["embeddings"].copy()
                done_mask = ckpt["done_batches"].copy()
                print(f"[BGE-Large OpenRouter] Melanjutkan checkpoint: {int(done_mask.sum())}/{len(batches)} batch selesai")

        pending = [(b_id, s, e) for b_id, s, e in batches if not done_mask[b_id]]
        print(f"[BGE-Large OpenRouter] Mengambil {len(pending)} batch (batch_size={batch_size}, workers={max_workers}) ...")

        lock = threading.Lock()
        completed_since_ckpt = 0
        t0 = time.time()

        def save_ckpt():
            tmp = EMB_DIR / "ckpt_bge_large.tmp.npz"
            np.savez(tmp, embeddings=art_emb_f16, done_batches=done_mask)
            os.replace(tmp, ckpt_path)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_fetch_openrouter_batch, b_id, texts[s:e], api_key): (b_id, s, e)
                for b_id, s, e in pending
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="BGE-Large API"):
                b_id, s, e = futures[fut]
                _, vecs = fut.result()
                with lock:
                    art_emb_f16[s:e] = vecs.astype(np.float16)
                    done_mask[b_id] = True
                    completed_since_ckpt += 1
                    if completed_since_ckpt >= 20:
                        save_ckpt()
                        completed_since_ckpt = 0

        if not bool(np.all(done_mask)):
            save_ckpt()
            raise RuntimeError("Sebagian batch BGE-Large belum selesai.")

        print(f"  BGE-Large OpenRouter selesai dalam {time.time() - t0:.1f}s")
        art_emb = art_emb_f16.astype(np.float32)
        np.savez_compressed(art_cache, embeddings=art_emb_f16)
        if ckpt_path.exists():
            ckpt_path.unlink()

    daily_mat = aggregate_daily_mean(art_emb, arts["target_date"], base["date"])
    save_daily_parquet(daily_mat, base, "bge_l_raw", daily_cache)
    return daily_mat


# ---------------------------------------------------------------------------
# 5. Reduksi Dimensi Anti-Leakage (fit HANYA di split train)
# ---------------------------------------------------------------------------
def build_pca_features(daily_mat: np.ndarray, is_train: np.ndarray, prefix: str) -> pd.DataFrame:
    """Fit StandardScaler + PCA hanya pada 720 hari train, lalu transform semua hari."""
    scaler = StandardScaler().fit(daily_mat[is_train])
    X_scaled = scaler.transform(daily_mat)

    blocks = []
    for k in PCA_DIMS:
        pca = PCA(n_components=k, random_state=SEED).fit(X_scaled[is_train])
        Z = pca.transform(X_scaled)
        var_sum = float(pca.explained_variance_ratio_.sum())
        print(f"  PCA {prefix:10s} k={k:2d}: explained_var={var_sum:.3f}")
        cols = [f"{prefix}_pca{k}_{i:02d}" for i in range(k)]
        blocks.append(pd.DataFrame(Z, columns=cols))
    return pd.concat(blocks, axis=1)


def main():
    parser = argparse.ArgumentParser(description="Ekstraksi 3 kategori embedding Tahap 1 Tugas 3")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["fasttext", "bge_large", "bge_small", "finbert"],
        choices=["fasttext", "finbert", "bge_small", "bge_large"],
        help="Daftar model yang diekstrak",
    )
    parser.add_argument("--force", action="store_true", help="Paksa ekstraksi ulang meski cache ada")
    args = parser.parse_args()

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    arts, base = load_articles_and_base()
    is_train = base["split"].eq("train").to_numpy()
    print(f"Artikel={len(arts)} | Hari={len(base)} (train={int(is_train.sum())})")

    daily_mats = {}
    sent_daily = None

    if "fasttext" in args.models:
        daily_mats["ft"] = extract_fasttext(arts, base, force=args.force)
    if "bge_large" in args.models:
        daily_mats["bge_l"] = extract_bge_large_openrouter(arts, base, force=args.force)
    if "bge_small" in args.models:
        daily_mats["bge_s"] = extract_bge_small(arts, base, force=args.force)
    if "finbert" in args.models:
        daily_mats["fb_emb"], sent_daily = extract_finbert(arts, base, force=args.force)

    # Jika dijalankan parsial, muat parquet harian lain yang sudah ada agar CSV gabungan lengkap
    parquet_map = {
        "ft": ("daily_fasttext.parquet", None),
        "fb_emb": ("daily_finbert_emb.parquet", "daily_finbert_sent.parquet"),
        "bge_s": ("daily_bge_small.parquet", None),
        "bge_l": ("daily_bge_large.parquet", None),
    }
    for prefix, (emb_file, sent_file) in parquet_map.items():
        if prefix not in daily_mats and (EMB_DIR / emb_file).exists():
            daily_mats[prefix] = pd.read_parquet(EMB_DIR / emb_file).drop(columns=["date"]).to_numpy(dtype=np.float32)
        if sent_file and sent_daily is None and (EMB_DIR / sent_file).exists():
            sent_daily = pd.read_parquet(EMB_DIR / sent_file)

    print("Menyusun PCA anti-leakage (fit di train) -> features_embeddings_daily.csv ...")
    out_parts = [pd.DataFrame({"date": base["date"].values})]
    if sent_daily is not None:
        out_parts.append(sent_daily.drop(columns=["date"]).reset_index(drop=True))
    for prefix in ("ft", "fb_emb", "bge_s", "bge_l"):
        if prefix in daily_mats:
            out_parts.append(build_pca_features(daily_mats[prefix], is_train, prefix))

    out_df = pd.concat(out_parts, axis=1)
    out_df.to_csv(FEATURES_EMB_DAILY, index=False)
    print(f"Tersimpan: {FEATURES_EMB_DAILY} | shape={out_df.shape} | NaN={int(out_df.isna().sum().sum())}")


if __name__ == "__main__":
    main()
