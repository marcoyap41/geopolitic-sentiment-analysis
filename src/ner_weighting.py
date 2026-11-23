"""
Tahap 2A - NER Tiered Entity Weighting berbasis Teori Ekonomi (spaCy en_core_web_sm).

Mengklasifikasikan tiap artikel CNBC ke dalam 3 tier kedekatan kausal ke USD/IDR:
  * Tier 1 (Moneter & Kurs Langsung): Bank sentral/pejabat moneter (Fed, FOMC, Powell,
    Bank Indonesia, ECB, IMF, BoJ, PBoC, BoE) & istilah suku bunga/valas.
  * Tier 2 (Makro & Geopolitik Penggerak): Negara kunci (U.S., China, Indonesia) via
    NER GPE/NORP & indikator makro/komoditas (inflation, CPI, tariff, oil, OPEC).
  * Tier 0 (Dasar): Berita di luar Tier 1 & Tier 2 (bobot dasar = 1.0).

Semua agregasi harian menormalisasi bobot seluruh artikel pada hari T sehingga total = 1.0.
"""
import re
import time

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm

from config import PROCESSED
from extract_embeddings import EMB_DIR, load_articles_and_base

NER_CACHE = EMB_DIR / "articles_ner_tiers.parquet"

# ---------------------------------------------------------------------------
# Aturan entitas & istilah per Tier
# ---------------------------------------------------------------------------
TIER1_ENTITIES_EXACT = {
    "fed", "the fed", "federal reserve", "the federal reserve", "fomc",
    "federal open market committee", "powell", "jerome powell",
    "bank indonesia", "bi", "perry warjiyo", "warjiyo",
    "ecb", "european central bank", "the european central bank", "lagarde", "christine lagarde",
    "imf", "international monetary fund", "the international monetary fund",
    "bank of japan", "boj", "ueda", "kazuo ueda",
    "pboc", "people's bank of china", "peoples bank of china",
    "bank of england", "boe", "bailey", "andrew bailey",
    "janet yellen", "yellen", "central bank", "central banks",
}

# Safeguard word-boundary untuk judul berita pendek (ketika casing/judul membuat spaCy miss)
TIER1_NER_SAFEGUARD_RE = re.compile(
    r"(?:\b(?:Federal Reserve|FOMC|Jerome Powell|Powell|Bank Indonesia|European Central Bank|"
    r"Christine Lagarde|Lagarde|International Monetary Fund|Bank of Japan|People's Bank of China|"
    r"Bank of England|central banks?)\b|\b(?:Fed|ECB|IMF|BOJ|PBOC|BOE)\b(?:'s)?)"
)

TIER1_TERMS_RE = re.compile(
    r"\b(?:interest rates?|rate hikes?|rate cuts?|basis points?|bps|monetary policy|"
    r"fed funds|hawkish|dovish|dollars?|usd|greenback|rupiah|idr|"
    r"exchange rates?|forex|currenc(?:y|ies))\b",
    flags=re.IGNORECASE,
)

TIER2_GPE_EXACT = {
    "u.s.", "us", "usa", "u.s.a.", "united states", "the united states",
    "america", "american", "americans",
    "china", "chinese", "beijing",
    "indonesia", "indonesian", "indonesians", "jakarta",
}

TIER2_TERMS_RE = re.compile(
    r"\b(?:inflation|cpi|pce|gdp|recession|unemployment|payrolls?|"
    r"tariffs?|trade war|oil|crude|brent|wti|opec)\b",
    flags=re.IGNORECASE,
)


def _norm_ent(text: str) -> str:
    s = re.sub(r"\s+", " ", text.strip().lower())
    if s.startswith("the "):
        s = s[4:]
    return s


def extract_ner_tiers(arts: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Jalankan spaCy en_core_web_sm pada judul artikel dan simpan cache tier ke parquet."""
    if NER_CACHE.exists() and not force:
        cached = pd.read_parquet(NER_CACHE)
        if len(cached) == len(arts):
            print(f"[NER] Memuat cache dari {NER_CACHE.name} ({len(cached)} artikel)")
            cached["target_date"] = pd.to_datetime(cached["target_date"])
            return cached

    print("[NER] Memuat spaCy en_core_web_sm (hanya komponen NER) ...")
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer", "attribute_ruler"])
    texts = arts["text_title"].fillna("").astype(str).tolist()

    t0 = time.time()
    records = []
    for i, doc in enumerate(tqdm(nlp.pipe(texts, batch_size=512), total=len(texts), desc="spaCy NER")):
        text = texts[i]
        orgs = [ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"]
        gpes = [ent.text.strip() for ent in doc.ents if ent.label_ in ("GPE", "NORP")]
        persons = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]

        org_person_norm = {_norm_ent(e) for e in (orgs + persons)}
        gpe_norm = {_norm_ent(e) for e in gpes}

        hit_t1_spacy = any(
            e in TIER1_ENTITIES_EXACT or "federal reserve" in e or "bank indonesia" in e or "central bank" in e
            for e in org_person_norm
        )
        hit_t1_ner = bool(hit_t1_spacy or TIER1_NER_SAFEGUARD_RE.search(text))
        hit_t1_term = bool(TIER1_TERMS_RE.search(text))
        is_t1 = hit_t1_ner or hit_t1_term

        hit_t2_gpe_spacy = any(e in TIER2_GPE_EXACT for e in gpe_norm)
        hit_t2_term = bool(TIER2_TERMS_RE.search(text) or "opec" in org_person_norm)
        is_t2 = hit_t2_gpe_spacy or hit_t2_term

        tier = 1 if is_t1 else (2 if is_t2 else 0)
        records.append({
            "url": arts["url"].iat[i],
            "target_date": arts["target_date"].iat[i],
            "split": arts["split"].iat[i],
            "hit_tier1_spacy": hit_t1_spacy,
            "hit_tier1_ner": hit_t1_ner,
            "hit_tier1_term": hit_t1_term,
            "hit_tier2_gpe": hit_t2_gpe_spacy,
            "hit_tier2_term": hit_t2_term,
            "is_tier1": is_t1,
            "is_tier2": is_t2,
            "tier": tier,
            "ner_orgs": "|".join(orgs[:8]),
            "ner_gpes": "|".join(gpes[:8]),
            "ner_persons": "|".join(persons[:8]),
        })

    ner_df = pd.DataFrame(records)
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    ner_df.to_parquet(NER_CACHE, index=False)
    print(f"[NER] Selesai dalam {time.time() - t0:.1f}s -> {NER_CACHE}")
    return ner_df


def compute_article_weights(
    ner_df: pd.DataFrame, b1: float, b2: float, mode: str = "hierarchical", t0_weight: float = 1.0
) -> np.ndarray:
    """Hitung bobot mentah per artikel sebelum normalisasi harian."""
    is_t1 = ner_df["is_tier1"].to_numpy(dtype=bool)
    is_t2 = ner_df["is_tier2"].to_numpy(dtype=bool)
    if mode == "hierarchical":
        w = np.where(is_t1, 1.0 + b1, np.where(is_t2, 1.0 + b2, t0_weight))
    elif mode == "additive":
        w = t0_weight + b1 * is_t1.astype(np.float64) + b2 * is_t2.astype(np.float64)
    else:
        raise ValueError(f"Mode bobot tidak dikenal: {mode}")
    return w.astype(np.float64)


def aggregate_daily_weighted(
    vectors: np.ndarray,
    target_dates: pd.Series,
    base_dates: pd.Series,
    raw_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Agregasi vektor artikel ke harian dengan normalisasi bobot sehingga jumlah bobot hari T = 1.0."""
    date_to_idx = {d: i for i, d in enumerate(pd.to_datetime(base_dates))}
    day_idx = pd.to_datetime(target_dates).map(date_to_idx).to_numpy(dtype=np.int64)
    n_days, dim = len(base_dates), vectors.shape[1]

    w = np.asarray(raw_weights, dtype=np.float64).copy()
    daily_w_sum = np.bincount(day_idx, weights=w, minlength=n_days)

    # Fallback seragam jika skema hard-filter menghasilkan total bobot 0 pada suatu hari
    zero_days = np.where(daily_w_sum <= 0.0)[0]
    if len(zero_days) > 0:
        zero_mask = np.isin(day_idx, zero_days)
        w[zero_mask] = 1.0
        daily_w_sum = np.bincount(day_idx, weights=w, minlength=n_days)

    norm_weights = w / daily_w_sum[day_idx]
    daily_norm_check = np.bincount(day_idx, weights=norm_weights, minlength=n_days)
    assert np.allclose(daily_norm_check, 1.0, atol=1e-6), "Total bobot harian harus = 1.0!"

    out = np.zeros((n_days, dim), dtype=np.float64)
    np.add.at(out, day_idx, vectors.astype(np.float64, copy=False) * norm_weights[:, None])
    return out.astype(np.float32), norm_weights


def compute_daily_ner_meta(ner_df: pd.DataFrame, base_dates: pd.Series, norm_weights: np.ndarray) -> pd.DataFrame:
    """Fitur komposisi & konsentrasi entitas per hari target."""
    date_to_idx = {d: i for i, d in enumerate(pd.to_datetime(base_dates))}
    day_idx = pd.to_datetime(ner_df["target_date"]).map(date_to_idx).to_numpy(dtype=np.int64)
    n_days = len(base_dates)

    counts = np.bincount(day_idx, minlength=n_days).astype(np.float64)
    t1_cnt = np.bincount(day_idx, weights=(ner_df["tier"] == 1).to_numpy(dtype=np.float64), minlength=n_days)
    t2_cnt = np.bincount(day_idx, weights=(ner_df["tier"] == 2).to_numpy(dtype=np.float64), minlength=n_days)
    w_sq_sum = np.bincount(day_idx, weights=norm_weights ** 2, minlength=n_days)

    eff_n = 1.0 / np.maximum(w_sq_sum, 1e-12)
    return pd.DataFrame({
        "ner_share_tier1": t1_cnt / np.maximum(counts, 1.0),
        "ner_share_tier2": t2_cnt / np.maximum(counts, 1.0),
        "ner_count_tier1_log": np.log1p(t1_cnt),
        "ner_eff_n_ratio": eff_n / np.maximum(counts, 1.0),
    })


def print_ner_summary(arts: pd.DataFrame, ner_df: pd.DataFrame) -> None:
    print("\n=== STATISTIK DISTRIBUSI TIER NER ===")
    for split_name in ("train", "val", "test"):
        sub = ner_df[ner_df["split"] == split_name]
        vc = sub["tier"].value_counts(normalize=True).to_dict()
        print(
            f"  {split_name:5s} (n={len(sub):5d}): "
            f"Tier1={vc.get(1, 0):.1%} (ner={sub['hit_tier1_ner'].mean():.1%}, "
            f"spacy={sub['hit_tier1_spacy'].mean():.1%}, term={sub['hit_tier1_term'].mean():.1%}) | "
            f"Tier2={vc.get(2, 0):.1%} (gpe={sub['hit_tier2_gpe'].mean():.1%}, term={sub['hit_tier2_term'].mean():.1%}) | "
            f"Tier0={vc.get(0, 0):.1%}"
        )
    overall = ner_df["tier"].value_counts().sort_index()
    print("  Total artikel per tier:", overall.to_dict())

    for t in (1, 2, 0):
        idx = ner_df.index[ner_df["tier"] == t][:3]
        print(f"\n  Contoh judul Tier {t}:")
        for i in idx:
            print(f"    - {arts['text_title'].iat[i]}  [ORG={ner_df['ner_orgs'].iat[i]} | GPE={ner_df['ner_gpes'].iat[i]}]")


def main():
    arts, base = load_articles_and_base()
    ner_df = extract_ner_tiers(arts)
    print_ner_summary(arts, ner_df)

    # Verifikasi normalisasi harian pada skema referensi Tier 1 (+2.0), Tier 2 (+1.0), Tier 0 (1.0)
    w = compute_article_weights(ner_df, b1=2.0, b2=1.0, mode="hierarchical")
    dummy = np.ones((len(arts), 2), dtype=np.float32)
    agg, norm_w = aggregate_daily_weighted(dummy, ner_df["target_date"], base["date"], w)
    assert np.allclose(agg, 1.0, atol=1e-5)
    meta = compute_daily_ner_meta(ner_df, base["date"], norm_w)
    print("\nVerifikasi normalisasi bobot harian total=1.0 LULUS. Rata-rata meta:", meta.mean().round(4).to_dict())


if __name__ == "__main__":
    main()
