"""
Skrip Fine-Tuning Transformer & Classification Head (Deliverable src/ Tugas 3).

Menyediakan dua mode:
  1. Mode Cepat / Frozen-Backbone Head (`TextProjectionHead`, PyTorch):
     Melatih projection + classification head (Linear -> LayerNorm -> GELU -> Dropout -> Linear)
     di atas embedding BGE/FinBERT dengan early stopping di `val` + OOF TimeSeriesSplit di `train`.
  2. Mode End-to-End Transformer Fine-Tuning (`--end-to-end`, untuk GPU Colab/Kaggle):
     Melakukan fine-tuning 2 encoder layer teratas + classification head pada `BAAI/bge-small-en-v1.5`
     atau `ProsusAI/finbert` secara langsung pada level artikel -> probabilitas harian.
"""
import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from config import PROCESSED, SEED


def _set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class TextProjectionHead(nn.Module):
    """Proyeksi MLP teregularisasi untuk embedding teks harian berdimensi tinggi."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.45):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _fit_single_head(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_es: np.ndarray,
    y_es: np.ndarray,
    hidden_dim: int = 64,
    dropout: float = 0.45,
    lr: float = 1e-3,
    weight_decay: float = 2e-2,
    max_epochs: int = 120,
    patience: int = 18,
    seed: int = SEED,
) -> tuple[TextProjectionHead, SimpleImputer, StandardScaler]:
    _set_seed(seed)
    imp = SimpleImputer(strategy="median").fit(X_tr)
    scaler = StandardScaler().fit(imp.transform(X_tr))

    Xt = torch.tensor(scaler.transform(imp.transform(X_tr)), dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    Xe = torch.tensor(scaler.transform(imp.transform(X_es)), dtype=torch.float32)
    ye = torch.tensor(y_es, dtype=torch.float32)

    model = TextProjectionHead(Xt.shape[1], hidden_dim=hidden_dim, dropout=dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait = 0

    for _ in range(max_epochs):
        model.train()
        opt.zero_grad()
        loss = crit(model(Xt), yt)
        loss.backward()
        opt.step()

        model.eval()
        with torch.inference_mode():
            vloss = float(crit(model(Xe), ye).item())
        if vloss < best_loss - 1e-5:
            best_loss = vloss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, imp, scaler


def _predict_head(model: TextProjectionHead, imp: SimpleImputer, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    Xs = scaler.transform(imp.transform(X))
    with torch.inference_mode():
        logits = model(torch.tensor(Xs, dtype=torch.float32))
        return torch.sigmoid(logits).cpu().numpy()


def train_text_projection_head(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    hidden_dim: int = 64,
    dropout: float = 0.45,
    lr: float = 1e-3,
    weight_decay: float = 2e-2,
    n_oof_splits: int = 5,
    seed: int = SEED,
) -> dict:
    """Latih TextProjectionHead di train (early stopping di val) + OOF TimeSeriesSplit untuk stacking."""
    model, imp, scaler = _fit_single_head(
        X_train, y_train, X_val, y_val,
        hidden_dim=hidden_dim, dropout=dropout, lr=lr, weight_decay=weight_decay, seed=seed,
    )
    p_val = _predict_head(model, imp, scaler, X_val)
    p_test = _predict_head(model, imp, scaler, X_test)

    # Out-of-fold train predictions (strictly causal) untuk Meta-Learner Late Fusion
    oof_train = np.full(len(y_train), np.nan, dtype=np.float64)
    tscv = TimeSeriesSplit(n_splits=n_oof_splits)
    for fold_i, (tr_idx, va_idx) in enumerate(tscv.split(X_train)):
        # Pakai 15% akhir tr_idx sebagai internal validation early stopping agar anti-leakage
        cut = max(int(len(tr_idx) * 0.85), 20)
        f_model, f_imp, f_sc = _fit_single_head(
            X_train[tr_idx[:cut]], y_train[tr_idx[:cut]],
            X_train[tr_idx[cut:]], y_train[tr_idx[cut:]],
            hidden_dim=hidden_dim, dropout=dropout, lr=lr, weight_decay=weight_decay, seed=seed + fold_i + 1,
        )
        oof_train[va_idx] = _predict_head(f_model, f_imp, f_sc, X_train[va_idx])

    return {
        "model": model,
        "oof_train": oof_train,
        "val_prob": p_val,
        "test_prob": p_test,
    }


def run_end_to_end_finetune(model_name: str = "BAAI/bge-small-en-v1.5", epochs: int = 2, batch_size: int = 32, lr: float = 2e-5):
    """Pipeline fine-tuning end-to-end (disarankan di GPU Colab/Kaggle) untuk cabang teks."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    arts = pd.read_csv(PROCESSED.parent / "splits" / "train_articles.csv", encoding="utf-8-sig", low_memory=False)
    base = pd.read_csv(PROCESSED / "daily_base.csv", parse_dates=["date"])
    arts["target_date"] = pd.to_datetime(arts["target_date"])
    arts = arts.merge(base[["date", "y"]], left_on="target_date", right_on="date", how="inner")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    # Freeze seluruh layer encoder bawah kecuali 2 layer transformer terakhir + classifier head
    for name, param in model.named_parameters():
        if not any(k in name for k in ("encoder.layer.4", "encoder.layer.5", "encoder.layer.10", "encoder.layer.11", "classifier", "pooler")):
            param.requires_grad = False

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    texts = arts["text_title"].fillna("").tolist()
    labels = arts["y"].to_numpy(dtype=np.int64)

    model.train()
    for ep in range(epochs):
        perm = np.random.RandomState(SEED + ep).permutation(len(texts))
        total_loss = 0.0
        for s in range(0, len(perm), batch_size):
            idx = perm[s : s + batch_size]
            enc = tokenizer([texts[i] for i in idx], padding=True, truncation=True, max_length=48, return_tensors="pt").to(device)
            y_b = torch.tensor(labels[idx], dtype=torch.long, device=device)
            opt.zero_grad()
            out = model(**enc, labels=y_b)
            out.loss.backward()
            opt.step()
            total_loss += float(out.loss.item())
        print(f"Epoch {ep + 1}/{epochs} | avg_loss={total_loss / max(len(perm) // batch_size, 1):.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning Text Head atau End-to-End Transformer")
    parser.add_argument("--end-to-end", action="store_true", help="Jalankan fine-tuning Transformer end-to-end (butuh GPU)")
    parser.add_argument("--model-name", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    if args.end_to_end:
        run_end_to_end_finetune(model_name=args.model_name, epochs=args.epochs)
    else:
        base = pd.read_parquet(PROCESSED / "features_tugas3_full.parquet")
        bge = pd.read_parquet(PROCESSED / "embeddings" / "daily_bge_small.parquet")
        df = base.merge(bge, on="date", how="left")
        cols = [c for c in bge.columns if c != "date"]
        m_tr = df["split"].eq("train").to_numpy()
        m_va = df["split"].eq("val").to_numpy()
        m_te = df["split"].eq("test").to_numpy()
        res = train_text_projection_head(
            df.loc[m_tr, cols].to_numpy(), df.loc[m_tr, "y"].to_numpy(),
            df.loc[m_va, cols].to_numpy(), df.loc[m_va, "y"].to_numpy(),
            df.loc[m_te, cols].to_numpy(),
        )
        print("TextProjectionHead val DA:", float(np.mean((res["val_prob"] >= 0.5) == df.loc[m_va, "y"].to_numpy())))
        print("TextProjectionHead test DA:", float(np.mean((res["test_prob"] >= 0.5) == df.loc[m_te, "y"].to_numpy())))


if __name__ == "__main__":
    main()
