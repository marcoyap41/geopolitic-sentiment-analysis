"""
Modul Multimodal Fusion & Optimasi Tugas 3:
  * Model Tabular: LogisticRegression L2, XGBoost, LightGBM (dengan TimeSeriesSplit CV + seleksi di val)
  * Early Fusion Sekuensial: PyTorch 1-layer LSTM
  * Late Fusion: Weighted Probability Averaging, Weighted Logit Averaging, & Stacking Meta-Learner (OOF)
  * Hybrid / Gated Neural Fusion: GatedMultimodalNet (PyTorch)
  * Threshold Optimization di val (anti-leakage) untuk menyeimbangkan DA dan F1 macro.
"""
import copy
import itertools
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier
from scipy.special import expit, logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from config import SEED
from evaluate import directional_accuracy

warnings.filterwarnings("ignore")


def _set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def eval_prob_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=np.float64)
    pred = (p >= threshold).astype(int)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5
    return {
        "DA": directional_accuracy(y, pred),
        "F1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "ROC_AUC": auc,
        "share_pred1": float(pred.mean()),
    }


def tune_threshold(
    y_val: np.ndarray, prob_val: np.ndarray, low: float = 0.35, high: float = 0.65, step: float = 0.01
) -> tuple[float, dict]:
    """Pilih threshold probabilitas di val yang memaksimalkan 0.5*(DA + F1_macro) tanpa kolaps ke 1 kelas."""
    grid = np.round(np.arange(low, high + 1e-9, step), 4)
    best_tau = 0.5
    best_score = (-1.0, -1.0)
    best_m = eval_prob_metrics(y_val, prob_val, 0.5)

    for tau in grid:
        m = eval_prob_metrics(y_val, prob_val, float(tau))
        if not (0.25 <= m["share_pred1"] <= 0.85):
            continue
        score = (0.5 * (m["DA"] + m["F1_macro"]), m["DA"])
        if score > best_score or (score == best_score and abs(tau - 0.5) < abs(best_tau - 0.5)):
            best_score = score
            best_tau = float(tau)
            best_m = m

    if best_score[0] < 0:
        best_tau = 0.5
        best_m = eval_prob_metrics(y_val, prob_val, 0.5)
    return best_tau, best_m


def _make_tabular_estimator(kind: str, hp: dict):
    if kind == "logreg":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(penalty="l2", max_iter=3000, random_state=SEED, **hp),
        )
    if kind == "xgb":
        return XGBClassifier(
            learning_rate=0.04,
            subsample=0.8,
            min_child_weight=5,
            random_state=SEED,
            eval_metric="logloss",
            n_jobs=2,
            **hp,
        )
    if kind == "lgbm":
        return LGBMClassifier(
            learning_rate=0.03,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.5,
            reg_lambda=5.0,
            random_state=SEED,
            n_jobs=2,
            verbose=-1,
            **hp,
        )
    raise ValueError(f"Estimator tidak dikenal: {kind}")


TABULAR_GRIDS = {
    "logreg": [{"C": c} for c in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)],
    "xgb": [
        {"max_depth": d, "n_estimators": n, "colsample_bytree": cs, "reg_lambda": rl}
        for d, n, cs, rl in itertools.product((2, 3), (100, 200, 300), (0.3, 0.5), (5.0, 15.0))
    ],
    "lgbm": [
        {"num_leaves": nl, "min_child_samples": mcs, "n_estimators": n}
        for nl, mcs, n in itertools.product((4, 7), (15, 30), (100, 250))
    ],
}


def _compute_oof_probs(kind: str, hp: dict, X_tr: pd.DataFrame, y_tr: pd.Series, n_splits: int = 5) -> np.ndarray:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof = np.full(len(y_tr), np.nan, dtype=np.float64)
    for tr_idx, va_idx in tscv.split(X_tr):
        est = _make_tabular_estimator(kind, hp)
        est.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
        oof[va_idx] = est.predict_proba(X_tr.iloc[va_idx])[:, 1]
    return oof


def fit_eval_tabular(df: pd.DataFrame, cols: list[str], kind: str, compute_oof: bool = False) -> dict:
    """Tune hyperparameter dengan TimeSeriesSplit CV (train) & pilih terbaik di val, evaluasi sekali di test."""
    m_tr = df["split"].eq("train").to_numpy()
    m_va = df["split"].eq("val").to_numpy()
    m_te = df["split"].eq("test").to_numpy()

    X_tr, y_tr = df.loc[m_tr, cols], df.loc[m_tr, "y"].astype(int)
    X_va, y_va = df.loc[m_va, cols], df.loc[m_va, "y"].astype(int)
    X_te, y_te = df.loc[m_te, cols], df.loc[m_te, "y"].astype(int)

    best_score = (-1.0, -1.0, -1.0)
    best_pack = None

    for hp in TABULAR_GRIDS[kind]:
        est = _make_tabular_estimator(kind, hp)
        est.fit(X_tr, y_tr)
        p_va = est.predict_proba(X_va)[:, 1]
        vm = eval_prob_metrics(y_va, p_va, 0.5)
        score = (0.5 * (vm["DA"] + vm["F1_macro"]), vm["DA"], vm["ROC_AUC"])
        if score > best_score:
            best_score = score
            best_pack = (hp, est, p_va, vm)

    best_hp, best_est, p_va, val_m = best_pack
    p_te = best_est.predict_proba(X_te)[:, 1]
    test_m = eval_prob_metrics(y_te, p_te, 0.5)

    oof = _compute_oof_probs(kind, best_hp, X_tr, y_tr) if compute_oof else None
    cv_da = float(np.mean((oof[~np.isnan(oof)] >= 0.5) == y_tr.to_numpy()[~np.isnan(oof)])) if oof is not None else np.nan

    return {
        "kind": kind,
        "hp": best_hp,
        "n_features": len(cols),
        "cols": cols,
        "model": best_est,
        "cv_da": cv_da,
        "oof_train": oof,
        "val_y": y_va.to_numpy(),
        "val_prob": p_va,
        "val_metrics": val_m,
        "test_y": y_te.to_numpy(),
        "test_prob": p_te,
        "test_metrics": test_m,
    }


def fit_eval_logreg(df: pd.DataFrame, cols: list[str], compute_oof: bool = False) -> dict:
    return fit_eval_tabular(df, cols, "logreg", compute_oof=compute_oof)


def fit_eval_xgb(df: pd.DataFrame, cols: list[str], compute_oof: bool = False) -> dict:
    return fit_eval_tabular(df, cols, "xgb", compute_oof=compute_oof)


def fit_eval_lgbm(df: pd.DataFrame, cols: list[str], compute_oof: bool = False) -> dict:
    return fit_eval_tabular(df, cols, "lgbm", compute_oof=compute_oof)


# ---------------------------------------------------------------------------
# Early Fusion Sekuensial: 1-layer LSTM
# ---------------------------------------------------------------------------
class _SeqLSTM(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32, dropout: float = 0.4):
        super().__init__()
        self.in_drop = nn.Dropout(dropout * 0.5)
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(self.in_drop(x))
        return self.head(out[:, -1, :]).squeeze(-1)


def fit_eval_lstm(df: pd.DataFrame, cols: list[str], seq_len: int = 5, hidden_dim: int = 32, dropout: float = 0.4) -> dict:
    _set_seed(SEED)
    m_tr = df["split"].eq("train").to_numpy()
    m_va = df["split"].eq("val").to_numpy()
    m_te = df["split"].eq("test").to_numpy()

    X_all = df[cols].to_numpy(dtype=np.float32)
    y_all = df["y"].to_numpy(dtype=np.float32)

    imp = SimpleImputer(strategy="median").fit(X_all[m_tr])
    scaler = StandardScaler().fit(imp.transform(X_all[m_tr]))
    X_scaled = scaler.transform(imp.transform(X_all)).astype(np.float32)

    X_seqs, y_seqs, seq_splits = [], [], []
    splits = df["split"].to_numpy()
    for t in range(seq_len - 1, len(df)):
        X_seqs.append(X_scaled[t - seq_len + 1 : t + 1])
        y_seqs.append(y_all[t])
        seq_splits.append(splits[t])

    X_seqs = np.asarray(X_seqs, dtype=np.float32)
    y_seqs = np.asarray(y_seqs, dtype=np.float32)
    seq_splits = np.asarray(seq_splits)

    s_tr = seq_splits == "train"
    s_va = seq_splits == "val"
    s_te = seq_splits == "test"

    dl_tr = DataLoader(TensorDataset(torch.tensor(X_seqs[s_tr]), torch.tensor(y_seqs[s_tr])), batch_size=32, shuffle=False)
    Xv_t = torch.tensor(X_seqs[s_va])
    yv_t = torch.tensor(y_seqs[s_va])
    Xt_t = torch.tensor(X_seqs[s_te])

    model = _SeqLSTM(X_all.shape[1], hidden_dim=hidden_dim, dropout=dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    crit = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait = 0
    for _ in range(100):
        model.train()
        for xb, yb in dl_tr:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.inference_mode():
            vloss = float(crit(model(Xv_t), yv_t).item())
        if vloss < best_loss - 1e-5:
            best_loss = vloss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= 15:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        p_va = torch.sigmoid(model(Xv_t)).cpu().numpy()
        p_te = torch.sigmoid(model(Xt_t)).cpu().numpy()

    return {
        "kind": "lstm",
        "hp": {"seq_len": seq_len, "hidden_dim": hidden_dim, "dropout": dropout},
        "n_features": len(cols),
        "val_y": df.loc[m_va, "y"].to_numpy(dtype=int),
        "val_prob": p_va,
        "val_metrics": eval_prob_metrics(df.loc[m_va, "y"].to_numpy(), p_va, 0.5),
        "test_y": df.loc[m_te, "y"].to_numpy(dtype=int),
        "test_prob": p_te,
        "test_metrics": eval_prob_metrics(df.loc[m_te, "y"].to_numpy(), p_te, 0.5),
    }


# ---------------------------------------------------------------------------
# Late Fusion: Weighted Prob/Logit Averaging & Stacking Meta-Learner
# ---------------------------------------------------------------------------
def run_late_weighted_fusion(
    y_val: np.ndarray,
    p_text_val: np.ndarray,
    p_time_val: np.ndarray,
    y_test: np.ndarray,
    p_text_test: np.ndarray,
    p_time_test: np.ndarray,
    space: str = "prob",
) -> dict:
    eps = 1e-6
    if space == "logit":
        t_v = logit(np.clip(p_text_val, eps, 1 - eps))
        m_v = logit(np.clip(p_time_val, eps, 1 - eps))
        t_t = logit(np.clip(p_text_test, eps, 1 - eps))
        m_t = logit(np.clip(p_time_test, eps, 1 - eps))
    else:
        t_v, m_v, t_t, m_t = p_text_val, p_time_val, p_text_test, p_time_test

    best_alpha = 0.5
    best_score = (-1.0, -1.0, -1.0)
    best_val_prob = None

    for alpha in np.round(np.arange(0.0, 1.001, 0.05), 2):
        comb_v = alpha * t_v + (1.0 - alpha) * m_v
        pv = expit(comb_v) if space == "logit" else comb_v
        vm = eval_prob_metrics(y_val, pv, 0.5)
        score = (0.5 * (vm["DA"] + vm["F1_macro"]), vm["DA"], vm["ROC_AUC"])
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
            best_val_prob = pv

    comb_t = best_alpha * t_t + (1.0 - best_alpha) * m_t
    p_test = expit(comb_t) if space == "logit" else comb_t

    return {
        "alpha": best_alpha,
        "space": space,
        "val_y": y_val,
        "val_prob": best_val_prob,
        "val_metrics": eval_prob_metrics(y_val, best_val_prob, 0.5),
        "test_y": y_test,
        "test_prob": p_test,
        "test_metrics": eval_prob_metrics(y_test, p_test, 0.5),
    }


def run_late_stacking_fusion(
    df: pd.DataFrame,
    oof_text_train: np.ndarray,
    oof_time_train: np.ndarray,
    p_text_val: np.ndarray,
    p_time_val: np.ndarray,
    p_text_test: np.ndarray,
    p_time_test: np.ndarray,
    context_cols: list[str] | None = None,
) -> dict:
    """Stacking Meta-Learner (LogReg L2) dilatih pada prediksi OOF TimeSeriesSplit di train."""
    m_tr = df["split"].eq("train").to_numpy()
    m_va = df["split"].eq("val").to_numpy()
    m_te = df["split"].eq("test").to_numpy()

    valid_oof = (~np.isnan(oof_text_train)) & (~np.isnan(oof_time_train))
    y_tr = df.loc[m_tr, "y"].to_numpy(dtype=int)[valid_oof]
    y_va = df.loc[m_va, "y"].to_numpy(dtype=int)
    y_te = df.loc[m_te, "y"].to_numpy(dtype=int)

    eps = 1e-6
    def _build_meta_X(pt, pm, ctx_df):
        lt = logit(np.clip(pt, eps, 1 - eps))
        lm = logit(np.clip(pm, eps, 1 - eps))
        arr = np.column_stack([lt, lm])
        if ctx_df is not None and len(ctx_df.columns) > 0:
            arr = np.hstack([arr, ctx_df.to_numpy(dtype=np.float64)])
        return arr

    ctx_tr = df.loc[m_tr, context_cols].iloc[valid_oof] if context_cols else None
    ctx_va = df.loc[m_va, context_cols] if context_cols else None
    ctx_te = df.loc[m_te, context_cols] if context_cols else None

    X_meta_tr = _build_meta_X(oof_text_train[valid_oof], oof_time_train[valid_oof], ctx_tr)
    X_meta_va = _build_meta_X(p_text_val, p_time_val, ctx_va)
    X_meta_te = _build_meta_X(p_text_test, p_time_test, ctx_te)

    best_score = (-1.0, -1.0)
    best_pack = None
    for C in (0.01, 0.05, 0.1, 0.5, 1.0, 5.0):
        meta = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(penalty="l2", C=C, max_iter=2000, random_state=SEED),
        )
        meta.fit(X_meta_tr, y_tr)
        pv = meta.predict_proba(X_meta_va)[:, 1]
        vm = eval_prob_metrics(y_va, pv, 0.5)
        sc = (0.5 * (vm["DA"] + vm["F1_macro"]), vm["DA"])
        if sc > best_score:
            best_score = sc
            best_pack = (C, meta, pv, vm)

    best_C, best_meta, p_va, val_m = best_pack
    p_te = best_meta.predict_proba(X_meta_te)[:, 1]
    return {
        "C": best_C,
        "meta_model": best_meta,
        "val_y": y_va,
        "val_prob": p_va,
        "val_metrics": val_m,
        "test_y": y_te,
        "test_prob": p_te,
        "test_metrics": eval_prob_metrics(y_te, p_te, 0.5),
    }


# ---------------------------------------------------------------------------
# Hybrid / Gated Neural Fusion (GatedMultimodalNet)
# ---------------------------------------------------------------------------
class GatedMultimodalNet(nn.Module):
    """Encoder proyeksi terpisah + Gating Mechanism (GMU-style) sebelum classification head."""

    def __init__(self, d_text: int, d_time: int, dropout: float = 0.45):
        super().__init__()
        self.text_enc = nn.Sequential(
            nn.Linear(d_text, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.text_proj = nn.Linear(32, 16)
        self.time_enc = nn.Sequential(
            nn.Linear(d_time, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
        )
        self.gate = nn.Sequential(
            nn.Linear(48, 16),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(16),
            nn.Dropout(dropout * 0.5),
            nn.Linear(16, 1),
        )

    def forward(self, x_text: torch.Tensor, x_time: torch.Tensor) -> torch.Tensor:
        ht = self.text_enc(x_text)
        hm = self.time_enc(x_time)
        g = self.gate(torch.cat([ht, hm], dim=-1))
        fused = g * torch.tanh(self.text_proj(ht)) + (1.0 - g) * hm
        return self.head(fused).squeeze(-1)


def fit_eval_gated_fusion(
    df: pd.DataFrame, text_cols: list[str], time_cols: list[str], dropout: float = 0.45
) -> dict:
    _set_seed(SEED)
    m_tr = df["split"].eq("train").to_numpy()
    m_va = df["split"].eq("val").to_numpy()
    m_te = df["split"].eq("test").to_numpy()

    def _prep(cols):
        X = df[cols].to_numpy(dtype=np.float32)
        imp = SimpleImputer(strategy="median").fit(X[m_tr])
        sc = StandardScaler().fit(imp.transform(X[m_tr]))
        return (
            torch.tensor(sc.transform(imp.transform(X[m_tr])), dtype=torch.float32),
            torch.tensor(sc.transform(imp.transform(X[m_va])), dtype=torch.float32),
            torch.tensor(sc.transform(imp.transform(X[m_te])), dtype=torch.float32),
        )

    Xt_tr, Xt_va, Xt_te = _prep(text_cols)
    Xm_tr, Xm_va, Xm_te = _prep(time_cols)
    y_tr = torch.tensor(df.loc[m_tr, "y"].to_numpy(dtype=np.float32))
    y_va_np = df.loc[m_va, "y"].to_numpy(dtype=int)
    y_va_t = torch.tensor(y_va_np, dtype=torch.float32)
    y_te_np = df.loc[m_te, "y"].to_numpy(dtype=int)

    model = GatedMultimodalNet(len(text_cols), len(time_cols), dropout=dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-2)
    crit = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait = 0
    for _ in range(120):
        model.train()
        opt.zero_grad()
        loss = crit(model(Xt_tr, Xm_tr), y_tr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.inference_mode():
            vloss = float(crit(model(Xt_va, Xm_va), y_va_t).item())
        if vloss < best_loss - 1e-5:
            best_loss = vloss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= 18:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        p_va = torch.sigmoid(model(Xt_va, Xm_va)).cpu().numpy()
        p_te = torch.sigmoid(model(Xt_te, Xm_te)).cpu().numpy()

    return {
        "kind": "gated_fusion",
        "hp": {"dropout": dropout, "d_text": len(text_cols), "d_time": len(time_cols)},
        "n_features": len(text_cols) + len(time_cols),
        "val_y": y_va_np,
        "val_prob": p_va,
        "val_metrics": eval_prob_metrics(y_va_np, p_va, 0.5),
        "test_y": y_te_np,
        "test_prob": p_te,
        "test_metrics": eval_prob_metrics(y_te_np, p_te, 0.5),
    }
