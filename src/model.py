"""
Model XGBoost untuk klasifikasi arah kurs.
Dipakai dua kali dengan kode yang sama:
  * baseline    : fitur kurs saja
  * model gabungan : fitur kurs + fitur NLP
Hyperparameter dipilih dari grid kecil berdasarkan akurasi arah di data VALIDASI.
"""
import itertools
import warnings

from xgboost import XGBClassifier

from config import SEED
from dataset import split_xy
from evaluate import directional_accuracy

warnings.filterwarnings("ignore")

# grid kecil: kedalaman pohon x jumlah pohon
GRID = [{"max_depth": d, "n_estimators": n} for d, n in itertools.product((2, 3), (100, 300))]


def make_model(**hp):
    """XGBoost dangkal dan teregularisasi karena datanya hanya ratusan hari."""
    return XGBClassifier(learning_rate=0.05, subsample=0.8, colsample_bytree=0.5,
                         reg_lambda=5.0, min_child_weight=5, random_state=SEED,
                         eval_metric="logloss", n_jobs=2, **hp)


def run_xgb(df, cols):
    """Latih di train, pilih hyperparameter dengan data val, lalu prediksi val dan test.
    Return: hyperparameter terpilih, (y, prediksi) untuk val dan test."""
    Xtr, ytr = split_xy(df, cols, "train")
    Xva, yva = split_xy(df, cols, "val")
    Xte, yte = split_xy(df, cols, "test")

    best_hp, best_model, best_da = None, None, -1
    for hp in GRID:
        model = make_model(**hp).fit(Xtr, ytr)
        da = directional_accuracy(yva, model.predict(Xva))
        if da > best_da:
            best_hp, best_model, best_da = hp, model, da

    return best_hp, {"val": (yva.values, best_model.predict(Xva)),
                     "test": (yte.values, best_model.predict(Xte))}
