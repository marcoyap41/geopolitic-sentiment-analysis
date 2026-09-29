"""Metrik evaluasi: Directional Accuracy dan F1 macro (+ confusion matrix)."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


def directional_accuracy(y, pred):
    """Directional Accuracy (DA): proporsi hari yang arahnya ditebak benar."""
    return float(np.mean(np.asarray(y) == np.asarray(pred)))


def metrics(y, pred) -> dict:
    """DA, F1 macro, dan porsi prediksi kelas 1.
    F1 macro penting karena model yang selalu menebak kelas mayoritas bisa punya DA
    lumayan tinggi tetapi F1 macro rendah."""
    y, pred = np.asarray(y), np.asarray(pred)
    return {"DA": directional_accuracy(y, pred),
            "F1_macro": f1_score(y, pred, average="macro", zero_division=0),
            "share_pred1": float(pred.mean())}


def cm_table(y, pred) -> pd.DataFrame:
    """Confusion matrix (baris = aktual, kolom = prediksi)."""
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return pd.DataFrame(cm, index=["aktual_0 (tidak melemah)", "aktual_1 (melemah)"],
                        columns=["pred_0", "pred_1"])
