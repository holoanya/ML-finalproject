"""Evaluation metrics. Models are trained on log1p(price); we always report the
headline numbers back on the original EUR scale (expm1), plus the log-scale RMSE
that the model actually optimizes.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import r2_score


def regression_report(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    """Both inputs are in log1p(price) space."""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    err = y_true - y_pred
    return {
        "rmse_eur": float(np.sqrt(np.mean(err ** 2))),
        "mae_eur": float(np.mean(np.abs(err))),
        "medae_eur": float(np.median(np.abs(err))),
        "r2_eur": float(r2_score(y_true, y_pred)),
        "rmse_log": float(np.sqrt(np.mean((y_true_log - y_pred_log) ** 2))),
        "r2_log": float(r2_score(y_true_log, y_pred_log)),
    }


def print_report(name: str, rep: dict) -> None:
    print(
        f"{name:24s}  RMSE {rep['rmse_eur']:7.1f} | MAE {rep['mae_eur']:6.1f} | "
        f"MdAE {rep['medae_eur']:6.1f} | R2(eur) {rep['r2_eur']:.3f} | "
        f"RMSE(log) {rep['rmse_log']:.3f} | R2(log) {rep['r2_log']:.3f}  [EUR]"
    )
