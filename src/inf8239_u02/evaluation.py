"""Métricas, intervalos, comparación pareada y selección de errores."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def classification_metrics(y_true, y_pred, labels=("fake", "true")) -> dict:
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), zero_division=0
    )
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    for i, label in enumerate(labels):
        out[f"precision_{label}"] = float(p[i])
        out[f"recall_{label}"] = float(r[i])
        out[f"f1_{label}"] = float(f[i])
        out[f"support_{label}"] = int(s[i])
    return out


def bootstrap_f1_ci(y_true, y_pred, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05):
    """Intervalo percentil bootstrap del F1 macro sobre el conjunto de prueba."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        scores[i] = f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)
    low, high = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def mcnemar_exact(y_true, pred_a, pred_b) -> dict:
    """Prueba exacta de McNemar: ¿los dos modelos se equivocan en casos distintos más de lo esperable?"""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    only_a = int(np.sum(a_ok & ~b_ok))
    only_b = int(np.sum(~a_ok & b_ok))
    n = only_a + only_b
    p = 1.0 if n == 0 else float(binomtest(only_a, n, 0.5).pvalue)
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": p}


def select_errors(errors: pd.DataFrame, n: int = 20, score_col: str = "confidence") -> pd.DataFrame:
    """Elige n errores balanceados por tipo (falso positivo / falso negativo),
    priorizando los de mayor confianza (los más informativos)."""
    if errors.empty:
        return errors.copy()
    ordered = errors.sort_values(score_col, ascending=False)
    groups = [g for _, g in ordered.groupby("error_type", sort=True)]
    per_group = n // max(len(groups), 1)
    picked = [g.head(per_group) for g in groups]
    chosen = pd.concat(picked)
    if len(chosen) < n:  # completa con los restantes más confiables
        rest = ordered.drop(chosen.index)
        chosen = pd.concat([chosen, rest.head(n - len(chosen))])
    return chosen.sort_values(["error_type", score_col], ascending=[True, False]).head(n)


def top_coefficients(pipeline, k: int = 15) -> pd.DataFrame:
    """Términos con mayor peso hacia cada clase en un modelo lineal."""
    vec, model = pipeline.named_steps["tfidf"], pipeline.named_steps["model"]
    names = np.asarray(vec.get_feature_names_out())
    coef = model.coef_.ravel()
    order = np.argsort(coef)
    neg = pd.DataFrame({"term": names[order[:k]], "weight": coef[order[:k]],
                        "class": model.classes_[0]})
    pos = pd.DataFrame({"term": names[order[-k:]][::-1], "weight": coef[order[-k:]][::-1],
                        "class": model.classes_[1]})
    return pd.concat([neg, pos], ignore_index=True)
