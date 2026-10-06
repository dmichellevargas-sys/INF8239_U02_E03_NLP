"""Entrenamiento, comparación y evaluación del clasificador de texto (Ejercicio 03).

Protocolo (declarado ANTES de mirar el conjunto de prueba):
  * Partición oficial del corpus: train (2018) para ajustar, development (2018) como prueba.
  * Duplicados internos y textos de prueba ya presentes en train se eliminan.
  * Solo se usa el texto. id, source y link se excluyen (fuga demostrada en la auditoría).
  * Métrica principal: F1 macro, estimada con validación cruzada estratificada de
    5 pliegues SOBRE TRAIN.
  * Regla de selección ("una desviación estándar", Hastie et al., 2009): entre los
    candidatos cuyo F1 de validación no es inferior al mejor menos un error estándar
    (std / raíz de 5), se elige el de menor tiempo de ajuste (más simple y barato).
  * La prueba se usa una sola vez, al final, para todos los modelos.

Uso:  uv run python scripts/train_text.py
"""
from __future__ import annotations

import json
import platform
import sys
from time import perf_counter

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold

from inf8239_u02.config import ROOT, settings
from inf8239_u02.data import (
    LEAKAGE_COLUMNS,
    load_dataset,
    normalize_text,
    official_split,
    validate_dataframe,
)
from inf8239_u02.evaluation import (
    bootstrap_f1_ci,
    classification_metrics,
    mcnemar_exact,
    select_errors,
    top_coefficients,
)
from inf8239_u02.modeling import (
    BASELINE,
    CANDIDATES,
    PARAM_GRIDS,
    build_models,
    decision_scores,
    lowercase_only,
)

REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
CATEGORIES_FILE = ROOT / "docs/error_categories.csv"
POSITIVE = "true"  # classes_ ordenadas: ["fake", "true"] -> classes_[1] = "true"


def fit_candidates(x_train, y_train, rs: int) -> tuple[dict, pd.DataFrame]:
    models = build_models(rs)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
    fitted, rows, cv_tables = {}, [], []
    for name, pipeline in models.items():
        search = GridSearchCV(pipeline, PARAM_GRIDS[name] or {}, scoring="f1_macro", cv=cv,
                              n_jobs=1, refit=True, return_train_score=True)
        search.fit(x_train, y_train)
        best = search.best_estimator_
        # Tiempo de ajuste final: mediana de 3 repeticiones del mejor pipeline.
        times = []
        for _ in range(3):
            model = clone(best)
            start = perf_counter()
            model.fit(x_train, y_train)
            times.append(perf_counter() - start)
        fitted[name] = model
        idx = search.best_index_
        rows.append({
            "model": name,
            "best_params": json.dumps(search.best_params_),
            "cv_f1_macro_mean": search.cv_results_["mean_test_score"][idx],
            "cv_f1_macro_std": search.cv_results_["std_test_score"][idx],
            "cv_train_f1_mean": search.cv_results_["mean_train_score"][idx],
            "fit_median_s": float(np.median(times)),
        })
        table = pd.DataFrame(search.cv_results_)
        table.insert(0, "model", name)
        cv_tables.append(table[["model", "params", "mean_test_score", "std_test_score",
                                "mean_train_score", "mean_fit_time"]])
    pd.concat(cv_tables).to_csv(REPORTS / "cv_results.csv", index=False)
    return fitted, pd.DataFrame(rows)


def choose(summary: pd.DataFrame, n_folds: int = 5) -> str:
    """Regla de un error estándar: el modelo más barato estadísticamente indistinguible del mejor."""
    cand = summary[summary["model"].isin(CANDIDATES)]
    best = cand.loc[cand["cv_f1_macro_mean"].idxmax()]
    threshold = best["cv_f1_macro_mean"] - best["cv_f1_macro_std"] / np.sqrt(n_folds)
    eligible = cand[cand["cv_f1_macro_mean"] >= threshold]
    return str(eligible.sort_values("fit_median_s").iloc[0]["model"])


def evaluate_on_test(fitted: dict, x_test, y_test, summary: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    preds = {}
    rows = []
    for name, model in fitted.items():
        start = perf_counter()
        pred = model.predict(x_test)
        predict_ms = (perf_counter() - start) * 1000
        preds[name] = pred
        path = MODELS / f"{name}.joblib"
        joblib.dump(model, path)
        m = classification_metrics(y_test, pred)
        low, high = bootstrap_f1_ci(y_test, pred, seed=settings.random_state)
        rows.append({"model": name, **m, "test_f1_ci95_low": low, "test_f1_ci95_high": high,
                     "predict_ms_total": predict_ms, "size_kb": path.stat().st_size / 1024})
    test = pd.DataFrame(rows).rename(columns={"accuracy": "test_accuracy", "f1_macro": "test_f1_macro"})
    table = summary.merge(test, on="model")
    return table, preds


def plot_confusions(y_test, preds: dict) -> None:
    fig, axes = plt.subplots(1, len(CANDIDATES), figsize=(10, 4))
    for ax, name in zip(axes, CANDIDATES):
        ConfusionMatrixDisplay.from_predictions(y_test, preds[name], labels=["fake", "true"],
                                                cmap="Blues", ax=ax, colorbar=False)
        ax.set(title=name, xlabel="Predicha", ylabel="Real")
    fig.suptitle("Matrices de confusión · conjunto de prueba (development 2018)")
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_confusion.png", dpi=170)
    plt.close(fig)


def plot_comparison(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    order = table.set_index("model").loc[[BASELINE, *CANDIDATES]]
    x = np.arange(len(order))
    ax.bar(x - 0.2, order["cv_f1_macro_mean"], 0.4, yerr=order["cv_f1_macro_std"],
           capsize=4, label="Validación cruzada (train)")
    err = np.vstack([order["test_f1_macro"] - order["test_f1_ci95_low"],
                     order["test_f1_ci95_high"] - order["test_f1_macro"]])
    ax.bar(x + 0.2, order["test_f1_macro"], 0.4, yerr=err, capsize=4, label="Prueba (IC 95 %)")
    ax.set_xticks(x, order.index)
    ax.set(ylim=(0, 1), ylabel="F1 macro", title="Línea base vs. pipelines")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_model_comparison.png", dpi=170)
    plt.close(fig)


def plot_terms(model) -> None:
    terms = top_coefficients(model, k=15)
    terms.to_csv(REPORTS / "top_terms.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 7))
    ordered = terms.sort_values("weight")
    colors = ["#c0392b" if c == "fake" else "#1f77b4" for c in ordered["class"]]
    ax.barh(ordered["term"], ordered["weight"], color=colors)
    ax.set(title="Términos con mayor peso (rojo = fake, azul = true)", xlabel="Coeficiente")
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_top_terms.png", dpi=170)
    plt.close(fig)


def plot_generalization(table: pd.DataFrame, robust: pd.DataFrame, external: dict,
                        selected: str) -> None:
    """Mismo modelo, tres escenarios cada vez más exigentes."""
    row = table.set_index("model").loc[selected]
    rob = robust.set_index("model").loc[selected]
    labels = ["Prueba oficial\n(medios vistos, 2018)", "Medios no vistos\n(CV por grupos)",
              "Corpus v2.0\n(2020-2021)"]
    values = [row["test_f1_macro"], rob["group_cv_f1_mean"], external.get("f1_macro", np.nan)]
    errors = [[row["test_f1_macro"] - row["test_f1_ci95_low"], rob["group_cv_f1_std"],
               external.get("f1_macro", 0) - external.get("f1_ci95", (0, 0))[0]],
              [row["test_f1_ci95_high"] - row["test_f1_macro"], rob["group_cv_f1_std"],
               external.get("f1_ci95", (0, 0))[1] - external.get("f1_macro", 0)]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, yerr=errors, capsize=5, color=["#1f77b4", "#ff7f0e", "#c0392b"])
    for i, v in enumerate(values):
        ax.text(i, v / 2, f"{v:.3f}", ha="center", color="white", fontweight="bold")
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.set(ylim=(0, 1), ylabel="F1 macro", title=f"Generalización de {selected}")
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_generalization.png", dpi=170)
    plt.close(fig)


def error_analysis(model, name: str, test: pd.DataFrame, pred) -> pd.DataFrame:
    scores = decision_scores(model, test["text"])
    if name == "tfidf_word_logreg":
        confidence = np.where(pred == POSITIVE, scores, 1 - scores)  # prob. de la clase predicha
    else:
        confidence = np.abs(scores)  # distancia al hiperplano
    frame = test[["id", "topic", "source", "label", "text"]].copy()
    frame["predicted"] = pred
    frame["confidence"] = np.round(confidence, 4)
    frame["words"] = frame["text"].str.split().str.len()
    errors = frame[frame["label"] != frame["predicted"]].copy()
    errors["error_type"] = np.where(errors["predicted"] == POSITIVE,
                                    "FP: falsa predicha como verdadera",
                                    "FN: verdadera predicha como falsa")
    errors.drop(columns="text").to_csv(REPORTS / "errors_all.csv", index=False)
    sample = select_errors(errors, n=20)
    sample["excerpt"] = sample["text"].str.replace(r"\s+", " ", regex=True).str[:240]
    sample = sample.drop(columns="text")
    if CATEGORIES_FILE.exists():  # categorías asignadas tras la lectura humana de cada texto
        cats = pd.read_csv(CATEGORIES_FILE, dtype={"id": str})
        sample = sample.merge(cats, on="id", how="left")
    else:
        sample["category"] = np.nan
        sample["explanation"] = np.nan
    sample["category"] = sample["category"].fillna("SIN CATEGORIZAR")
    sample.to_csv(REPORTS / "error_analysis_20.csv", index=False)
    counts = pd.crosstab(sample["category"], sample["error_type"])
    fig, ax = plt.subplots(figsize=(9, 4))
    counts.sort_index(ascending=False).plot.barh(stacked=True, ax=ax, color=["#1f77b4", "#c0392b"])
    ax.set(title="20 errores analizados por categoría", xlabel="Errores", ylabel="")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.4, -0.18), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_error_categories.png", dpi=170)
    plt.close(fig)
    return sample


def robustness_unseen_sources(df: pd.DataFrame, fitted: dict) -> pd.DataFrame:
    """Validación por grupos: los medios de prueba nunca aparecen en entrenamiento.
    Mide cuánto del desempeño depende de reconocer el estilo de un medio."""
    keys = df["source"].str.strip().str.lower()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=settings.random_state)
    rows = []
    for name in CANDIDATES:
        scores = []
        for tr, te in cv.split(df["text"], df["label"], groups=keys):
            model = clone(fitted[name]).fit(df["text"].iloc[tr], df["label"].iloc[tr])
            pred = model.predict(df["text"].iloc[te])
            scores.append(classification_metrics(df["label"].iloc[te], pred)["f1_macro"])
        rows.append({"model": name, "group_cv_f1_mean": float(np.mean(scores)),
                     "group_cv_f1_std": float(np.std(scores)),
                     "folds": json.dumps([round(s, 4) for s in scores])})
    return pd.DataFrame(rows)


def external_evaluation(model, train: pd.DataFrame) -> dict:
    """Evaluación temporal externa con la versión 2.0 del corpus (2020-2021)."""
    if not settings.external_path.exists():
        return {}
    ext = pd.read_csv(settings.external_path, dtype={"id": str})
    known = set(train["text"].map(normalize_text))
    ext = ext[~ext["text"].map(normalize_text).isin(known)]
    pred = model.predict(ext["text"])
    out = classification_metrics(ext["label"], pred)
    out["rows"] = len(ext)
    out["f1_ci95"] = bootstrap_f1_ci(ext["label"], pred, seed=settings.random_state)
    # Ablación: mismo modelo sin unificar cifras (los dígitos de la v2.0 quedan como tokens propios).
    variant = clone(model).set_params(tfidf__preprocessor=lowercase_only)
    variant.fit(train["text"], train["label"])
    out["ablation_keep_digits_f1_macro"] = float(
        classification_metrics(ext["label"], variant.predict(ext["text"]))["f1_macro"])
    out["ablation_keep_digits_recall_fake"] = float(
        classification_metrics(ext["label"], variant.predict(ext["text"]))["recall_fake"])
    return out


def run() -> dict:
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    df = load_dataset()
    validate_dataframe(df, settings.text_column, settings.target_column)
    train, test, split_log = official_split(df)
    features = [settings.text_column]
    assert not set(features) & set(LEAKAGE_COLUMNS), "Columna con fuga entre los predictores"

    fitted, summary = fit_candidates(train["text"], train["label"], settings.random_state)
    selected = choose(summary)
    table, preds = evaluate_on_test(fitted, test["text"], test["label"], summary)
    table["selected"] = table["model"] == selected
    table.to_csv(REPORTS / "text_metrics.csv", index=False)

    mcnemar = mcnemar_exact(test["label"], preds[CANDIDATES[0]], preds[CANDIDATES[1]])
    plot_confusions(test["label"], preds)
    plot_comparison(table)
    plot_terms(fitted["tfidf_word_logreg"])
    errors = error_analysis(fitted[selected], selected, test, preds[selected])

    full = pd.concat([train, test])
    robust = robustness_unseen_sources(full, fitted)
    robust.to_csv(REPORTS / "robustness_unseen_sources.csv", index=False)
    external = external_evaluation(fitted[selected], train)
    plot_generalization(table, robust, external, selected)

    joblib.dump(fitted[selected], MODELS / "text_model.joblib")
    results = {
        "split": split_log,
        "selected_model": selected,
        "selection_rule": "regla de 1 error estándar sobre F1 macro CV en train; entre elegibles, menor tiempo de ajuste",
        "mcnemar_" + "_vs_".join(CANDIDATES): mcnemar,
        "external_v2_selected": external,
        "errors_in_test_selected": int((preds[selected] != test["label"].to_numpy()).sum()),
        "environment": {
            "python": sys.version.split()[0], "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "scikit_learn": sklearn.__version__, "pandas": pd.__version__,
        },
    }
    (REPORTS / "text_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    results["table"] = table
    results["robustness"] = robust
    results["errors"] = errors
    return results


def main() -> None:
    r = run()
    cols = ["model", "cv_f1_macro_mean", "cv_f1_macro_std", "test_f1_macro",
            "test_f1_ci95_low", "test_f1_ci95_high", "fit_median_s", "size_kb", "selected"]
    print("Partición:", r["split"])
    print(r["table"][cols].round(4).to_string(index=False))
    print("Seleccionado:", r["selected_model"])
    print("McNemar:", r[next(k for k in r if k.startswith("mcnemar"))])
    print("Medios no vistos:\n", r["robustness"].round(4).to_string(index=False))
    print("Externo v2:", {k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in r["external_v2_selected"].items()})
    print("Errores en prueba:", r["errors_in_test_selected"],
          "| muestra analizada:", len(r["errors"]))
    print("Artefactos en reports/ y models/")


if __name__ == "__main__":
    main()
