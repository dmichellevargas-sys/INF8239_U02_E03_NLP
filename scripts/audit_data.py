"""Auditoría del corpus: calidad, duplicados y fuga. Escribe reports/audit_*.

Uso:  uv run python scripts/audit_data.py
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from inf8239_u02.audit import build_audit, source_purity
from inf8239_u02.config import ROOT, settings
from inf8239_u02.data import load_dataset, sha256, validate_dataframe

REPORTS = ROOT / "reports"


def source_only_leakage(df: pd.DataFrame) -> dict:
    """Experimento de control: ¿cuánto predice la etiqueta SOLO el nombre del medio?
    Si es muy alto, 'source' es una fuga y debe excluirse."""
    x = df[["source"]].apply(lambda s: s.str.strip().str.lower())
    model = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ("model", LogisticRegression(max_iter=2000)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=settings.random_state)
    scores = cross_val_score(model, x, df["label"], cv=cv, scoring="f1_macro")
    return {"f1_macro_mean": float(scores.mean()), "f1_macro_std": float(scores.std())}


def plots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    pd.crosstab(df["topic"], df["label"]).sort_values("fake").plot.barh(ax=axes[0])
    axes[0].set(title="Noticias por tema y clase", xlabel="Noticias", ylabel="")
    for label, group in df.groupby("label"):
        axes[1].hist(group["text"].str.split().str.len().clip(upper=1500), bins=40,
                     alpha=0.55, label=label)
    axes[1].set(title="Longitud del texto (palabras, recortado a 1500)",
                xlabel="Palabras", ylabel="Noticias")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_audit_overview.png", dpi=170)
    plt.close(fig)


def run() -> dict:
    REPORTS.mkdir(exist_ok=True)
    df = load_dataset()
    validate_dataframe(df, settings.text_column, settings.target_column)
    audit = build_audit(df)
    audit["sha256_corpus_csv"] = sha256(settings.dataset_path)
    audit["leakage_source_only_model"] = source_only_leakage(df)
    purity = source_purity(df)
    purity.to_csv(REPORTS / "audit_source_purity.csv")
    (REPORTS / "audit_summary.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plots(df)
    return audit


def main() -> None:
    audit = run()
    print("Filas:", audit["rows"], "| Clases:", audit["label_distribution"])
    print("Por partición:", audit["label_by_split"])
    print("Duplicados:", audit["duplicates"])
    print(f"Medios: {audit['sources']} | medios de una sola clase: {audit['pure_sources']} "
          f"({audit['rows_in_pure_sources']} noticias)")
    print("F1 macro usando SOLO el medio (fuga):", audit["leakage_source_only_model"])
    print("Medio mencionado dentro del texto:", audit["source_mentioned_in_text"])
    print("Artefactos: reports/audit_summary.json, audit_source_purity.csv, fig_audit_overview.png")


if __name__ == "__main__":
    main()
