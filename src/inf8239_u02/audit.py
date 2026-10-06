"""Funciones de auditoría: calidad, duplicados y señales de fuga."""
from __future__ import annotations

import pandas as pd

from .data import normalize_text


def source_purity(df: pd.DataFrame, source: str = "source", target: str = "label") -> pd.DataFrame:
    """Por medio: cuántas noticias publica y qué proporción pertenece a su clase mayoritaria."""
    key = df[source].astype(str).str.strip().str.lower()
    table = pd.crosstab(key, df[target])
    out = pd.DataFrame({
        "rows": table.sum(axis=1),
        "majority_label": table.idxmax(axis=1),
        "purity": table.max(axis=1) / table.sum(axis=1),
    }).sort_values("rows", ascending=False)
    out.index.name = "source"
    return out


def duplicate_report(df: pd.DataFrame, text: str = "text", split: str = "split") -> dict:
    keys = df[text].map(normalize_text)
    exact = int(df[text].duplicated().sum())
    normalized = int(keys.duplicated().sum())
    out = {"exact_duplicates": exact, "normalized_duplicates": normalized}
    if split in df.columns:
        groups = {name: set(keys[df[split] == name]) for name in df[split].unique()}
        names = sorted(groups)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                out[f"cross_{a}_{b}"] = len(groups[a] & groups[b])
        conflicting = (
            pd.DataFrame({"k": keys, "y": df["label"]}).groupby("k")["y"].nunique().gt(1).sum()
        )
        out["duplicates_with_conflicting_label"] = int(conflicting)
    return out


def source_mentioned_in_text(df: pd.DataFrame, source: str = "source", text: str = "text") -> int:
    return int(sum(
        str(s).strip().lower() in str(t).lower() for s, t in zip(df[source], df[text])
    ))


def build_audit(df: pd.DataFrame) -> dict:
    lengths = df["text"].str.len()
    words = df["text"].str.split().str.len()
    purity = source_purity(df)
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "nulls": {k: int(v) for k, v in df.isna().sum().items()},
        "empty_text": int(df["text"].fillna("").str.strip().eq("").sum()),
        "label_distribution": {k: int(v) for k, v in df["label"].value_counts().items()},
        "label_by_split": {
            s: {k: int(v) for k, v in g["label"].value_counts().items()}
            for s, g in df.groupby("split")
        },
        "topics": {k: int(v) for k, v in df["topic"].value_counts().items()},
        "chars_by_label": {
            k: {m: round(float(x), 1) for m, x in v.describe().items()}
            for k, v in lengths.groupby(df["label"])
        },
        "median_words": float(words.median()),
        "duplicates": duplicate_report(df),
        "sources": int(purity.shape[0]),
        "pure_sources": int((purity["purity"] == 1).sum()),
        "rows_in_pure_sources": int(purity.loc[purity["purity"] == 1, "rows"].sum()),
        "source_mentioned_in_text": source_mentioned_in_text(df),
        # Artefacto de formato: la v1.0 sustituyó las cifras por el token NUMBER.
        "share_with_NUMBER_token_by_label": {
            k: round(float(v), 3)
            for k, v in df["text"].str.contains(r"\bNUMBER\b").groupby(df["label"]).mean().items()
        },
        "share_with_digits_by_label": {
            k: round(float(v), 3)
            for k, v in df["text"].str.contains(r"\d").groupby(df["label"]).mean().items()
        },
    }
