"""Carga, estandarización, validación y deduplicación del corpus."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

import pandas as pd

from .config import settings

# Columnas que existen en el corpus pero NO pueden usarse como predictores:
# - id: identificador administrativo, sin significado.
# - source / link: el medio que publica está casi perfectamente asociado a la etiqueta
#   (147 de 150 medios publican una sola clase), porque así se construyó el corpus.
#   Usarlas convertiría el problema en "reconocer el medio", no la veracidad.
LEAKAGE_COLUMNS = ("id", "source", "link")
LABELS = ("fake", "true")
STANDARD_COLUMNS = ["id", "split", "label", "topic", "source", "link", "text"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_label(value: object) -> str:
    """Unifica etiquetas de las versiones 1.0 (Fake/True) y 2.0 (False/True o booleanos)."""
    text = str(value).strip().lower()
    if text in {"fake", "false", "0"}:
        return "fake"
    if text in {"true", "1"}:
        return "true"
    raise ValueError(f"Etiqueta desconocida: {value!r}")


def normalize_text(value: object) -> str:
    """Clave de comparación para duplicados: minúsculas, sin tildes ni signos, espacios simples."""
    text = unicodedata.normalize("NFKD", str(value).lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\W+", " ", text).strip()


def standardize(raw: pd.DataFrame, split: str) -> pd.DataFrame:
    """Lleva cualquier versión del corpus al mismo esquema en minúsculas."""
    df = raw.rename(columns=lambda c: str(c).strip().lower()).rename(columns={"topics": "topic"})
    df = df.rename(columns={"category": "label"})
    required = {"id", "label", "topic", "source", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el archivo crudo: {sorted(missing)}")
    if "link" not in df.columns:
        df["link"] = ""
    out = pd.DataFrame({
        "id": df["id"].astype(str).str.strip(),
        "split": split,
        "label": df["label"].map(normalize_label),
        "topic": df["topic"].astype(str).str.strip(),
        "source": df["source"].astype(str).str.strip(),
        "link": df["link"].fillna("").astype(str).str.strip(),
        # El texto ya contiene el titular en 98 % de los casos; se usa solo "text"
        # para no duplicar el titular.
        "text": df["text"].fillna("").astype(str).str.strip(),
    })
    return out[STANDARD_COLUMNS]


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    selected = Path(path or settings.dataset_path)
    if not selected.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: {selected}. Ejecute: uv run python scripts/download_data.py"
        )
    return pd.read_csv(selected, dtype={"id": str})


def validate_dataframe(df: pd.DataFrame, text_column: str, target_column: str) -> None:
    missing = {text_column, target_column} - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    if df.empty:
        raise ValueError("El dataset está vacío")
    if df[text_column].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Existen textos vacíos")
    if df[target_column].isna().any():
        raise ValueError("Existen etiquetas nulas")
    if df[target_column].nunique(dropna=True) < 2:
        raise ValueError("Se requieren al menos dos clases")


def drop_internal_duplicates(df: pd.DataFrame, text_column: str = "text") -> tuple[pd.DataFrame, int]:
    """Elimina duplicados normalizados dentro de un mismo conjunto (conserva el primero)."""
    keys = df[text_column].map(normalize_text)
    mask = keys.duplicated(keep="first")
    return df.loc[~mask].copy(), int(mask.sum())


def remove_cross_split_duplicates(
    train: pd.DataFrame, test: pd.DataFrame, text_column: str = "text"
) -> tuple[pd.DataFrame, int]:
    """Quita de test los textos cuya forma normalizada ya aparece en train (fuga por copia)."""
    train_keys = set(train[text_column].map(normalize_text))
    mask = test[text_column].map(normalize_text).isin(train_keys)
    return test.loc[~mask].copy(), int(mask.sum())


def official_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Partición oficial del corpus (train vs development) sin duplicados internos ni cruzados."""
    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "development"].copy()
    train, dup_train = drop_internal_duplicates(train)
    test, dup_test = drop_internal_duplicates(test)
    test, dup_cross = remove_cross_split_duplicates(train, test)
    log = {
        "train_rows": len(train), "test_rows": len(test),
        "removed_dup_in_train": dup_train, "removed_dup_in_test": dup_test,
        "removed_cross_split": dup_cross,
    }
    return train, test, log
