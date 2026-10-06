"""Configuración central del proyecto (leída desde .env con valores por defecto seguros)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# Corpus: The Spanish Fake News Corpus (Posadas-Durán et al., 2019). Licencia CC BY 4.0.
# Se fija el commit para que la descarga sea idéntica en cualquier máquina.
CORPUS_REPO = "https://github.com/jpposadas/FakeNewsCorpusSpanish"
CORPUS_COMMIT = "732f4870abd8d3f5ab4956fd92b040a7913614e3"
RAW_BASE_URL = f"https://raw.githubusercontent.com/jpposadas/FakeNewsCorpusSpanish/{CORPUS_COMMIT}"
RAW_FILES = {
    "train": "train.xlsx",          # versión 1.0 (2018), partición oficial de entrenamiento
    "development": "development.xlsx",  # versión 1.0 (2018), partición oficial de evaluación
    "external_v2": "test.xlsx",     # versión 2.0 (2020-2021), evaluación externa temporal
}


@dataclass(frozen=True)
class Settings:
    data_source: str = os.getenv("DATA_SOURCE", "url")
    dataset_path: Path = ROOT / os.getenv("DATASET_PATH", "data/processed/corpus.csv")
    external_path: Path = ROOT / os.getenv("EXTERNAL_PATH", "data/processed/external_v2.csv")
    text_column: str = os.getenv("TEXT_COLUMN", "text")
    target_column: str = os.getenv("TARGET_COLUMN", "label")
    group_column: str = os.getenv("GROUP_COLUMN", "source")
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))


settings = Settings()
