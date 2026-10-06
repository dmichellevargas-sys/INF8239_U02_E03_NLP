"""Catálogo de modelos: una línea base y dos pipelines comparables (solo reciben texto)."""
from __future__ import annotations

import re

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

BASELINE = "dummy"
CANDIDATES = ("tfidf_word_logreg", "tfidf_char_svm")


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def preprocess(text: str) -> str:
    """Minúsculas y cifras -> token 'number'.

    La versión 1.0 del corpus ya trae las cifras sustituidas por 'NUMBER'; la versión 2.0
    conserva los dígitos. Unificar ambas convenciones evita que el modelo aprenda un
    artefacto de formato y permite aplicar el mismo pipeline a textos nuevos.
    """
    return _NUMBER.sub(" number ", str(text).lower())


def lowercase_only(text: str) -> str:
    """Variante de ablación: conserva los dígitos tal como vienen (sin unificar cifras)."""
    return str(text).lower()


def word_vectorizer() -> TfidfVectorizer:
    # Unigramas y bigramas de palabras; min_df=2 descarta términos únicos (ruido y nombres raros).
    return TfidfVectorizer(
        preprocessor=preprocess, ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True
    )


def char_vectorizer() -> TfidfVectorizer:
    # n-gramas de caracteres dentro de palabra: capturan morfología, ortografía y estilo.
    return TfidfVectorizer(
        preprocessor=preprocess, analyzer="char_wb", ngram_range=(3, 5), min_df=2,
        sublinear_tf=True,
    )


def build_models(random_state: int = 42) -> dict[str, Pipeline]:
    return {
        BASELINE: Pipeline([
            ("tfidf", word_vectorizer()),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]),
        "tfidf_word_logreg": Pipeline([
            ("tfidf", word_vectorizer()),
            ("model", LogisticRegression(max_iter=3000, random_state=random_state)),
        ]),
        "tfidf_char_svm": Pipeline([
            ("tfidf", char_vectorizer()),
            ("model", LinearSVC(random_state=random_state)),
        ]),
    }


# Búsqueda pequeña y simétrica: mismo número de configuraciones para ambos candidatos.
PARAM_GRIDS: dict[str, dict] = {
    BASELINE: {},
    "tfidf_word_logreg": {"model__C": [0.1, 1.0, 10.0]},
    "tfidf_char_svm": {"model__C": [0.1, 1.0, 10.0]},
}


def decision_scores(model: Pipeline, texts) -> list[float] | None:
    """Puntaje continuo a favor de la clase positiva (classes_[1]); None si no existe."""
    if hasattr(model, "predict_proba") and not isinstance(model[-1], DummyClassifier):
        return model.predict_proba(texts)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(texts)
    return None
