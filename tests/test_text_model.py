import numpy as np
import pandas as pd

from inf8239_u02.modeling import BASELINE, CANDIDATES, PARAM_GRIDS, build_models, preprocess

TEXTS = ["el gobierno anunció 25 millones", "increíble descubren que nadie sabía esto",
         "el ministro aseguró que en 2018", "ya todo el mundo lo sabe pues nadie dice nada"] * 3
LABELS = ["true", "fake", "true", "fake"] * 3


def test_each_pipeline_predicts_one_label_per_text():
    for model in build_models().values():
        model.fit(TEXTS, LABELS)
        assert len(model.predict(["el ministro anunció"])) == 1
        assert list(model.named_steps) == ["tfidf", "model"]


def test_catalog_has_baseline_and_two_candidates_with_symmetric_grids():
    models = build_models()
    assert BASELINE in models and all(name in models for name in CANDIDATES)
    sizes = [len(next(iter(PARAM_GRIDS[n].values()))) for n in CANDIDATES]
    assert sizes[0] == sizes[1]


def test_preprocess_unifies_digits_and_number_token():
    import re

    def tokens(text):
        return re.findall(r"\w+", preprocess(text))

    assert tokens("Ganó 1.500 votos") == tokens("Ganó *NUMBER* votos") == ["ganó", "number", "votos"]
    assert "number" in tokens("En 2018")


def test_pipelines_only_need_text_not_metadata():
    model = build_models()["tfidf_word_logreg"]
    model.fit(pd.Series(TEXTS), LABELS)  # una sola columna de texto: no hay acceso a source/link
    proba = model.predict_proba(["el gobierno anunció"])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
