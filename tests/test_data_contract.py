import pandas as pd
import pytest

from inf8239_u02.data import (
    LEAKAGE_COLUMNS,
    normalize_label,
    normalize_text,
    official_split,
    remove_cross_split_duplicates,
    standardize,
    validate_dataframe,
)


def test_accepts_valid_dataframe():
    df = pd.DataFrame({"text": ["uno", "dos"], "label": ["a", "b"]})
    validate_dataframe(df, "text", "label")


def test_rejects_missing_target():
    df = pd.DataFrame({"text": ["uno"]})
    with pytest.raises(ValueError, match="Faltan columnas"):
        validate_dataframe(df, "text", "label")


def test_rejects_empty_text():
    df = pd.DataFrame({"text": [""], "label": ["a"]})
    with pytest.raises(ValueError, match="textos vacíos"):
        validate_dataframe(df, "text", "label")


@pytest.mark.parametrize("raw,expected", [("Fake", "fake"), ("True", "true"),
                                          (False, "fake"), (True, "true"), (" FALSE ", "fake")])
def test_labels_from_both_corpus_versions_are_unified(raw, expected):
    assert normalize_label(raw) == expected


def test_unknown_label_is_rejected():
    with pytest.raises(ValueError):
        normalize_label("quizás")


def test_normalize_text_ignores_case_accents_and_punctuation():
    assert normalize_text("¡Noticia FALSA, señor!") == normalize_text("noticia falsa senor")


def test_standardize_accepts_v2_column_names():
    raw = pd.DataFrame({"ID": [1], "CATEGORY": [False], "TOPICS": ["Science"],
                        "SOURCE": ["X"], "HEADLINE": ["h"], "TEXT": ["cuerpo"], "LINK": ["u"]})
    out = standardize(raw, "external_v2")
    assert list(out.columns) == ["id", "split", "label", "topic", "source", "link", "text"]
    assert out.loc[0, "label"] == "fake"


def test_cross_split_duplicates_are_removed_from_test():
    train = pd.DataFrame({"text": ["Hola mundo", "otra nota"]})
    test = pd.DataFrame({"text": ["hola, MUNDO!", "nota nueva"]})
    clean, removed = remove_cross_split_duplicates(train, test)
    assert removed == 1
    assert clean["text"].tolist() == ["nota nueva"]


def test_official_split_has_no_overlap():
    df = pd.DataFrame({
        "split": ["train", "train", "development", "development", "development"],
        "text": ["a b", "c d", "A B", "e f", "e f"],
        "label": ["fake", "true", "fake", "true", "true"],
    })
    train, test, log = official_split(df)
    assert set(train["text"].map(normalize_text)).isdisjoint(test["text"].map(normalize_text))
    assert log["removed_cross_split"] == 1 and log["removed_dup_in_test"] == 1


def test_leakage_columns_are_declared():
    assert {"source", "link", "id"} <= set(LEAKAGE_COLUMNS)
