import pandas as pd

from inf8239_u02.audit import duplicate_report, source_purity
from inf8239_u02.evaluation import (
    bootstrap_f1_ci,
    classification_metrics,
    mcnemar_exact,
    select_errors,
)


def test_metrics_perfect_prediction():
    m = classification_metrics(["fake", "true"], ["fake", "true"])
    assert m["f1_macro"] == 1.0 and m["support_fake"] == 1


def test_bootstrap_interval_contains_point_estimate():
    y = ["fake", "true"] * 50
    pred = ["fake", "true"] * 40 + ["true", "fake"] * 10
    low, high = bootstrap_f1_ci(y, pred, n_boot=300)
    point = classification_metrics(y, pred)["f1_macro"]
    assert 0 <= low <= point <= high <= 1


def test_mcnemar_identical_models_is_not_significant():
    y = ["fake", "true", "fake"]
    out = mcnemar_exact(y, y, y)
    assert out["p_value"] == 1.0 and out["only_a_correct"] == 0


def test_select_errors_returns_twenty_balanced_and_most_confident_first():
    errors = pd.DataFrame({
        "error_type": ["FN"] * 30 + ["FP"] * 15,
        "confidence": list(range(30)) + list(range(15)),
    })
    chosen = select_errors(errors, n=20)
    assert len(chosen) == 20
    assert (chosen["error_type"] == "FP").sum() == 10
    assert chosen[chosen["error_type"] == "FN"]["confidence"].min() == 20


def test_source_purity_detects_single_class_sources():
    df = pd.DataFrame({"source": ["A", "a ", "B", "B"], "label": ["fake", "fake", "true", "fake"]})
    purity = source_purity(df)
    assert purity.loc["a", "purity"] == 1.0
    assert purity.loc["b", "purity"] == 0.5


def test_duplicate_report_counts_cross_split():
    df = pd.DataFrame({"text": ["Hola", "hola!", "x"], "split": ["train", "development", "train"],
                       "label": ["fake", "fake", "true"]})
    rep = duplicate_report(df)
    assert rep["cross_development_train"] == 1
    assert rep["duplicates_with_conflicting_label"] == 0
