import numpy as np
import pandas as pd

from fold.loop import train_backtest
from fold.models.wrappers.gbd import WrapXGB
from fold.splitters import ExpandingWindowSplitter, SingleWindowSplitter
from fold.utils.tests import (
    generate_monotonous_data,
    generate_sine_wave_data,
    generate_zeros_and_ones,
    generate_zeros_and_ones_skewed,
    tuneability_test,
)


def test_xgboost_regression() -> None:
    from xgboost import XGBRegressor

    X, y = generate_sine_wave_data()
    sample_weights = pd.Series(np.ones(len(y)), index=y.index)

    splitter = ExpandingWindowSplitter(initial_train_window=500, step=100)
    transformations = WrapXGB.from_model(XGBRegressor())
    pred, _ = train_backtest(
        transformations, X, y, splitter, sample_weights=sample_weights
    )
    assert (y.squeeze()[pred.index] - pred.squeeze()).abs().sum() < 20
    tuneability_test(
        WrapXGB.from_model(XGBRegressor(n_estimators=1)),
        different_params=dict(n_estimators=100),
        init_function=lambda **kwargs: WrapXGB.from_model(XGBRegressor(**kwargs)),
    )


def test_xgboost_classification() -> None:
    from xgboost import XGBClassifier

    X, y = generate_zeros_and_ones()
    sample_weights = pd.Series(np.ones(len(y)), index=y.index)

    splitter = ExpandingWindowSplitter(initial_train_window=500, step=100)
    transformations = WrapXGB.from_model(XGBClassifier())
    pred, _ = train_backtest(
        transformations, X, y, splitter, sample_weights=sample_weights
    )
    assert "predictions_XGBClassifier" in pred.columns
    assert "probabilities_XGBClassifier_0.0" in pred.columns
    assert "probabilities_XGBClassifier_1.0" in pred.columns


def test_xgboost_classification_skewed() -> None:
    from xgboost import XGBClassifier

    X, y = generate_zeros_and_ones_skewed()
    sample_weights = pd.Series(np.ones(len(y)), index=y.index)

    splitter = ExpandingWindowSplitter(initial_train_window=500, step=100)
    transformations = WrapXGB.from_model(XGBClassifier(), set_class_weights="balanced")
    pred, _ = train_backtest(
        transformations, X, y, splitter, sample_weights=sample_weights
    )
    assert "predictions_XGBClassifier" in pred.columns
    assert "probabilities_XGBClassifier_0.0" in pred.columns
    assert "probabilities_XGBClassifier_1.0" in pred.columns


def test_automatic_wrapping_xgboost() -> None:
    from xgboost import XGBRegressor

    X, y = generate_monotonous_data()
    train_backtest(
        XGBRegressor(),
        X,
        y,
        splitter=SingleWindowSplitter(0.5),
    )


def test_xgboost_init_with_args() -> None:
    from xgboost import XGBRegressor

    X, y = generate_sine_wave_data()

    splitter = ExpandingWindowSplitter(initial_train_window=500, step=100)
    transformations = WrapXGB(XGBRegressor, {"n_estimators": 100})
    pred, _ = train_backtest(transformations, X, y, splitter)
    assert (y.squeeze()[pred.index] - pred.squeeze()).abs().sum() < 20
