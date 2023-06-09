import numpy as np
import pandas as pd

from fold.loop import backtest, train, train_backtest
from fold.loop.encase import train_evaluate
from fold.models.baseline import Naive
from fold.models.random import RandomBinaryClassifier, RandomClassifier
from fold.splitters import ExpandingWindowSplitter, SingleWindowSplitter
from fold.transformations.columns import OnlyPredictions
from fold.transformations.dev import Test
from fold.utils.tests import (
    generate_all_zeros,
    generate_sine_wave_data,
    generate_zeros_and_ones,
)


def test_random_classifier() -> None:
    X, y = generate_all_zeros(1000)

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    transformations = [RandomClassifier(all_classes=[0, 1]), OnlyPredictions()]
    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    assert pred.squeeze().sum() > 1


def test_random_binary_classifier() -> None:
    X, y = generate_zeros_and_ones(10000)
    sample_weights = pd.Series(np.random.uniform(0, 1, len(y)), index=y.index)

    splitter = SingleWindowSplitter(0.2)
    transformations = [RandomBinaryClassifier()]
    _, preds, _ = train_evaluate(
        transformations, X, y, splitter, sample_weights=sample_weights
    )
    assert 0.4 <= preds.iloc[:, 0].mean() <= 0.6
    assert 0.4 <= preds.iloc[:, 1].mean() <= 0.6
    assert 0.4 <= preds.iloc[:, 2].mean() <= 0.6

    pos_noise = np.random.normal(0.65, 0.1, len(y)).clip(0, 1)
    neg_noise = np.random.normal(0.35, 0.1, len(y)).clip(0, 1)
    skewed_sample_weights = pd.Series(
        np.where(y == 1.0, pos_noise, neg_noise), index=y.index
    )

    splitter = SingleWindowSplitter(0.2)
    transformations = [RandomBinaryClassifier()]
    _, preds, _ = train_evaluate(
        transformations, X, y, splitter, sample_weights=skewed_sample_weights
    )
    assert 0.5 <= preds.iloc[:, 2].mean() <= 0.8
    assert 0.2 <= preds.iloc[:, 1].mean() < 0.5


def check_if_not_nan(x):
    assert not x.isna().squeeze().any()


test_assert = Test(fit_func=check_if_not_nan, transform_func=lambda X: X)

test_length = 1200


def test_baseline_naive() -> None:
    X, y = generate_sine_wave_data(cycles=10, length=1200)

    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
    transformations = [
        Naive(),
        test_assert,
    ]
    pred, _ = train_backtest(transformations, X, y, splitter)
    assert (
        pred.squeeze() == y.shift(1)[pred.index]
    ).all()  # last year's value should match this year's value, with the sine wave we generated
    assert (
        len(pred) == 1200 * 0.8
    )  # should return non-NaN predictions for the all out-of-sample sets


def test_baseline_naive_online() -> None:
    X, y = generate_sine_wave_data(cycles=10, length=1200)

    naive = Naive()
    naive.properties._internal_supports_minibatch_backtesting = False
    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
    pred, _ = train_backtest(naive, X, y, splitter)
    assert (
        pred.squeeze() == y.shift(1)[pred.index]
    ).all()  # last year's value should match this year's value, with the sine wave we generated
    assert (
        len(pred) == 1200 * 0.8
    )  # should return non-NaN predictions for the all out-of-sample sets
