import numpy as np

from fold.loop import backtest, train
from fold.models.baseline import BaselineMean, BaselineNaiveSeasonal
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.columns import OnlyPredictions
from fold.transformations.dev import Test
from fold.utils.tests import generate_sine_wave_data


def test_baseline_naive_seasonal() -> None:
    X, y = generate_sine_wave_data(
        cycles=10, length=120, freq="M"
    )  # create a sine wave with yearly seasonality

    def check_if_not_nan(x):
        assert not x.isna().squeeze().any()

    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
    transformations = [
        BaselineNaiveSeasonal(seasonal_length=12),
        Test(fit_func=check_if_not_nan, transform_func=lambda X: X),
        OnlyPredictions(),
    ]
    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert np.isclose(
        pred.squeeze(), y[pred.index], atol=0.02
    ).all()  # last year's value should match this year's value, with the sine wave we generated
    assert (
        len(pred) == 120 * 0.8
    )  # should return non-NaN predictions for the all out-of-sample sets


def test_baseline_mean() -> None:
    X, y = generate_sine_wave_data(cycles=10, length=1200)

    def check_if_not_nan(x):
        assert not x.isna().squeeze().any()

    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
    transformations = [
        BaselineMean(window_size=12),
        Test(fit_func=check_if_not_nan, transform_func=lambda X: X),
        OnlyPredictions(),
    ]
    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert np.isclose(
        y.shift(1).rolling(12).mean()[pred.index], pred.squeeze(), atol=0.01
    ).all()
    assert (
        len(pred) == 1200 * 0.8
    )  # should return non-NaN predictions for the all out-of-sample sets
