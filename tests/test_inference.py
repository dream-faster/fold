from drift.loop import backtest, train
from drift.models import BaselineRegressor
from drift.splitters import ExpandingWindowSplitter
from drift.transformations.test import Test
from drift.utils.tests import generate_all_zeros, generate_sine_wave_data


def test_loop() -> None:

    # the naive model returns X as prediction, so y.shift(1) should be == pred
    X = generate_sine_wave_data()
    y = X["sine"].shift(-1)

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [BaselineRegressor(strategy=BaselineRegressor.Strategy.naive)]

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = backtest(transformations_over_time, X, y, splitter)
    assert (X.squeeze()[pred.index] == pred.squeeze()).all()
