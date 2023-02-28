from drift.loop import backtest, train
from drift.loop.types import Backend, TrainMethod
from drift.models import BaselineRegressor
from drift.splitters import ExpandingWindowSplitter
from drift.transformations.test import Test
from drift.utils.tests import generate_all_zeros, generate_sine_wave_data


def run_loop(train_method: TrainMethod, backend: Backend) -> None:

    # the naive model returns X as prediction, so y.shift(1) should be == pred
    X = generate_sine_wave_data()
    y = X["sine"].shift(-1)

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [BaselineRegressor(strategy=BaselineRegressor.Strategy.naive)]

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = backtest(transformations_over_time, X, y, splitter)
    assert (X.squeeze()[pred.index] == pred.squeeze()).all()


def test_loop():
    run_loop(TrainMethod.sequential, Backend.no)
    run_loop(TrainMethod.parallel, Backend.no)


def test_sameple_weights() -> None:
    def assert_sample_weights_exist(X, y, sample_weight):
        assert sample_weight is not None
        assert sample_weight[0] == 0

    test_sample_weights_exist = Test(
        fit_func=assert_sample_weights_exist, transform_func=lambda X: X
    )

    X = generate_all_zeros(1000)
    y = X.squeeze()
    sameple_weights = generate_all_zeros(1000).squeeze()

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [
        test_sample_weights_exist,
    ]
    _ = train(transformations, X, y, splitter, sample_weights=sameple_weights)
