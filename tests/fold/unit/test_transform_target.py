import numpy as np

from fold.composites.target import TransformTarget
from fold.loop.encase import train_backtest
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.dev import Test
from fold.transformations.math import TakeLog
from fold.utils.tests import generate_sine_wave_data, generate_zeros_and_ones_skewed


def all_y_values_above_1(X, y):
    assert np.all(y >= 1.0)


def all_y_values_below_1(X, y):
    assert np.all(y <= 1.0)


test_all_y_values_above_1 = Test(
    fit_func=all_y_values_above_1, transform_func=lambda X: X
)
test_transform_plus_2 = Test(
    fit_func=lambda x: x,
    transform_func=lambda x: x + 2.0,
    inverse_transform_func=lambda x: x - 2.0,
)


def test_target_transformation_dummy() -> None:
    X, y = generate_zeros_and_ones_skewed(length=1000, weights=[0.5, 0.5])
    X.columns = ["predictions_woo"]
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)

    pipeline = TransformTarget(
        [lambda x: x + 1, test_all_y_values_above_1],
        y_pipeline=test_transform_plus_2,
    )
    pred, _, _, _ = train_backtest(pipeline, X, y, splitter)
    assert (X.squeeze()[pred.index] - 1.0 == pred.squeeze()).all()


def test_target_transformation_dummy_dont_invert() -> None:
    X, y = generate_zeros_and_ones_skewed(length=1000, weights=[0.5, 0.5])
    X.columns = ["predictions_woo"]
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)

    pipeline = TransformTarget(
        [lambda x: x + 1, test_all_y_values_above_1],
        y_pipeline=test_transform_plus_2,
        invert_wrapped_output=False,
    )
    pred, _, _, _ = train_backtest(pipeline, X, y, splitter)
    assert (X.squeeze()[pred.index] + 1.0 == pred.squeeze()).all()


def test_target_transformation_log() -> None:
    X, y = generate_sine_wave_data()
    X, y = X + 2, y + 2
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=100)

    def assert_y_not_nan(X, y):
        assert np.allclose(np.log(X.squeeze())[1:], y.shift(1)[1:], atol=0.001)

    pipeline = TransformTarget(
        [
            Test(fit_func=assert_y_not_nan, transform_func=lambda x: x),
        ],
        y_pipeline=TakeLog(),
    )

    _, _, _, _ = train_backtest(pipeline, X, y, splitter)
