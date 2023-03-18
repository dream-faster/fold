import numpy as np

from fold.composites.target import TransformTarget
from fold.loop import backtest, train
from fold.models.dummy import DummyRegressor
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.difference import Difference
from fold.transformations.test import Test
from fold.utils.tests import generate_monotonous_data, generate_zeros_and_ones_skewed


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

    transformations = [
        TransformTarget(
            [lambda x: x + 1, test_all_y_values_above_1],
            y_transformation=test_transform_plus_2,
        ),
    ]

    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert ((X.squeeze()[pred.index] - 1.0) == pred.squeeze()).all()


def test_target_transformation_difference() -> None:
    X, y = generate_monotonous_data()
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=100)

    def assert_y_not_nan(X, y):
        assert not np.isnan(y).any()
        # When differencing is applied to `y`, the first value will be NaN, and it is then dropped.
        assert (len(X) - 99) % 100 == 0

    transformations = [
        TransformTarget(
            [
                Test(fit_func=assert_y_not_nan, transform_func=lambda x: x),
                DummyRegressor(0.000999),
            ],
            y_transformation=Difference(),
        ),
    ]

    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert np.isclose(y[pred.index], pred.squeeze(), atol=0.0001).all()
