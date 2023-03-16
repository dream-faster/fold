import numpy as np

from fold.loop import backtest, train
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.target import TransformTarget
from fold.transformations.test import Test
from fold.utils.tests import generate_zeros_and_ones_skewed


def all_y_values_above_1(X, y):
    assert np.all(y >= 1.0)


def all_y_values_below_1(X, y):
    assert np.all(y <= 1.0)


test_all_y_values_below_1 = Test(
    fit_func=all_y_values_below_1, transform_func=lambda X: X
)
test_all_y_values_above_1 = Test(
    fit_func=all_y_values_above_1, transform_func=lambda X: X
)
test_transform_plus_2 = Test(
    lambda x: x, lambda x: x + 2.0, inverse_transform_func=lambda x: x - 2.0
)


def test_target_transformation() -> None:
    X, y = generate_zeros_and_ones_skewed(length=1000, weights=[0.5, 0.5])

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)

    transformations = [
        TransformTarget(
            [lambda x: x + 1, test_all_y_values_above_1], test_transform_plus_2
        ),
        test_all_y_values_below_1,
    ]

    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert ((X.squeeze()[pred.index] + 1) == pred.squeeze()).all()
