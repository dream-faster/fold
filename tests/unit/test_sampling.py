from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from fold.composites.sample import Sample
from fold.loop import train_backtest
from fold.models.dummy import DummyRegressor
from fold.splitters import SingleWindowSplitter
from fold.transformations.dev import Test
from fold.utils.tests import generate_zeros_and_ones_skewed


def assert_on_fit_under(X, y):
    assert y[y == 1].sum() >= len(y) * 0.45
    assert len(y) < 90000


def assert_on_fit_over(X, y):
    assert y[y == 1].sum() >= len(y) * 0.45
    assert len(y) > 90000


def test_sampling_under() -> None:
    X, y = generate_zeros_and_ones_skewed(
        length=100000, labels=[1, 0], weights=[0.2, 0.8]
    )

    test_regressor = Test(fit_func=assert_on_fit_under, transform_func=lambda X: X)

    splitter = SingleWindowSplitter(train_window=90000)
    pipeline = [
        Sample(
            RandomUnderSampler(), [test_regressor, DummyRegressor(predicted_value=1.0)]
        ),
    ]

    pred, _ = train_backtest(pipeline, X, y, splitter)
    assert len(pred) == 10000


def test_sampling_over() -> None:
    X, y = generate_zeros_and_ones_skewed(
        length=100000, labels=[1, 0], weights=[0.2, 0.8]
    )

    test_regressor = Test(fit_func=assert_on_fit_over, transform_func=lambda X: X)

    splitter = SingleWindowSplitter(train_window=90000)
    pipeline = [
        Sample(
            RandomOverSampler(), [test_regressor, DummyRegressor(predicted_value=1.0)]
        ),
    ]

    pred, _ = train_backtest(pipeline, X, y, splitter)
    assert len(pred) == 10000
