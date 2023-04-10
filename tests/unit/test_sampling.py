from imblearn.under_sampling import RandomUnderSampler

from fold.composites.sample import Sample
from fold.loop import train
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.dev import Test
from fold.utils.tests import generate_zeros_and_ones_skewed


def assert_on_fit(X, y):
    assert y[y == 1].sum() >= len(y) * 0.45
    assert len(y) < 90000


test_regressor = Test(fit_func=assert_on_fit, transform_func=lambda X: X)


def test_sampling() -> None:
    X, y = generate_zeros_and_ones_skewed(
        length=100000, labels=[1, 0], weights=[0.2, 0.8]
    )

    splitter = ExpandingWindowSplitter(initial_train_window=90000, step=90000)
    transformations = [
        Sample(RandomUnderSampler(), test_regressor),
    ]

    _ = train(transformations, X, y, splitter)
