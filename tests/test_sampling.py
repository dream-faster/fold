from typing import Optional, Union

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

from drift.loop import backtest, train
from drift.models.base import Model
from drift.models.dummy import DummyClassifier
from drift.splitters import ExpandingWindowSplitter
from drift.transformations.base import Transformation
from drift.transformations.sampling import Sampling
from drift.transformations.test import Test
from drift.utils.tests import generate_zeros_and_ones_skewed


def assert_on_fit(X, y):
    assert y[y == 1].sum() >= len(y) * 0.45
    assert len(y) < 90000


test_regressor = Test(fit_func=assert_on_fit, transform_func=lambda X: X)


def test_sampling() -> None:

    X = generate_zeros_and_ones_skewed(length=100000, labels=[1, 0], weights=[0.2, 0.8])
    y = X.squeeze()

    splitter = ExpandingWindowSplitter(train_window_size=90000, step=90000)
    transformations = [
        Sampling(RandomUnderSampler(), test_regressor),
    ]

    _ = train(transformations, X, y, splitter)
