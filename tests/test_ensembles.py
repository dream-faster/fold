from random import randint

import numpy as np

from drift.loop import backtest, train
from drift.models.ensemble import Ensemble, PerColumnEnsemble
from drift.splitters import ExpandingWindowSplitter
from drift.utils.tests import generate_sine_wave_data


def test_ensemble_regression() -> None:

    # the naive model returns X as prediction, so y.shift(1) should be == pred
    X = generate_sine_wave_data()
    y = X.shift(-1).squeeze()

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [
        Ensemble(
            [
                lambda x: (x - 1.0).rename({"sine": "predictions_1"}, axis=1),
                lambda x: (x - 2.0).rename({"sine": "predictions_2"}, axis=1),
                lambda x: (x - 3.0).rename({"sine": "predictions_3"}, axis=1),
            ]
        ),
    ]

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = backtest(transformations_over_time, X, y, splitter)
    assert np.all(np.isclose((X.squeeze()[pred.index]).values, (pred + 2.0).values))


def test_per_column_transform() -> None:

    X = generate_sine_wave_data()
    X["sine_2"] = X["sine"] + 1.0
    X["sine_3"] = X["sine"] + 2.0
    X["sine_4"] = X["sine"] + 3.0
    y = X.shift(-1).squeeze()

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [
        PerColumnEnsemble(
            lambda x: (x + 1.0).squeeze().rename(f"predictions_{randint(1, 1000)}").to_frame()
        ),
    ]

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = backtest(transformations_over_time, X, y, splitter)
    expected = X.mean(axis=1) + 1.0
    assert np.all(
        np.isclose(expected[pred.index].squeeze().values, pred.squeeze().values)
    )
