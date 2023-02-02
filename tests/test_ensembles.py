import numpy as np

from drift.loop import infer, train
from drift.models.ensemble import Ensemble, PerColumnEnsemble
from drift.splitters import ExpandingWindowSplitter
from drift.utils.tests import generate_sine_wave_data


def test_ensemble() -> None:

    # the naive model returns X as prediction, so y.shift(1) should be == pred
    X = generate_sine_wave_data()
    y = X.shift(-1)

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [
        Ensemble(
            [
                lambda x: x - 1.0,
                lambda x: x - 2.0,
                lambda x: x - 3.0,
            ]
        ),
        lambda x: x.mean(axis=1),
    ]

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = infer(transformations_over_time, X, splitter)
    assert np.all(np.isclose((X.squeeze()[pred.index]).values, (pred + 2.0).values))


def test_per_column_transform() -> None:

    X = generate_sine_wave_data()
    X["sine_2"] = X["sine"] + 1.0
    X["sine_3"] = X["sine"] + 2.0
    X["sine_4"] = X["sine"] + 3.0
    y = X.shift(-1)

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [
        PerColumnEnsemble([lambda x: x + 1.0]),
    ]

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = infer(transformations_over_time, X, splitter)
    assert np.all(np.isclose((X["sine"][pred.index] + 2.5).values, pred.values))
