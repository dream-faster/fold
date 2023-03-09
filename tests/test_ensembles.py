from random import randint

import numpy as np

from fold.loop import backtest, train
from fold.models.dummy import DummyClassifier
from fold.models.ensemble import Ensemble, PerColumnEnsemble
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_all_zeros, generate_sine_wave_data


def test_ensemble_regression() -> None:
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
    pred = backtest(transformations_over_time, X, y, splitter)
    assert (np.isclose((X.squeeze()[pred.index]), (pred.squeeze() + 2.0))).all()


def test_ensemble_classification() -> None:
    X = generate_all_zeros(1000)
    y = X.shift(-1).squeeze()

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [
        Ensemble(
            [
                DummyClassifier(
                    predicted_value=1,
                    all_classes=[1, 0],
                    predicted_probabilities=[1.0, 0.0],
                ),
                DummyClassifier(
                    predicted_value=1,
                    all_classes=[1, 0],
                    predicted_probabilities=[0.5, 0.5],
                ),
                DummyClassifier(
                    predicted_value=0,
                    all_classes=[1, 0],
                    predicted_probabilities=[0.0, 1.0],
                ),
            ]
        ),
    ]

    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert (
        pred["probabilities_Ensemble-DummyClassifier-DummyClassifier-DummyClassifier_1"]
        == 0.5
    ).all()
    assert (
        pred["probabilities_Ensemble-DummyClassifier-DummyClassifier-DummyClassifier_0"]
        == 0.5
    ).all()


def test_per_column_transform_predictions() -> None:
    X = generate_sine_wave_data()
    X["sine_2"] = X["sine"] + 1.0
    X["sine_3"] = X["sine"] + 2.0
    X["sine_4"] = X["sine"] + 3.0
    y = X["sine"].shift(-1).squeeze()

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [
        PerColumnEnsemble(
            lambda x: (x + 1.0)
            # .squeeze()
            .rename(f"predictions_{randint(1, 1000)}").to_frame()
        ),
    ]

    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    expected = X.mean(axis=1) + 1.0
    assert (np.isclose(expected[pred.index].squeeze(), pred.squeeze())).all()
