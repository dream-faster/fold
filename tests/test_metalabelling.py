from random import randint

import numpy as np
import pandas as pd

from drift.loop import backtest, train
from drift.models.base import Model
from drift.models.dummy import DummyClassifier
from drift.models.metalabeling import MetaLabeling
from drift.splitters import ExpandingWindowSplitter
from drift.utils.tests import generate_all_zeros, generate_sine_wave_data


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
    _, pred = backtest(transformations_over_time, X, y, splitter)
    assert np.all(np.isclose((X.squeeze()[pred.index]).values, (pred + 2.0).values))


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
    _, pred = backtest(transformations_over_time, X, y, splitter)
    assert (
        pred["probabilities_Ensemble-DummyClassifier-DummyClassifier-DummyClassifier_1"]
        == 0.5
    ).all()
    assert (
        pred["probabilities_Ensemble-DummyClassifier-DummyClassifier-DummyClassifier_0"]
        == 0.5
    ).all()
