from random import randint

import numpy as np
import pandas as pd

from drift.loop import backtest, train
from drift.models.base import Model
from drift.models.ensemble import Ensemble, PerColumnEnsemble
from drift.splitters import ExpandingWindowSplitter
from drift.utils.tests import generate_all_zeros, generate_sine_wave_data


class DummyClassifier(Model):

    name = "DummyClassifier"

    def __init__(self, predicted_value, all_classes, predicted_probabilities) -> None:
        self.predicted_value = predicted_value
        self.all_classes = all_classes
        self.predicted_probabilities = predicted_probabilities

    def fit(self, X, y):
        pass

    def predict(self, X):
        predictions = pd.Series(
            [self.predicted_value] * len(X),
            index=X.index,
            name="predictions_DummyClassifier",
        )
        probabilities = [
            pd.Series(
                [prob] * len(X),
                index=X.index,
                name=f"probabilities_DummyClassifier_{associated_class}",
            )
            for associated_class, prob in zip(
                self.all_classes, self.predicted_probabilities
            )
        ]

        return pd.concat([predictions] + probabilities, axis=1)


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
        pred[
            "probabilities_Ensemble-DummyClassifier-DummyClassifier-DummyClassifier_1"
        ]
        == 0.5
    ).all()
    assert (
        pred[
            "probabilities_Ensemble-DummyClassifier-DummyClassifier-DummyClassifier_0"
        ]
        == 0.5
    ).all()


def test_per_column_transform_predictions() -> None:

    X = generate_sine_wave_data()
    X["sine_2"] = X["sine"] + 1.0
    X["sine_3"] = X["sine"] + 2.0
    X["sine_4"] = X["sine"] + 3.0
    y = X.shift(-1).squeeze()

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [
        PerColumnEnsemble(
            lambda x: (x + 1.0)
            .squeeze()
            .rename(f"predictions_{randint(1, 1000)}")
            .to_frame()
        ),
    ]

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = backtest(transformations_over_time, X, y, splitter)
    expected = X.mean(axis=1) + 1.0
    assert np.all(
        np.isclose(expected[pred.index].squeeze().values, pred.squeeze().values)
    )
