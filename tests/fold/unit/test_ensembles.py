from random import randint

import numpy as np

from fold.composites.columns import EnsembleEachColumn
from fold.composites.concat import Sequence
from fold.composites.ensemble import Ensemble
from fold.loop.encase import train_backtest
from fold.models.dummy import DummyClassifier
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_sine_wave_data, generate_zeros_and_ones


def test_ensemble_regression() -> None:
    X, y = generate_sine_wave_data()

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pipeline = [
        Ensemble(
            [
                lambda x: (x - 1.0).rename({"sine": "predictions_1"}, axis=1),
                lambda x: (x - 2.0).rename({"sine": "predictions_2"}, axis=1),
                lambda x: (x - 3.0).rename({"sine": "predictions_3"}, axis=1),
            ]
        ),
    ]

    pred, _, _, _ = train_backtest(pipeline, X, y, splitter)
    assert np.allclose((X.squeeze()[pred.index]), (pred.squeeze() + 2.0))


def test_ensemble_classification() -> None:
    X, y = generate_zeros_and_ones(1000)

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pipeline = Ensemble(
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
        ],
        verbose=True,
    )

    pred, _, _, _ = train_backtest(pipeline, X, y, splitter)
    assert (
        pred[
            "probabilities_Ensemble-DummyClassifier-1-DummyClassifier-1-DummyClassifier-0_0"
        ]
        == 0.5
    ).all()
    assert (
        pred[
            "probabilities_Ensemble-DummyClassifier-1-DummyClassifier-1-DummyClassifier-0_1"
        ]
        == 0.5
    ).all()


def test_per_column_transform_predictions() -> None:
    X, y = generate_sine_wave_data()
    X["sine_2"] = X["sine"] + 1.0
    X["sine_3"] = X["sine"] + 2.0
    X["sine_4"] = X["sine"] + 3.0

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pipeline = [
        EnsembleEachColumn(
            lambda x: (x + 1.0)
            .squeeze()
            .rename(f"predictions_{randint(1, 1000)}")
            .to_frame()
        ),
    ]

    pred, _, _, _ = train_backtest(pipeline, X, y, splitter)
    expected = X.mean(axis=1) + 1.0
    assert np.allclose(expected[pred.index].squeeze(), pred.squeeze())


def test_dummy_pipeline_classification() -> None:
    X, y = generate_zeros_and_ones(1000)

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pipeline = Sequence(
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
    )

    pred, _, _, _ = train_backtest(pipeline, X, y, splitter)

    pipeline_baseline = [
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

    pred_baseline, _, _, _ = train_backtest(pipeline_baseline, X, y, splitter)

    assert (pred == pred_baseline).all(axis=0).all()
