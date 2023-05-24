from random import randint

import numpy as np

from fold.composites.columns import EnsembleEachColumn
from fold.composites.concat import Pipeline
from fold.composites.ensemble import Ensemble
from fold.loop import backtest, train
from fold.models.dummy import DummyClassifier
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_all_zeros, generate_sine_wave_data


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

    trained_pipelines = train(pipeline, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    assert (np.isclose((X.squeeze()[pred.index]), (pred.squeeze() + 2.0))).all()


def test_ensemble_classification() -> None:
    X, y = generate_all_zeros(1000)

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pipeline = [
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

    trained_pipelines = train(pipeline, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
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

    trained_pipelines = train(pipeline, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    expected = X.mean(axis=1) + 1.0
    assert (np.isclose(expected[pred.index].squeeze(), pred.squeeze())).all()


def test_dummy_pipeline_classification() -> None:
    X, y = generate_all_zeros(1000)

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pipeline = Pipeline(
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

    trained_pipelines = train(pipeline, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)

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

    trained_pipelines_baseline = train(pipeline_baseline, X, y, splitter)
    pred_baseline = backtest(trained_pipelines_baseline, X, y, splitter)

    assert (pred == pred_baseline).all(axis=0).all()
