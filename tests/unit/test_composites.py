import numpy as np
import pandas as pd

from fold.composites import Concat, TransformEachColumn
from fold.composites.imbalance import FindThreshold
from fold.composites.residual import ModelResiduals
from fold.loop import backtest, train
from fold.loop.utils import deepcopy_pipelines
from fold.models.dummy import DummyClassifier, DummyRegressor
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.columns import DropColumns, OnlyPredictions, RenameColumns
from fold.transformations.dev import Identity
from fold.transformations.features import AddWindowFeatures
from fold.transformations.lags import AddLagsY
from fold.utils.checks import get_prediction_column, get_probabilities_columns
from fold.utils.tests import (
    generate_monotonous_data,
    generate_sine_wave_data,
    generate_zeros_and_ones_skewed,
)


def test_composite_cloning():
    instance = TransformEachColumn([lambda x: x + 1, lambda x: x + 2])
    clone = instance.clone(clone_children=deepcopy_pipelines)
    assert instance is not clone
    assert instance.pipeline is not clone.pipeline
    assert len(instance.pipeline[0]) == 2
    assert len(clone.pipeline[0]) == 2


def test_concat_and_metadata():
    X, y = generate_sine_wave_data()
    pipeline1 = AddWindowFeatures(("sine", 10, "mean"))
    pipeline2 = AddLagsY([1])
    concat = Concat([pipeline1, pipeline2], if_duplicate_keep="first")
    splitter = ExpandingWindowSplitter(0.2, 0.1)
    trained_pipelines = train(concat, X, y, splitter)
    preds = backtest(trained_pipelines, X, y, splitter)
    assert preds["y_lag_1"] is not None
    assert preds["sine~10_mean"] is not None
    for i in range(0, 8):
        assert trained_pipelines[0].iloc[i].metadata.fold_index == i


def test_concat_resolution_left():
    X, y = generate_sine_wave_data()
    pipeline1 = [
        AddWindowFeatures(("sine", 10, "mean")),
        DropColumns(["sine"]),
        RenameColumns({"sine~10_mean": "sine"}),
    ]
    pipeline2 = [
        AddWindowFeatures(("sine", 10, "mean")),
        DropColumns(["sine"]),
    ]
    concat = Concat([pipeline1, pipeline2], if_duplicate_keep="first")
    splitter = ExpandingWindowSplitter(0.2, 0.1)
    trained_pipelines = train(concat, X, y, splitter)

    preds = backtest(trained_pipelines, X, y, splitter)
    assert isinstance(preds["sine"], pd.Series)
    assert (preds["sine"] == preds["sine~10_mean"]).all()


def test_concat_resolution_right():
    X, y = generate_sine_wave_data()
    pipeline1 = [
        AddWindowFeatures(("sine", 10, "mean")),
        DropColumns(["sine"]),
        RenameColumns({"sine_10_mean": "sine"}),
    ]
    concat = Concat([pipeline1, Identity()], if_duplicate_keep="last")
    splitter = ExpandingWindowSplitter(0.2, 0.1)
    trained_pipelines = train(concat, X, y, splitter)
    preds = backtest(trained_pipelines, X, y, splitter)
    assert isinstance(preds["sine"], pd.Series)
    assert (preds["sine"] == X.loc[preds.index]["sine"]).all()


def test_concat_resolution_both():
    X, y = generate_sine_wave_data()
    pipeline1 = [
        AddWindowFeatures(("sine", 10, "mean")),
        DropColumns(["sine"]),
        RenameColumns({"sine~10_mean": "sine"}),
    ]
    concat = Concat([pipeline1, Identity()], if_duplicate_keep="both")
    splitter = ExpandingWindowSplitter(0.2, 0.1)
    trained_pipelines = train(concat, X, y, splitter)
    preds = backtest(trained_pipelines, X, y, splitter)
    assert isinstance(preds["sine"], pd.DataFrame)


def test_model_residuals() -> None:
    X, y = generate_monotonous_data(1000)

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    transformations = [
        ModelResiduals(
            primary=[
                lambda x: x,
                DummyRegressor(
                    predicted_value=0.5,
                ),
            ],
            meta=[
                lambda x: x,
                DummyRegressor(
                    predicted_value=0.5,
                ),
            ],
        ),
        OnlyPredictions(),
    ]

    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    assert (pred.squeeze() == 1.0).all()


def test_imbalance():
    X, y = generate_zeros_and_ones_skewed(
        length=100000, labels=[1, 0], weights=[0.2, 0.8]
    )

    transformations_dummy = [FindThreshold([DummyClassifier(1, [0, 1], [0.2, 0.8])])]
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    trained_pipelines = train(transformations_dummy, X, y.astype(int), splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    assert (get_prediction_column(pred) == 1.0).all()

    def close_enough_predictor(X: pd.DataFrame) -> pd.DataFrame:
        preds = pd.DataFrame([], index=y.index)
        preds["predictions_"] = y
        random_idx = np.random.randint(0, len(y) - 1, size=len(y) // 10)
        preds["predictions_"].iloc[random_idx] = abs(y.iloc[random_idx] - 1)
        preds["probabilities_0"] = 0.2
        preds["probabilities_1"] = 0.2
        preds["probabilities_0"].iloc[preds["predictions_"] == 0] = 0.8
        preds["probabilities_1"].iloc[preds["predictions_"] == 1] = 0.8
        return preds

    transformations_close_enough_predictor = [FindThreshold([close_enough_predictor])]
    trained_pipelines = train(
        transformations_close_enough_predictor, X, y.astype(int), splitter
    )
    pred = backtest(trained_pipelines, X, y, splitter)
    assert (
        get_prediction_column(pred)[get_probabilities_columns(pred).iloc[:, 1] >= 0.8]
        == 1.0
    ).all()
