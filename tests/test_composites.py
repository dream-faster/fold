import pandas as pd

from fold.composites import Concat, PerColumnTransform
from fold.loop import backtest, train
from fold.loop.common import deepcopy_pipelines
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.columns import DropColumns, RenameColumns
from fold.transformations.dev import Identity
from fold.transformations.lags import AddLagsY
from fold.transformations.window import AddWindowFeatures
from fold.utils.tests import generate_sine_wave_data


def test_composite_cloning():
    instance = PerColumnTransform([lambda x: x + 1, lambda x: x + 2])
    clone = instance.clone(clone_child_transformations=deepcopy_pipelines)
    assert instance is not clone
    assert instance.pipeline is not clone.pipeline
    assert len(instance.pipeline[0]) == 2
    assert len(clone.pipeline[0]) == 2


def test_concat():
    X, y = generate_sine_wave_data()
    pipeline1 = AddWindowFeatures(("sine", 10, "mean"))
    pipeline2 = AddLagsY([1])
    concat = Concat([pipeline1, pipeline2], if_duplicate_keep="left")
    splitter = ExpandingWindowSplitter(0.2, 0.1)
    trained_pipelines = train(concat, X, y, splitter)
    preds = backtest(trained_pipelines, X, y, splitter)
    assert preds["y_lag_1"] is not None
    assert preds["sine_10_mean"] is not None


def test_concat_resolution_left():
    X, y = generate_sine_wave_data()
    pipeline1 = [
        AddWindowFeatures(("sine", 10, "mean")),
        DropColumns(["sine"]),
        RenameColumns({"sine_10_mean": "sine"}),
    ]
    pipeline2 = [
        AddWindowFeatures(("sine", 10, "mean")),
        DropColumns(["sine"]),
    ]
    concat = Concat([pipeline1, pipeline2], if_duplicate_keep="left")
    splitter = ExpandingWindowSplitter(0.2, 0.1)
    trained_pipelines = train(concat, X, y, splitter)
    preds = backtest(trained_pipelines, X, y, splitter)
    assert isinstance(preds["sine"], pd.Series)
    assert (preds["sine"] == preds["sine_10_mean"]).all()


def test_concat_resolution_right():
    X, y = generate_sine_wave_data()
    pipeline1 = [
        AddWindowFeatures(("sine", 10, "mean")),
        DropColumns(["sine"]),
        RenameColumns({"sine_10_mean": "sine"}),
    ]
    concat = Concat([pipeline1, Identity()], if_duplicate_keep="right")
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
        RenameColumns({"sine_10_mean": "sine"}),
    ]
    concat = Concat([pipeline1, Identity()], if_duplicate_keep="both")
    splitter = ExpandingWindowSplitter(0.2, 0.1)
    trained_pipelines = train(concat, X, y, splitter)
    preds = backtest(trained_pipelines, X, y, splitter)
    assert isinstance(preds["sine"], pd.DataFrame)
