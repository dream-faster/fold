import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.linear_model import LinearRegression

from drift.loop import infer, train
from drift.models import Baseline, BaselineStrategy, Ensemble
from drift.splitters import ExpandingWindowSplitter
from drift.transformations import Concat, Identity, SelectColumns
from drift.transformations.columns import TransformColumn
from tests.utils import generate_sine_wave_data


def test_baseline_naive_model() -> None:

    y = generate_sine_wave_data()
    X = y.shift(1)

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    pipeline = [
        lambda x: x + 1,
        Concat([Identity(), Identity()]),
        SelectColumns("sine"),
        Ensemble(
            [
                Baseline(strategy=BaselineStrategy.naive),
                Baseline(strategy=BaselineStrategy.naive),
            ]
        ),
    ]

    # transformations_over_time = train(pipeline, X, y, splitter)
    # _, pred = infer(transformations_over_time, X, splitter)

    X = generate_sine_wave_data()
    X["sine_2"] = X["sine"] + 1
    X["constant"] = 1.0
    X["constant2"] = 2.0
    y = X["sine"].shift(-1)

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [
        VarianceThreshold(),
        TransformColumn("sine_2", [lambda x: x + 1.0]),
        SelectKBest(k=1),
        Ensemble([LinearRegression(), RandomForestRegressor()]),
    ]

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = infer(transformations_over_time, X, splitter)


test_baseline_naive_model()
