from typing import Optional, Union

import pandas as pd

from fold.loop import train
from fold.loop.backtesting import backtest
from fold.loop.types import TrainMethod
from fold.models.base import Model
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.base import Transformation
from fold.transformations.dev import Test
from fold.utils.tests import generate_sine_wave_data


class TestNoOverlap(Model):
    properties = Model.Properties()
    name = "TestNoOverlap"

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        self.fit_index = X.index

    def update(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        assert not any([i in self.fit_index for i in X.index])
        # append to fit_index
        self.update_index = X.index

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        assert not any([i in self.fit_index for i in X.index])
        assert not any([i in self.predict_in_sample_index for i in X.index])
        return X

    def predict_in_sample(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        self.predict_in_sample_index = X.index
        assert all([i in self.fit_index for i in X.index])
        return X


def get_transformations_to_test(mode):
    t = [
        Test(fit_func=lambda x: x, transform_func=lambda x: x),
        TestNoOverlap(),
    ]
    t[0].properties.mode = mode
    t[1].properties.mode = mode
    return t


def test_loop_parallel_minibatch_call_times() -> None:
    X, y = generate_sine_wave_data()

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=200)
    transformations_over_time = train(
        get_transformations_to_test(mode=Transformation.Properties.Mode.minibatch),
        X,
        y,
        splitter,
        train_method=TrainMethod.parallel,
    )

    assert transformations_over_time[0].iloc[0].no_of_calls_fit == 1
    assert transformations_over_time[0].iloc[0].no_of_calls_update == 0
    assert transformations_over_time[0].iloc[0].no_of_calls_transform_insample == 1
    assert transformations_over_time[0].iloc[0].no_of_calls_transform_outofsample == 0

    assert transformations_over_time[0].iloc[1].no_of_calls_fit == 1
    assert transformations_over_time[0].iloc[1].no_of_calls_update == 0
    assert transformations_over_time[0].iloc[1].no_of_calls_transform_insample == 1
    assert transformations_over_time[0].iloc[1].no_of_calls_transform_outofsample == 0

    assert transformations_over_time[0].iloc[2].no_of_calls_fit == 1
    assert transformations_over_time[0].iloc[2].no_of_calls_update == 0
    assert transformations_over_time[0].iloc[2].no_of_calls_transform_insample == 1
    assert transformations_over_time[0].iloc[2].no_of_calls_transform_outofsample == 0

    _ = backtest(transformations_over_time, X, y, splitter, mutate=True)
    assert transformations_over_time[0].iloc[0].no_of_calls_fit == 1
    assert transformations_over_time[0].iloc[0].no_of_calls_update == 0
    assert transformations_over_time[0].iloc[0].no_of_calls_transform_insample == 1
    assert transformations_over_time[0].iloc[0].no_of_calls_transform_outofsample == 1

    assert transformations_over_time[0].iloc[1].no_of_calls_fit == 1
    assert transformations_over_time[0].iloc[1].no_of_calls_update == 0
    assert transformations_over_time[0].iloc[1].no_of_calls_transform_insample == 1
    assert transformations_over_time[0].iloc[1].no_of_calls_transform_outofsample == 1

    assert transformations_over_time[0].iloc[2].no_of_calls_fit == 1
    assert transformations_over_time[0].iloc[2].no_of_calls_update == 0
    assert transformations_over_time[0].iloc[2].no_of_calls_transform_insample == 1
    assert transformations_over_time[0].iloc[2].no_of_calls_transform_outofsample == 1


def test_loop_parallel_online_call_times() -> None:
    X, y = generate_sine_wave_data()

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=200)
    transformations_over_time = train(
        get_transformations_to_test(mode=Transformation.Properties.Mode.online),
        X,
        y,
        splitter,
        train_method=TrainMethod.parallel,
    )

    assert transformations_over_time[0].iloc[0].no_of_calls_fit == 1
    assert transformations_over_time[0].iloc[0].no_of_calls_update == 0
    assert transformations_over_time[0].iloc[0].no_of_calls_transform_insample == 1
    assert transformations_over_time[0].iloc[0].no_of_calls_transform_outofsample == 0

    assert transformations_over_time[0].iloc[1].no_of_calls_fit == 1
    assert transformations_over_time[0].iloc[1].no_of_calls_update == 0
    assert transformations_over_time[0].iloc[1].no_of_calls_transform_insample == 1
    assert transformations_over_time[0].iloc[1].no_of_calls_transform_outofsample == 0

    assert transformations_over_time[0].iloc[2].no_of_calls_fit == 1
    assert transformations_over_time[0].iloc[2].no_of_calls_update == 0
    assert transformations_over_time[0].iloc[2].no_of_calls_transform_insample == 1
    assert transformations_over_time[0].iloc[2].no_of_calls_transform_outofsample == 0

    _ = backtest(transformations_over_time, X, y, splitter, mutate=True)
    assert transformations_over_time[0].iloc[0].no_of_calls_fit == 1
    assert transformations_over_time[0].iloc[0].no_of_calls_update == 200
    assert transformations_over_time[0].iloc[0].no_of_calls_transform_insample == 1
    assert transformations_over_time[0].iloc[0].no_of_calls_transform_outofsample == 200

    assert transformations_over_time[0].iloc[1].no_of_calls_fit == 1
    assert transformations_over_time[0].iloc[1].no_of_calls_update == 200
    assert transformations_over_time[0].iloc[1].no_of_calls_transform_insample == 1
    assert transformations_over_time[0].iloc[1].no_of_calls_transform_outofsample == 200

    assert transformations_over_time[0].iloc[2].no_of_calls_fit == 1
    assert transformations_over_time[0].iloc[2].no_of_calls_update == 200
    assert transformations_over_time[0].iloc[2].no_of_calls_transform_insample == 1
    assert transformations_over_time[0].iloc[2].no_of_calls_transform_outofsample == 200


def test_loop_sequential_minibatch_call_times() -> None:
    X, y = generate_sine_wave_data()

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=200)
    transformations_over_time = train(
        get_transformations_to_test(mode=Transformation.Properties.Mode.minibatch),
        X,
        y,
        splitter,
        train_method=TrainMethod.sequential,
    )

    assert transformations_over_time[0].iloc[0].no_of_calls_fit == 1
    assert transformations_over_time[0].iloc[0].no_of_calls_update == 0
    assert transformations_over_time[0].iloc[0].no_of_calls_transform_insample == 1
    assert transformations_over_time[0].iloc[0].no_of_calls_transform_outofsample == 0

    assert transformations_over_time[0].iloc[1].no_of_calls_fit == 1
    assert transformations_over_time[0].iloc[1].no_of_calls_update == 1
    assert transformations_over_time[0].iloc[1].no_of_calls_transform_insample == 1
    assert transformations_over_time[0].iloc[1].no_of_calls_transform_outofsample == 1

    assert transformations_over_time[0].iloc[2].no_of_calls_fit == 1
    assert transformations_over_time[0].iloc[2].no_of_calls_update == 2
    assert transformations_over_time[0].iloc[2].no_of_calls_transform_insample == 1
    assert transformations_over_time[0].iloc[2].no_of_calls_transform_outofsample == 2

    _ = backtest(transformations_over_time, X, y, splitter, mutate=True)
    assert transformations_over_time[0].iloc[0].no_of_calls_fit == 1
    assert transformations_over_time[0].iloc[0].no_of_calls_update == 0
    assert transformations_over_time[0].iloc[0].no_of_calls_transform_insample == 1
    assert transformations_over_time[0].iloc[0].no_of_calls_transform_outofsample == 1

    assert transformations_over_time[0].iloc[1].no_of_calls_fit == 1
    assert transformations_over_time[0].iloc[1].no_of_calls_update == 1
    assert transformations_over_time[0].iloc[1].no_of_calls_transform_insample == 1
    assert transformations_over_time[0].iloc[1].no_of_calls_transform_outofsample == 2

    assert transformations_over_time[0].iloc[2].no_of_calls_fit == 1
    assert transformations_over_time[0].iloc[2].no_of_calls_update == 2
    assert transformations_over_time[0].iloc[2].no_of_calls_transform_insample == 1
    assert transformations_over_time[0].iloc[2].no_of_calls_transform_outofsample == 3


def test_loop_method_sequential_online_call_times() -> None:
    X, y = generate_sine_wave_data()

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=200)
    transformations_over_time = train(
        get_transformations_to_test(mode=Transformation.Properties.Mode.online),
        X,
        y,
        splitter,
        train_method=TrainMethod.sequential,
    )

    assert transformations_over_time[0].iloc[0].no_of_calls_fit == 1
    assert transformations_over_time[0].iloc[0].no_of_calls_update == 0
    assert transformations_over_time[0].iloc[0].no_of_calls_transform_insample == 1
    assert transformations_over_time[0].iloc[0].no_of_calls_transform_outofsample == 0

    assert transformations_over_time[0].iloc[1].no_of_calls_fit == 1
    assert transformations_over_time[0].iloc[1].no_of_calls_update == 200
    assert transformations_over_time[0].iloc[1].no_of_calls_transform_insample == 1
    assert transformations_over_time[0].iloc[1].no_of_calls_transform_outofsample == 200

    assert transformations_over_time[0].iloc[2].no_of_calls_fit == 1
    assert transformations_over_time[0].iloc[2].no_of_calls_update == 400
    assert transformations_over_time[0].iloc[2].no_of_calls_transform_insample == 1
    assert transformations_over_time[0].iloc[2].no_of_calls_transform_outofsample == 400

    _ = backtest(transformations_over_time, X, y, splitter, mutate=True)
    assert transformations_over_time[0].iloc[0].no_of_calls_fit == 1
    assert transformations_over_time[0].iloc[0].no_of_calls_update == 200
    assert transformations_over_time[0].iloc[0].no_of_calls_transform_insample == 1
    assert transformations_over_time[0].iloc[0].no_of_calls_transform_outofsample == 200

    assert transformations_over_time[0].iloc[1].no_of_calls_fit == 1
    assert transformations_over_time[0].iloc[1].no_of_calls_update == 400
    assert transformations_over_time[0].iloc[1].no_of_calls_transform_insample == 1
    assert transformations_over_time[0].iloc[1].no_of_calls_transform_outofsample == 400

    assert transformations_over_time[0].iloc[2].no_of_calls_fit == 1
    assert transformations_over_time[0].iloc[2].no_of_calls_update == 600
    assert transformations_over_time[0].iloc[2].no_of_calls_transform_insample == 1
    assert transformations_over_time[0].iloc[2].no_of_calls_transform_outofsample == 600
