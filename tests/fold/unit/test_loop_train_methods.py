import pandas as pd

from fold.base import Artifact
from fold.loop import train
from fold.loop.backtesting import backtest
from fold.models.base import Model
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.dev import Test
from fold.utils.tests import generate_sine_wave_data


class TestNoOverlap(Model):
    properties = Model.Properties(requires_X=False)
    name = "TestNoOverlap"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series | None = None,
        raw_y: pd.Series | None = None,
    ) -> Artifact | None:
        self.fit_index = X.index

    def update(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series | None = None,
        raw_y: pd.Series | None = None,
    ) -> Artifact | None:
        assert not any(i in self.fit_index for i in X.index)
        # append to fit_index
        self.update_index = X.index

    def predict(self, X: pd.DataFrame) -> pd.Series | pd.DataFrame:
        assert not any(i in self.fit_index for i in X.index)
        assert not any(i in self.predict_in_sample_index for i in X.index)
        return X

    def predict_in_sample(self, X: pd.DataFrame) -> pd.Series | pd.DataFrame:
        self.predict_in_sample_index = X.index
        assert all(i in self.fit_index for i in X.index)
        return X


def get_transformations_to_test():
    return [
        Test(fit_func=lambda x: x, transform_func=lambda x: x),
        TestNoOverlap(),
    ]


def test_loop_parallel_minibatch_call_times() -> None:
    X, y = generate_sine_wave_data()

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=200)
    trained, _, _ = train(
        get_transformations_to_test(),
        X,
        y,
        splitter,
    )

    assert trained.pipeline[0].iloc[0].no_of_calls_fit == 1
    assert trained.pipeline[0].iloc[0].no_of_calls_update == 0
    assert trained.pipeline[0].iloc[0].no_of_calls_transform_insample == 1
    assert trained.pipeline[0].iloc[0].no_of_calls_transform_outofsample == 0

    assert trained.pipeline[0].iloc[1].no_of_calls_fit == 1
    assert trained.pipeline[0].iloc[1].no_of_calls_update == 0
    assert trained.pipeline[0].iloc[1].no_of_calls_transform_insample == 1
    assert trained.pipeline[0].iloc[1].no_of_calls_transform_outofsample == 0

    assert trained.pipeline[0].iloc[2].no_of_calls_fit == 1
    assert trained.pipeline[0].iloc[2].no_of_calls_update == 0
    assert trained.pipeline[0].iloc[2].no_of_calls_transform_insample == 1
    assert trained.pipeline[0].iloc[2].no_of_calls_transform_outofsample == 0

    _ = backtest(trained, X, y, splitter, mutate=True)
    assert trained.pipeline[0].iloc[0].no_of_calls_fit == 1
    assert trained.pipeline[0].iloc[0].no_of_calls_update == 0
    assert trained.pipeline[0].iloc[0].no_of_calls_transform_insample == 1
    assert trained.pipeline[0].iloc[0].no_of_calls_transform_outofsample == 1

    assert trained.pipeline[0].iloc[1].no_of_calls_fit == 1
    assert trained.pipeline[0].iloc[1].no_of_calls_update == 0
    assert trained.pipeline[0].iloc[1].no_of_calls_transform_insample == 1
    assert trained.pipeline[0].iloc[1].no_of_calls_transform_outofsample == 1

    assert trained.pipeline[0].iloc[2].no_of_calls_fit == 1
    assert trained.pipeline[0].iloc[2].no_of_calls_update == 0
    assert trained.pipeline[0].iloc[2].no_of_calls_transform_insample == 1
    assert trained.pipeline[0].iloc[2].no_of_calls_transform_outofsample == 1
