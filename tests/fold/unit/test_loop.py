import numpy as np
import pandas as pd
import pytest

from fold.base.classes import Artifact, Clonable, Pipeline, PipelineCard
from fold.base.scoring import score_results
from fold.base.utils import traverse_apply
from fold.composites.concat import Concat, Sequence
from fold.events.labeling.fixed import FixedForwardHorizon
from fold.events.labeling.strategies import NoLabel
from fold.loop import train
from fold.loop.backtesting import backtest
from fold.loop.encase import train_backtest
from fold.loop.types import TrainMethod
from fold.models.baseline import Naive
from fold.splitters import ExpandingWindowSplitter, SlidingWindowSplitter
from fold.transformations.dev import Identity, Test
from fold.transformations.features import AddWindowFeatures
from fold.transformations.function import ApplyFunction
from fold.transformations.lags import AddLagsX, AddLagsY
from fold.utils.dataset import get_preprocessed_dataset
from fold.utils.tests import generate_sine_wave_data, generate_zeros_and_ones

naive = Naive()


@pytest.mark.parametrize(
    {
        "train_method": [TrainMethod.sequential, TrainMethod.parallel],
        "transformations": [naive],
    }
)
def run_loop(train_method: TrainMethod, transformations: Pipeline) -> None:
    # the naive model returns X as prediction, so y.shift(1) should be == pred
    X, y = generate_sine_wave_data(length=1000)

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    trained_pipelines = train(
        transformations,
        None,
        y,
        splitter,
        train_method=train_method,
        silent=False,
    )
    pred = backtest(trained_pipelines, X, y, splitter)
    assert (X.squeeze()[pred.index] == pred.squeeze()).all()


def test_loop_online_model_no_minibatching_backtest():
    # _internal_supports_minibatch_backtesting is True by default for Naive model, but backtesting should work even if it's False
    naive = Naive()
    naive.properties._internal_supports_minibatch_backtesting = False
    run_loop(
        TrainMethod.parallel,
        naive,
    )


def test_loop_raises_error_if_requires_X_not_satified():
    naive = Naive()
    naive.properties.requires_X = True
    with pytest.raises(
        ValueError, match="X is None, but transformation Naive requires it."
    ):
        run_loop(
            TrainMethod.parallel,
            naive,
        )


def test_trim_na() -> None:
    _, y = generate_sine_wave_data(cycles=10, length=120, freq="M")

    def check_if_not_nan(x):
        assert not x.isna().any().any()

    splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
    transformations = [
        AddLagsY([1]),
        Test(fit_func=check_if_not_nan, transform_func=lambda X: X),
    ]
    transformations_over_time = train(transformations, None, y, splitter)
    _ = backtest(transformations_over_time, None, y, splitter)


def test_score_results():
    X, y = generate_zeros_and_ones(length=1000)

    results = pd.DataFrame({"x_predictions": y.squeeze()})
    sc, _ = score_results(
        results,
        y,
        artifacts=Artifact.empty(y.index),
    )

    assert sc["accuracy"].result == 1.0

    back_shifted = y.shift(1)
    events = FixedForwardHorizon(
        time_horizon=1, labeling_strategy=NoLabel(), weighting_strategy=None
    ).label_events(back_shifted.index, back_shifted)
    assert (y[:-1] == events.label).all()

    # test that there's a "label" in artifacts, it is used for scoring
    sc, _ = score_results(
        results[:-1],
        pd.Series(0, index=results[:-1].index),
        artifacts=events,
    )
    assert sc["accuracy"].result == 1.0


def test_disable_memory():
    X, y = get_preprocessed_dataset(
        "weather/historical_hourly_la",
        target_col="temperature",
        shorten=1000,
    )
    splitter = SlidingWindowSplitter(train_window=0.2, step=0.2)
    memory_size = 30

    def assert_len_can_be_divided_by_memory_size_2x(x, in_sample):
        if not in_sample:
            assert (len(x) - memory_size * 2) % 100 == 0

    def modify_pipeline(x, clone_children):
        if isinstance(x, Test):
            x.properties.disable_memory = False
            x.transform_func = assert_len_can_be_divided_by_memory_size_2x
        if isinstance(x, Clonable):
            return x.clone(clone_children)
        else:
            return x

    pipeline = Concat(
        [
            Concat(
                [
                    Sequence(
                        AddLagsX(columns_and_lags=[("pressure", list(range(1, 3)))])
                    ),
                    AddWindowFeatures(("pressure", 14, "mean")),
                    Test(
                        fit_func=lambda x: x,
                        transform_func=assert_len_can_be_divided_by_memory_size_2x,
                        memory_size=30,
                    ),
                ]
            ),
            AddWindowFeatures(("humidity", 26, "std")),
            Concat(
                [
                    ApplyFunction(lambda x: x.rolling(30).mean(), past_window_size=30),
                ]
            ),
        ]
    )

    pred, _ = train_backtest(
        traverse_apply(pipeline, modify_pipeline),
        X,
        y,
        splitter,
    )

    def assert_len_can_be_divided_by_memory_size(x, in_sample):
        if not in_sample:
            assert (len(x) - memory_size) % 100 == 0

    def modify_pipeline(x, clone_children):
        if isinstance(x, Test):
            x.properties.disable_memory = True
            x.transform_func = assert_len_can_be_divided_by_memory_size
        if isinstance(x, Clonable):
            return x.clone(clone_children)
        else:
            return x

    pred_disable_memory, _ = train_backtest(
        traverse_apply(pipeline, modify_pipeline),
        X,
        y,
        splitter,
    )
    assert np.allclose(pred_disable_memory, pred, atol=1e-20)


def test_preprocessing():
    X, y = get_preprocessed_dataset(
        "weather/historical_hourly_la",
        target_col="temperature",
        shorten=1000,
    )
    splitter = SlidingWindowSplitter(train_window=0.2, step=0.2)
    memory_size = 30

    def assert_len_can_be_divided_by_memory_size(x, in_sample):
        if not in_sample:
            assert (len(x) - memory_size) % 100 == 0

    test_trans = Test(
        fit_func=lambda x: x, transform_func=assert_len_can_be_divided_by_memory_size
    )
    test_trans.properties.memory_size = memory_size
    test_trans.properties.disable_memory = True
    pipeline = [
        Concat(
            [
                Concat(
                    [
                        Sequence(
                            AddLagsX(columns_and_lags=[("pressure", list(range(1, 3)))])
                        ),
                        AddWindowFeatures(("pressure", 14, "mean")),
                        test_trans,
                    ]
                ),
                AddWindowFeatures(("humidity", 26, "std")),
                Concat(
                    [
                        ApplyFunction(
                            lambda x: x.rolling(30).mean(), past_window_size=30
                        ),
                    ]
                ),
            ]
        ),
        Identity(),
    ]

    def assert_len_can_be_divided_by_window_size(x, in_sample):
        if not in_sample:
            assert len(x) == len(X)

    test_trans_preprocessing = Test(
        fit_func=lambda x: x, transform_func=assert_len_can_be_divided_by_window_size
    )
    test_trans_preprocessing.properties.memory_size = memory_size
    test_trans_preprocessing.properties.disable_memory = True

    equivalent_preprocessing_pipeline = [
        Concat(
            [
                Concat(
                    [
                        Sequence(
                            AddLagsX(columns_and_lags=[("pressure", list(range(1, 3)))])
                        ),
                        AddWindowFeatures(("pressure", 14, "mean")),
                        test_trans_preprocessing,
                    ]
                ),
                AddWindowFeatures(("humidity", 26, "std")),
                Concat(
                    [
                        ApplyFunction(
                            lambda x: x.rolling(30).mean(), past_window_size=30
                        ),
                    ]
                ),
            ]
        ),
        Identity(),
    ]

    pred, trained, insample = train_backtest(
        pipeline, X, y, splitter, return_insample=True
    )

    pred_preprocessing, trained_preprocessing, preprocessing_insample = train_backtest(
        PipelineCard(
            preprocessing=equivalent_preprocessing_pipeline,
            pipeline=[Identity()],  # MinMaxScaler()
        ),
        X,
        y,
        splitter,
        return_insample=True,
    )
    # insample results should be different, as AddLagsX's first values are gonna be 0.0s in-sample,
    # also 0.0s may skew the scaler
    assert not np.allclose(insample, preprocessing_insample, atol=1e-2)

    assert np.allclose(pred_preprocessing, pred, atol=1e-20)
