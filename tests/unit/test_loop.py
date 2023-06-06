import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from fold.base import Transformations
from fold.base.classes import Extras
from fold.base.scoring import score_results
from fold.events.labeling.fixed import SumFixedForwardHorizon
from fold.loop import train
from fold.loop.backtesting import backtest
from fold.loop.types import Backend, TrainMethod
from fold.models.baseline import Naive
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.dev import Test
from fold.transformations.lags import AddLagsY
from fold.utils.tests import (
    generate_all_zeros,
    generate_sine_wave_data,
    generate_zeros_and_ones,
)


def run_loop(
    train_method: TrainMethod, backend: Backend, transformations: Transformations
) -> None:
    # the naive model returns X as prediction, so y.shift(1) should be == pred
    X, y = generate_sine_wave_data(length=1000)

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    trained_pipelines = train(
        transformations,
        None,
        y,
        splitter,
        train_method=train_method,
        backend=backend,
        silent=False,
    )
    pred = backtest(trained_pipelines, X, y, splitter)
    assert (X.squeeze()[pred.index] == pred.squeeze()).all()


def test_loop_sequential():
    naive = Naive()
    run_loop(
        TrainMethod.sequential,
        Backend.no,
        naive,
    )


def test_loop_parallel():
    run_loop(
        TrainMethod.parallel,
        Backend.no,
        Naive(),
    )


def test_loop_online_model_no_minibatching_backtest():
    # _internal_supports_minibatch_backtesting is True by default for Naive model, but backtesting should work even if it's False
    naive = Naive()
    naive.properties._internal_supports_minibatch_backtesting = False
    run_loop(
        TrainMethod.parallel,
        Backend.no,
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
            Backend.no,
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


def test_sameple_weights() -> None:
    def assert_sample_weights_exist(X, y, sample_weight):
        assert sample_weight is not None
        assert sample_weight[0] == 0

    test_sample_weights_exist = Test(
        fit_func=assert_sample_weights_exist, transform_func=lambda X: X
    )

    X, y = generate_all_zeros(1000)
    sample_weights = y

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    transformations = [
        test_sample_weights_exist,
    ]
    _ = train(transformations, X, y, splitter, sample_weights=sample_weights)


def test_score_results():
    X, y = generate_zeros_and_ones(length=1000)

    results = pd.DataFrame({"x_predictions": y.squeeze()})
    sc = score_results(
        results, y, Extras(), artifacts=pd.DataFrame(), evaluation_func=accuracy_score
    )
    assert sc["accuracy"].result == 1.0

    back_shifted = y.shift(1)
    extras = Extras(
        events=SumFixedForwardHorizon(1).label_events(back_shifted.index, back_shifted)
    )
    assert (y[:-1] == extras.events.label).all()

    # test that if extras.events is not None, it is used for scoring
    sc = score_results(
        results,
        pd.Series(0, index=y.index),
        extras,
        artifacts=pd.DataFrame(),
        evaluation_func=accuracy_score,
    )
    assert sc["accuracy"].result == 1.0

    # test that there's a "label" in artifacts, it is used for scoring
    sc = score_results(
        results,
        pd.Series(0, index=y.index),
        Extras(),
        artifacts=extras.events,
        evaluation_func=accuracy_score,
    )
    assert sc["accuracy"].result == 1.0

    sc = score_results(
        results[200:],
        pd.Series(0, index=y.index),
        extras,
        artifacts=pd.DataFrame(),
        evaluation_func=accuracy_score,
    )
    assert sc["accuracy"].result == 1.0
    assert len(sc.y) == len(extras.events) - 200
