from fold.loop import backtest, train
from fold.loop.types import Backend, TrainMethod
from fold.models.baseline import BaselineNaive, BaselineRegressorDeprecated
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.base import Transformations
from fold.transformations.test import Test
from fold.transformations.update import InjectPastDataAtInference
from fold.utils.tests import generate_all_zeros, generate_sine_wave_data


def run_loop(
    train_method: TrainMethod, backend: Backend, transformations: Transformations
) -> None:
    # the naive model returns X as prediction, so y.shift(1) should be == pred
    X, y = generate_sine_wave_data()

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations_over_time = train(
        transformations, X, y, splitter, train_method=train_method, backend=backend
    )
    pred = backtest(transformations_over_time, X, y, splitter)
    assert (X.squeeze()[pred.index] == pred.squeeze()).all()


def test_loop_sequential():
    run_loop(
        TrainMethod.sequential,
        Backend.no,
        InjectPastDataAtInference(
            BaselineRegressorDeprecated(
                strategy=BaselineRegressorDeprecated.Strategy.naive
            )
        ),
    )


def test_loop_parallel():
    run_loop(
        TrainMethod.parallel,
        Backend.no,
        InjectPastDataAtInference(
            BaselineRegressorDeprecated(
                strategy=BaselineRegressorDeprecated.Strategy.naive
            )
        ),
    )


def test_loop_with_continuous_transformation():
    run_loop(
        TrainMethod.parallel,
        Backend.no,
        BaselineNaive(),
    )


def test_sameple_weights() -> None:
    def assert_sample_weights_exist(X, y, sample_weight):
        assert sample_weight is not None
        assert sample_weight[0] == 0

    test_sample_weights_exist = Test(
        fit_func=assert_sample_weights_exist, transform_func=lambda X: X
    )

    X, y = generate_all_zeros(1000)
    sample_weights = y

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [
        test_sample_weights_exist,
    ]
    _ = train(transformations, X, y, splitter, sample_weights=sample_weights)
