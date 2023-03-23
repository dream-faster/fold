from fold.loop import train
from fold.loop.backtest import backtest
from fold.loop.types import Backend, TrainMethod
from fold.models.baseline import Naive, Transformation
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.base import Transformations
from fold.transformations.dev import Test
from fold.utils.tests import generate_all_zeros, generate_sine_wave_data


def run_loop(
    train_method: TrainMethod, backend: Backend, transformations: Transformations
) -> None:
    # the naive model returns X as prediction, so y.shift(1) should be == pred
    X, y = generate_sine_wave_data(length=1000)

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    transformations_over_time = train(
        transformations,
        None,
        y,
        splitter,
        train_method=train_method,
        backend=backend,
        silent=False,
    )
    pred = backtest(transformations_over_time, X, y, splitter)
    assert (X.squeeze()[pred.index] == pred.squeeze()).all()


def test_loop_sequential():
    naive = Naive()
    naive.properties.mode = Transformation.Properties.Mode.online
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


def test_loop_with_online_transformation():
    run_loop(
        TrainMethod.parallel,
        Backend.no,
        Naive(),
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

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    transformations = [
        test_sample_weights_exist,
    ]
    _ = train(transformations, X, y, splitter, sample_weights=sample_weights)
