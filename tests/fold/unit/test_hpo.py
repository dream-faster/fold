from sklearn.dummy import DummyRegressor as SklearnDummyRegressor

from fold.composites import SelectBest, TransformTarget
from fold.loop import train_backtest
from fold.models import DummyClassifier, DummyRegressor, WrapSKLearnRegressor
from fold.splitters import ExpandingWindowSplitter, ForwardSingleWindowSplitter
from fold.transformations import Difference
from fold.transformations.dev import Identity
from fold.transformations.difference import StationaryMethod
from fold.utils.tests import generate_monotonous_data
from fold_extensions.optimize_optuna import OptimizeOptuna


def test_grid_hpo() -> None:
    X, y = generate_monotonous_data(1000)

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pipeline = OptimizeOptuna(
        pipeline=[
            DummyClassifier(
                predicted_value=1.0,
                predicted_probabilities=[1.0, 2.0],
                all_classes=[1, 2, 3],
                params_to_try=dict(
                    predicted_value=[2.0, 3.0],
                    predicted_probabilities=[[5.0, 6.0], [3.0, 4.0]],
                ),
            ),
            DummyRegressor(
                predicted_value=3.0,
                params_to_try=dict(predicted_value=[22.0, 32.0]),
            ),
            Difference(method=StationaryMethod.difference),
        ],
        krisi_metric_key="mse",
        is_scorer_loss=True,
        trials=10,
        splitter=ForwardSingleWindowSplitter(0.6),
    )

    pred, trained_pipelines, _, _ = train_backtest(pipeline, X, y, splitter)


def test_gridsearch_sklearn() -> None:
    X, y = generate_monotonous_data(1000)

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pipeline = [
        OptimizeOptuna(
            pipeline=[
                Identity(),
                WrapSKLearnRegressor.from_model(
                    SklearnDummyRegressor(strategy="constant", constant=1),
                    params_to_try=dict(constant=[1, 2]),
                    name="dummy",
                ),
                WrapSKLearnRegressor.from_model(
                    SklearnDummyRegressor(strategy="constant", constant=1),
                    params_to_try=dict(constant=[1, 2]),
                    name="dummy2",
                ),
            ],
            krisi_metric_key="mse",
            is_scorer_loss=True,
            trials=30,
            splitter=ForwardSingleWindowSplitter(0.6),
        )
    ]

    pred, _, _, _ = train_backtest(pipeline, X, y, splitter)
    assert (pred.squeeze() == 1).all()


def test_grid_passthrough():
    X, y = generate_monotonous_data(1000)

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pipeline = OptimizeOptuna(
        pipeline=[
            Identity(),
            DummyRegressor(
                predicted_value=3.0,
                params_to_try=dict(
                    predicted_value=[22.0, 32.0], passthrough=[True, False]
                ),
            ),
        ],
        krisi_metric_key="mse",
        is_scorer_loss=True,
        trials=10,
        splitter=ForwardSingleWindowSplitter(0.6),
    )

    pred, trained, _, _ = train_backtest(pipeline, X, y, splitter)
    assert (X.loc[pred.index].squeeze() == pred.squeeze()).all()


def test_selectbest() -> None:
    X, y = generate_monotonous_data(1000)
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pipeline = OptimizeOptuna(
        [
            Identity(),
            SelectBest(
                [
                    DummyRegressor(
                        predicted_value=3.0,
                        name="3.0",
                    ),
                    DummyRegressor(
                        predicted_value=0.5,
                        name="0.5",
                    ),
                ]
            ),
            Identity(),
        ],
        krisi_metric_key="mse",
        is_scorer_loss=True,
        trials=10,
        splitter=ForwardSingleWindowSplitter(0.6),
    )
    pred, trained_pipelines, _, _ = train_backtest(pipeline, X, y, splitter)
    assert pred.squeeze()[0] == 0.5


def test_selectbest_nested():
    X, y = generate_monotonous_data(1000)
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pipeline = OptimizeOptuna(
        [
            Identity(),
            SelectBest(
                [
                    TransformTarget(
                        DummyRegressor(
                            predicted_value=1.0,
                        ),
                        Identity(name="identity-1.0"),
                        name="1.0",
                    ),
                    TransformTarget(
                        DummyRegressor(
                            predicted_value=0.5,
                        ),
                        Identity(name="identity-0.5"),
                        name="0.5",
                    ),
                ]
            ),
            Identity(),
        ],
        splitter=ForwardSingleWindowSplitter(0.6),
        trials=10,
        krisi_metric_key="mse",
        is_scorer_loss=True,
    )
    pred, _, _, _ = train_backtest(pipeline, X, y, splitter)
    assert pred.squeeze()[0] == 0.5
