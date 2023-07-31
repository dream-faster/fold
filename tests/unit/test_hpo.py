from sklearn.dummy import DummyRegressor as SklearnDummyRegressor

from fold.composites import OptimizeGridSearch, SelectBest, TransformTarget
from fold.loop import train_backtest
from fold.models import DummyClassifier, DummyRegressor, WrapSKLearnRegressor
from fold.splitters import ExpandingWindowSplitter
from fold.transformations import Difference
from fold.transformations.dev import Identity
from fold.utils.tests import generate_monotonous_data


def test_grid_hpo() -> None:
    X, y = generate_monotonous_data(1000)

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pipeline = OptimizeGridSearch(
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
            Difference(),
        ],
        krisi_metric_key="mse",
        is_scorer_loss=True,
    )

    pred, trained_pipelines = train_backtest(pipeline, X, y, splitter)
    assert len(trained_pipelines.pipeline[0].iloc[0].param_permutations) > 4


def test_gridsearch_sklearn() -> None:
    X, y = generate_monotonous_data(1000)

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pipeline = [
        OptimizeGridSearch(
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
        )
    ]

    pred, _ = train_backtest(pipeline, X, y, splitter)
    assert (pred.squeeze() == 1).all()


def test_grid_passthrough():
    X, y = generate_monotonous_data(1000)

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pipeline = OptimizeGridSearch(
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
    )

    pred, trained = train_backtest(pipeline, X, y, splitter)
    assert len(trained.pipeline[0].iloc[0].param_permutations) >= 4
    assert (X.loc[pred.index].squeeze() == pred.squeeze()).all()


def test_selectbest() -> None:
    X, y = generate_monotonous_data(1000)
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pipeline = OptimizeGridSearch(
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
    )
    pred, trained_pipelines = train_backtest(pipeline, X, y, splitter)
    assert pred.squeeze()[0] == 0.5


def test_selectbest_nested():
    X, y = generate_monotonous_data(1000)
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pipeline = OptimizeGridSearch(
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
        krisi_metric_key="mse",
        is_scorer_loss=True,
    )
    pred, _ = train_backtest(pipeline, X, y, splitter)
    assert pred.squeeze()[0] == 0.5
