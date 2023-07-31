from sklearn.feature_selection import SelectKBest, f_regression

from fold.composites.optimize import OptimizeGridSearch
from fold.events import CreateEvents
from fold.events.filters.everynth import EveryNth
from fold.events.labeling import FixedForwardHorizon, NoLabel
from fold.events.weights import NoWeighting
from fold.loop import train, train_evaluate
from fold.models import DummyRegressor
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_monotonous_data, generate_sine_wave_data


def test_artifacts_transformation_fit() -> None:
    X, y = generate_sine_wave_data()
    X["sine_2"] = X["sine"] ** 2
    X["constant"] = 1.0
    X["constant2"] = 2.0

    splitter = ExpandingWindowSplitter(initial_train_window=100, step=100)
    transformations = [
        CreateEvents(
            [
                SelectKBest(score_func=f_regression, k=1),
                DummyRegressor(predicted_value=1.0),
            ],
            FixedForwardHorizon(
                time_horizon=1, labeling_strategy=NoLabel(), weighting_strategy=None
            ),
            EveryNth(2),
        )
    ]

    _, pred, _, artifacts = train_evaluate(
        transformations, X, y, splitter, return_artifacts=True
    )
    assert artifacts is not None
    assert artifacts["selected_features_SelectKBest"].dropna().iloc[-1] == ["sine"]


def test_artifacts_optimizer() -> None:
    X, y = generate_monotonous_data(1000)

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pipeline = [
        OptimizeGridSearch(
            pipeline=DummyRegressor(
                predicted_value=1, params_to_try=dict(predicted_value=[100, 25, 50])
            ),
            krisi_metric_key="mse",
            is_scorer_loss=True,
        )
    ]
    _, artifacts = train(pipeline, X, y, splitter, return_artifacts=True)
    assert artifacts is not None
    assert list(artifacts["selected_params"].dropna().iloc[0].values())[0] == {
        "predicted_value": 25
    }


def test_artifacts_events() -> None:
    X, y = generate_monotonous_data(1000, freq="1min")

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pipeline = [
        CreateEvents(
            DummyRegressor(
                predicted_value=1, params_to_try=dict(predicted_value=[100, 25, 50])
            ),
            FixedForwardHorizon(
                time_horizon=3,
                labeling_strategy=NoLabel(),
                weighting_strategy=NoWeighting(),
            ),
            EveryNth(2),
        )
    ]
    _, artifacts = train(pipeline, X, y, splitter, return_artifacts=True)
    assert artifacts is not None
    assert artifacts.index.duplicated().sum() == 0
