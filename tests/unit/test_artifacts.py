from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error

from fold.composites.optimize import OptimizeGridSearch
from fold.events import CreateEvents, NoFilter
from fold.events.labeling import SumFixedForwardHorizon
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
            SumFixedForwardHorizon(time_horizon=1),
            NoFilter(),
        )
    ]

    _, pred, _, artifacts = train_evaluate(
        transformations, X, y, splitter, return_artifacts=True
    )
    assert artifacts is not None
    assert artifacts["selected_features"].dropna().iloc[-1] == ["sine"]


def test_artifacts_optimizer() -> None:
    X, y = generate_monotonous_data(1000)

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pipeline = [
        OptimizeGridSearch(
            pipeline=DummyRegressor(
                predicted_value=1, params_to_try=dict(predicted_value=[100, 25, 50])
            ),
            scorer=mean_squared_error,
            is_scorer_loss=True,
        )
    ]
    _, artifacts = train(pipeline, X, y, splitter, return_artifacts=True)
    assert artifacts is not None
    assert list(artifacts["selected_params"].iloc[0].values())[0] == {
        "predicted_value": 25
    }
