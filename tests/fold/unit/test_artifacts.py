from sklearn.feature_selection import SelectKBest, f_regression

from fold.base.classes import PipelineCard
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
    pipeline = PipelineCard(
        preprocessing=None,
        pipeline=[
            SelectKBest(score_func=f_regression, k=1),
            DummyRegressor(predicted_value=1.0),
        ],
        event_labeler=FixedForwardHorizon(
            time_horizon=1, labeling_strategy=NoLabel(), weighting_strategy=None
        ),
        event_filter=EveryNth(2),
    )

    _, pred, _, artifacts, _ = train_evaluate(pipeline, X, y, splitter)
    assert artifacts is not None
    assert artifacts["selected_features_SelectKBest"].dropna().iloc[-1] == ["sine"]


def test_artifacts_events() -> None:
    X, y = generate_monotonous_data(1000, freq="1min")

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pipeline = PipelineCard(
        preprocessing=None,
        pipeline=DummyRegressor(
            predicted_value=1, params_to_try=dict(predicted_value=[100, 25, 50])
        ),
        event_labeler=FixedForwardHorizon(
            time_horizon=3,
            labeling_strategy=NoLabel(),
            weighting_strategy=NoWeighting(),
        ),
        event_filter=EveryNth(2),
    )
    _, artifacts, _ = train(pipeline, X, y, splitter)
    assert artifacts is not None
    assert artifacts.index.duplicated().sum() == 0
