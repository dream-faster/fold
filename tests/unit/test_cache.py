from sklearn.feature_selection import SelectKBest, f_regression

from fold.composites.optimize import OptimizeGridSearch
from fold.events import CreateEvents
from fold.events.filters.everynth import EveryNth
from fold.events.labeling import FixedForwardHorizon, NoLabel
from fold.events.weights import NoWeighing
from fold.loop import train, train_evaluate
from fold.models import DummyRegressor
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_monotonous_data, generate_sine_wave_data


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
                weighing_strategy=NoWeighing(),
            ),
            EveryNth(2),
        )
    ]
    _, artifacts = train(pipeline, X, y, splitter, return_artifacts=True)
    assert artifacts is not None
    assert artifacts.index.duplicated().sum() == 0
