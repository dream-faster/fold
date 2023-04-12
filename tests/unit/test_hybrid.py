from fold.composites.hybrid import Hybrid
from fold.loop import backtest, train
from fold.models.dummy import DummyRegressor
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.columns import OnlyPredictions
from fold.utils.tests import generate_monotonous_data


def test_hybrid() -> None:
    X, y = generate_monotonous_data(1000)

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    transformations = [
        Hybrid(
            primary=[
                lambda x: x,
                DummyRegressor(
                    predicted_value=0.5,
                ),
            ],
            meta=[
                lambda x: x,
                DummyRegressor(
                    predicted_value=0.5,
                ),
            ],
        ),
        OnlyPredictions(),
    ]

    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    assert (pred.squeeze() == 1.0).all()