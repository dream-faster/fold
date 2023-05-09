from sklearn.metrics import mean_squared_error

from fold.composites.optimize import OptimizeGridSearch
from fold.loop import backtest, train
from fold.models.dummy import DummyClassifier, DummyRegressor
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_monotonous_data


def test_grid_hpo() -> None:
    X, y = generate_monotonous_data(1000)

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pipeline = [
        OptimizeGridSearch(
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
            ],
            scorer=mean_squared_error,
        )
    ]

    trained_pipelines = train(pipeline, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)

    assert pred is not None
    assert len(pred) == 600
