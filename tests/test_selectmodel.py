from sklearn.metrics import mean_squared_error

from fold.composites.select import SelectBest
from fold.loop import backtest, train
from fold.models.dummy import DummyRegressor
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_monotonous_data


def test_selectbest() -> None:
    X, y = generate_monotonous_data(1000)

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    transformations = [
        SelectBest(
            models=[
                [
                    lambda x: x,
                    DummyRegressor(
                        predicted_value=0.1,
                    ),
                ],
                [
                    lambda x: x,
                    DummyRegressor(
                        predicted_value=0.2,
                    ),
                ],
            ],
            scorer=mean_squared_error,
            is_scorer_loss=True,
        ),
    ]

    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    assert (pred.squeeze() == 0.2).all()
