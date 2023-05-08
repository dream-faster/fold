from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error

from fold.composites.optimize import OptimizeGridSearch
from fold.loop import train_backtest
from fold.models.sklearn import WrapSKLearnRegressor
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_monotonous_data


def test_gridsearch() -> None:
    X, y = generate_monotonous_data(1000)

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pipeline = [
        OptimizeGridSearch(
            pipeline=WrapSKLearnRegressor.from_model(
                DummyRegressor(strategy="constant", constant=1)
            ),
            scorer=mean_squared_error,
            is_scorer_loss=True,
        )
    ]

    pred, _ = train_backtest(pipeline, X, y, splitter)
    assert (pred.squeeze() == 1).all()
