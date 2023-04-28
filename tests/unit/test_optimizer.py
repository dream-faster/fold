import pytest
from sklearn.metrics import mean_squared_error

from fold.composites.optimize import SelectGridSearch
from fold.loop import train_backtest
from fold.models.dummy import DummyRegressor
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_monotonous_data


@pytest.mark.parametrize("cls", [SelectGridSearch])
def test_selectbest(cls) -> None:
    X, y = generate_monotonous_data(1000)

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    pipeline = cls(
        pipelines=[
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
    )

    pred, _ = train_backtest(pipeline, X, y, splitter)
    assert (pred.squeeze() == 0.2).all()
