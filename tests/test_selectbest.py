import numpy as np

from fold.loop import backtest, train
from fold.models import SelectBest
from fold.models.dummy import DummyClassifier
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_sine_wave_data


def test_ensemble_regression() -> None:

    X = generate_sine_wave_data()
    y = X.shift(-1).squeeze()

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [
        SelectBest(
            [
                [lambda x: (x - 1.0).rename({"sine": "predictions_1"}, axis=1)],
                [lambda x: (x - 2.0).rename({"sine": "predictions_2"}, axis=1)],
                [lambda x: (x - 3.0).rename({"sine": "predictions_3"}, axis=1)],
            ],
            scorer=
        ),
    ]

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = backtest(transformations_over_time, X, y, splitter)
    assert (np.isclose((X.squeeze()[pred.index]), (pred.squeeze() + 2.0))).all()
