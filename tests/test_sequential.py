from drift.loop import backtest, train
from drift.splitters import ExpandingWindowSplitter
from drift.transformations.columns import OnlyPredictions
from drift.utils.tests import generate_all_zeros

from drift.transformations.sequential import SequentialTransformation
import pandas as pd

def test_sequential() -> None:
    from neuralforecast.models import NBEATS

    X = generate_all_zeros(1000)
    y = X.squeeze()
    
    X.reset_index(inplace=True)
    X.rename({"index":"ds"})
    X.rename(columns={ X.columns[0]: "ds", X.columns[1]:'y' }, inplace = True)

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    horizon = 2
    transformations = [
        SequentialTransformation(NBEATS(input_size=2 * horizon, h=horizon, max_epochs=50)),
        OnlyPredictions(),
    ]
    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = backtest(transformations_over_time, X, y, splitter)
    assert (pred.squeeze() == y[pred.index]).all()

test_sequential()
