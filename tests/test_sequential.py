from drift.loop import backtest, train
from drift.splitters import ExpandingWindowSplitter
from drift.transformations.columns import OnlyPredictions
from drift.utils.tests import generate_all_zeros

from drift.transformations.sequential import NeuralForecastWrapper
import pandas as pd

def test_sequential() -> None:
    from neuralforecast.models import NBEATS

    X = generate_all_zeros(1000)
    y = X.squeeze()
    
    X.reset_index(inplace=True)
    X.rename({"index":"ds"})
    X.rename(columns={ X.columns[0]: "ds", X.columns[1]:'y' }, inplace = True)

    step = 400
    splitter = ExpandingWindowSplitter(train_window_size=400, step=step)
    horizon = 2
    transformations = [
        NeuralForecastWrapper(NBEATS(input_size=step, h=horizon, max_epochs=50)),
        OnlyPredictions(),
    ]
    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = backtest(transformations_over_time, X, y, splitter)
    assert (pred.squeeze() == y[pred.index]).all()

test_sequential()
