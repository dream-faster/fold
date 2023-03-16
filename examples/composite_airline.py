import numpy as np
import pandas as pd
from fold_models.sktime import WrapSktime
from krisi import score
from sktime.datasets import load_airline
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.naive import NaiveForecaster

from fold.loop import backtest, train
from fold.models.ensemble import Ensemble
from fold.splitters import ExpandingWindowSplitter

y = load_airline()


splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=1)
X = pd.DataFrame(np.zeros(len(y)), index=y.index)

transformations = [
    Ensemble(
        [
            WrapSktime.from_model(AutoARIMA(), use_exogenous=False),
            WrapSktime.from_model(
                NaiveForecaster(), use_exogenous=False, online_mode=True
            ),
        ]
    ),
]

transformations_over_time = train(transformations, X, y, splitter)
pred = backtest(transformations_over_time, X, y, splitter)

score(y[pred.index], pred.iloc[:, 0]).print_summary()
