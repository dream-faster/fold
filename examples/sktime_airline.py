import numpy as np
import pandas as pd
from fold.loop import backtest, train
from fold.splitters import ExpandingWindowSplitter, SingleWindowSplitter
from krisi import score
from sktime.datasets import load_airline
from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.naive import NaiveForecaster

from fold_models.sktime import WrapSktime

y = load_airline()
splitter = ExpandingWindowSplitter(initial_train_window=50, step=1)
X = pd.DataFrame(np.zeros(len(y)))

model = WrapSktime.from_model(AutoARIMA(), use_exogenous=False)


transformations_over_time = train(model, X, y, splitter)
pred = backtest(transformations_over_time, X, y, splitter)

score(y, pred)
