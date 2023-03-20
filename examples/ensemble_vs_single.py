import pandas as pd
from fold_models.sktime import WrapSktime
from krisi import score
from sktime.datasets import load_airline
from sktime.forecasting.arima import AutoARIMA

from fold.loop import backtest, train
from fold.splitters import ExpandingWindowSplitter

X, y = load_airline(return_X_y=True)
y = pd.Series(y)
X = X.reset_index(drop=True)

# Let's build a simple, one model pipeline with Sktime's AutoARIMA module
pipeline_arima = WrapSktime(AutoARIMA, None, use_exogenous=True)

splitter = ExpandingWindowSplitter(initial_train_window=0.15, step=0.1)

transformations_over_time = train(pipeline_arima, X, y, splitter)
pred = backtest(transformations_over_time, X, y, splitter)

score(y[pred.index], pred)
