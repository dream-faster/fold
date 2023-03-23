#%%
import pandas as pd
from fold_models import WrapStatsModels, WrapXGB
from krisi import score
from sklearn.feature_selection import SelectKBest, f_regression
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

from fold.composites.target import TransformTarget
from fold.loop import backtest, train
from fold.models import Naive, NaiveSeasonal
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.difference import Difference
from fold.transformations.lags import AddLagsX
from fold.transformations.window import AddWindowFeatures

df = (
    pd.read_csv(
        "https://raw.githubusercontent.com/dream-faster/datasets/main/datasets/weather/historical_hourly_la.csv",
        index_col=0,
        parse_dates=True,
    )
    # .resample("1D")
    # .last()
)

# %%
plot_acf(df.temperature, lags=50)
# We see two things:
# 1. There is a strong daily seasonality
# 2. There

# %%
# Let's start by training the simplest possible model: one that repeats the last value.

y = df["temperature"].shift(-1)[:-1]
X = df[:-1]

splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
scorecards = []

transformations_over_time = train(Naive(), None, y, splitter)
pred = backtest(transformations_over_time, None, y, splitter)

sc = score(y[pred.index], pred.squeeze())
sc.print_summary(extended=False)
scorecards.append(sc)

# %%
transformations_over_time = train(NaiveSeasonal(seasonal_length=24), None, y, splitter)
pred = backtest(transformations_over_time, None, y, splitter)

sc = score(y[pred.index], pred.squeeze())
sc.print_summary(extended=False)
scorecards.append(sc)


# %% A simple ARIMA model without differencing

pipeline = [
    WrapStatsModels(
        model_class=ARIMA,
        init_args=dict(order=(1, 0, 0)),
        use_exogenous=False,
        online_mode=True,
    )
]
transformations_over_time = train(pipeline, None, y, splitter)
pred = backtest(transformations_over_time, None, y, splitter)
sc = score(y[pred.index], pred.squeeze())
sc.print_summary(extended=False)
scorecards.append(sc)


# %% A simple ARIMA model with our own differencing

# pipeline = [
#     TransformTarget(
#         [
#             WrapStatsModels(
#                 model_class=ARIMA,
#                 init_args=dict(order=(1, 0, 0)),
#                 use_exogenous=True,
#                 online_mode=True,
#             )
#         ],
#         Difference(),
#     )
# ]
# transformations_over_time = train(pipeline, X, y, splitter)
# pred = backtest(transformations_over_time, X, y, splitter)
# sc = score(y[pred.index], pred.squeeze())
# sc.print_summary(extended=False)
# scorecards.append(sc)


# %% XGB with lags
from fold.composites.ensemble import Ensemble

splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
tabular_pipeline = [
    AddWindowFeatures([("temperature", 14, "mean")]),
    AddLagsX(columns_and_lags=[("temperature", list(range(1, 5)))]),
    WrapXGB.from_model(XGBRegressor(max_depth=20)),
    # PerColumnTransform(lambda X: X.rolling(14).mean()),
    # AddWindowFeature(window = 14, [("temperature", lambda X: X.mean()]),
    # lambda X: X["temperature"].rolling(14).mean(),
]

arima_pipeline = [
    WrapStatsModels(
        model_class=ARIMA,
        init_args=dict(order=(1, 0, 0)),
        use_exogenous=False,
        online_mode=True,
    )
]
ensemble = Ensemble([tabular_pipeline, arima_pipeline])

transformations_over_time = train(tabular_pipeline, X, y, splitter)
pred = backtest(transformations_over_time, X, y, splitter)
sc = score(y[pred.index], pred.squeeze())
sc.print_summary(extended=False)
scorecards.append(sc)


# %% XGB with lags and extra features
splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
pipeline = TransformTarget(
    [
        Difference(),
        SelectKBest(k=10, score_func=f_regression),
        AddLagsX(columns_and_lags=[("temperature", list(range(1, 5)))]),
        WrapXGB.from_model(XGBRegressor()),
    ],
    Difference(),
)

transformations_over_time = train(pipeline, X, y, splitter)
pred = backtest(transformations_over_time, X, y, splitter)
sc = score(y[pred.index], pred.squeeze())
sc.print_summary(extended=False)
scorecards.append(sc)


# %%
# continue with exogenous data
