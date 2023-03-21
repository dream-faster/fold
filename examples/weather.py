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
from fold.models.baseline import BaselineNaive, BaselineNaiveSeasonal
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.difference import Difference
from fold.transformations.lags import AddLagsX

df = (
    pd.read_csv(
        "https://raw.githubusercontent.com/dream-faster/datasets/main/datasets/weather/historical_hourly_la.csv",
        index_col=0,
        parse_dates=True,
    )
    .resample("1D")
    .last()
)

# %%
plot_acf(df.temperature, lags=500)

# %%

y = df["temperature"].shift(-1)[:-1]
X = df[:-1]

splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
scorecards = []

pipeline = [BaselineNaive()]
transformations_over_time = train(pipeline, None, y, splitter)
pred = backtest(transformations_over_time, None, y, splitter)

sc = score(y[pred.index], pred.squeeze())
sc.print_summary(extended=False)
scorecards.append(sc)


# %%
pipeline = [BaselineNaiveSeasonal(seasonal_length=365)]
transformations_over_time = train(pipeline, None, y, splitter)
pred = backtest(transformations_over_time, None, y, splitter)

sc = score(y[pred.index], pred.squeeze())
sc.print_summary(extended=False)
sc.save()
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

transformations_over_time = train(ensemble, X, y, splitter)
pred = backtest(transformations_over_time, X, y, splitter)
sc = score(y[pred.index], pred.squeeze())
sc.print_summary(extended=False)
scorecards.append(sc)


#%%
from fold.loop import infer, train_for_deployment, update

deployable_pipeline = train_for_deployment(ensemble, X, y, splitter)
infer(deployable_pipeline, X)
update(deployable_pipeline, X, y)


# %% XGB with lags and extra features
for col in X.columns:
    X[f"{col}_14_mean"] = X[col].rolling(14).mean()

X = X.dropna()
y = y.loc[X.index]

X["month"] = X.index.month
X["day"] = X.index.day

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

# splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
# pipeline1 = TransformTarget([WrapXGB.from_model(XGBRegressor())], Difference())
# pipeline2 = WrapXGB.from_model(XGBRegressor())
# ensemble = Ensemble([pipeline1, pipeline2])
# transformations_over_time = train(ensemble, None, y, splitter)
# pred = backtest(transformations_over_time, None, y, splitter)
# score(y[pred.index], pred.squeeze()).print_summary(extended=False)

# %%
# continue with exogenous data
