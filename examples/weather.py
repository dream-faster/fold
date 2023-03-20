#%%
import pandas as pd
from fold_models import WrapXGB
from krisi import score
from xgboost import XGBRegressor

from fold.loop import backtest, train
from fold.models.baseline import BaselineNaive
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.lags import AddLagsX

city_name = "Los Angeles"
humidity = pd.read_csv(
    "examples/data/weather/humidity.csv", delimiter=",", index_col=0, parse_dates=True
)[city_name].rename("humidity")
pressure = pd.read_csv(
    "examples/data/weather/pressure.csv", delimiter=",", index_col=0, parse_dates=True
)[city_name].rename("pressure")
wind_speed = pd.read_csv(
    "examples/data/weather/wind_speed.csv", delimiter=",", index_col=0, parse_dates=True
)[city_name].rename("wind_speed")
wind_direction = pd.read_csv(
    "examples/data/weather/wind_direction.csv",
    delimiter=",",
    index_col=0,
    parse_dates=True,
)[city_name].rename("wind_direction")
temperature = pd.read_csv(
    "examples/data/weather/temperature.csv",
    delimiter=",",
    index_col=0,
    parse_dates=True,
)[city_name].rename("temperature")
df = (
    (
        pd.concat([humidity, pressure, wind_speed, wind_direction, temperature], axis=1)
        .iloc[1:]
        .ffill()
    )
    .resample("1D")
    .last()
)
df.iloc[:1000].to_csv("tests/data/weather.csv")

# %%

y = df["temperature"]

splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
pipeline = [BaselineNaive()]
transformations_over_time = train(pipeline, None, y, splitter)
pred = backtest(transformations_over_time, None, y, splitter)

score(y[pred.index], pred.squeeze()).print_summary(extended=False)

# %%

# plot_acf(df.temperature, lags=500)

# %%
# pipeline = [BaselineNaiveSeasonal(seasonal_length=365)]
# transformations_over_time = train(pipeline, None, y, splitter)
# pred = backtest(transformations_over_time, None, y, splitter)
# score(y[pred.index], pred.squeeze()).print_summary(extended=False)
# RMSE: 2.6


# %%


# pipeline = [
#     WrapStatsModels(
#         model_class=ARIMA,
#         init_args=dict(order=(1, 0, 0)),
#         use_exogenous=False,
#         online_mode=False,
#     )
# ]
# transformations_over_time = train(pipeline, None, y, splitter)
# pred = backtest(transformations_over_time, None, y, splitter)
# score(y[pred.index], pred.squeeze()).print_summary(extended=False)
# RMSE: 6.0

# %%


# y = df["temperature"]

# pipeline = [
#     WrapStatsModels(
#         model_class=ARIMA,
#         init_args=dict(order=(1, 0, 0)),
#         use_exogenous=False,
#         online_mode=True,
#     )
# ]
# transformations_over_time = train(pipeline, None, y, splitter)
# pred = backtest(transformations_over_time, None, y, splitter)
# # how was this calculated??
# score(y[pred.index], pred.squeeze()).print_summary(extended=False)

# RMSE: 2.6


# %%


X = df.shift(1)[:-1]  # current timestamp - which is just the last value
y = df["temperature"][:-1]  # next timestamp

# step_size = 4500 timestamp
# sequence model = expanding window means it'll get all
# the data and it'll predict the next 4500 timestamps

splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
pipeline = [
    AddLagsX(columns_and_lags=[("temperature", list(range(1, 5)))]),
    WrapXGB.from_model(XGBRegressor()),
]
transformations_over_time = train(pipeline, X, y, splitter)
pred = backtest(
    transformations_over_time, X, y, splitter
)  # simulated inference over past data
score(y[pred.index], pred.squeeze()).print_summary(extended=False)
# ?


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
