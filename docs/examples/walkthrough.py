from fold.utils.dataset import get_preprocessed_dataset

X, y = get_preprocessed_dataset(
    "energy/industrial_pv_load",
    target_col="residual_load",
    resample="H",
    deduplication_strategy="first",
    shorten=10000,
)
y.plot(figsize=(20, 5), grid=True)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
pipeline = [model]

from fold import ExpandingWindowSplitter

splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
from fold import train_evaluate

scorecard, predictions, trained_pipeline_over_time = train_evaluate(
    pipeline, X, y, splitter, krisi_args={"model_name": "pipeline_sklearn_base"}
)
scorecard.print_summary(extended=False)
from krisi.report import plot_y_predictions

plot_y_predictions(y, predictions)
results = [(scorecard, predictions)]
from sklearn.linear_model import LinearRegression

from fold.transformations import AddLagsX

pipeline_1 = [
    AddLagsX(("all", list(range(1, 15)))),
    lambda x: x.fillna(0.0),
    LinearRegression(),
]
from fold_models.xgboost import WrapXGB
from xgboost import XGBRegressor

model = WrapXGB.from_model(XGBRegressor())
pipeline_2 = [AddLagsX(("all", list(range(1, 15)))), lambda x: x.fillna(0.0), model]
from fold.composites import Ensemble

ensemble_pipeline = Ensemble([pipeline_1, pipeline_2])
from fold import train_evaluate

for name, pipeline in [
    ("pipeline_sklearn_lags", pipeline_1),
    ("pipeline_xgboost_lags", pipeline_2),
    ("ensemble_pipeline", ensemble_pipeline),
]:
    scorecard, predictions, pipeline_trained = train_evaluate(
        pipeline, X, y, splitter, krisi_args={"model_name": name}
    )
    results.append((scorecard, predictions))
from fold.transformations import AddHolidayFeatures

ensemble_pipeline_with_holiday = [AddHolidayFeatures(["DE"]), ensemble_pipeline]
from fold import train_evaluate

scorecard, predictions, pipeline_trained = train_evaluate(
    ensemble_pipeline_with_holiday,
    X,
    y,
    splitter,
    krisi_args={"model_name": "ensemble_pipeline_with_holiday"},
)
results.append((scorecard, predictions))
from krisi import compare

compare([scorecard for scorecard, predictions in results])
from krisi.report import plot_y_predictions

all_predictions = [
    predictions.squeeze().rename(scorecard.metadata.model_name)
    for scorecard, predictions in results
]
plot_y_predictions(y, all_predictions, value_name="residual_load")
