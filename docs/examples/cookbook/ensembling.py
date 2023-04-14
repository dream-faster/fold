"""
Ensembling (Composite Models)
===========================
"""

from fold_models import WrapStatsForecast, WrapXGB
from statsforecast.models import ARIMA
from xgboost import XGBRegressor

from fold.composites.ensemble import Ensemble
from fold.loop import train_evaluate
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.difference import Difference
from fold.transformations.lags import AddLagsX
from fold.transformations.window import AddWindowFeatures
from fold.utils.dataset import get_preprocessed_dataset

X, y = get_preprocessed_dataset(
    "weather/historical_hourly_la", target_col="temperature", shorten=1000
)

splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
pipeline_tabular = [
    Difference(),
    AddWindowFeatures([("temperature", 14, "mean")]),
    AddLagsX(columns_and_lags=[("temperature", list(range(1, 5)))]),
    WrapXGB.from_model(XGBRegressor()),
]
pipeline_arima = WrapStatsForecast(ARIMA, {"order": (1, 1, 0)}, use_exogenous=False)
ensemble = Ensemble([pipeline_tabular, pipeline_arima])

scorecard, prediction, trained_pipelines = train_evaluate(ensemble, X, y, splitter)
