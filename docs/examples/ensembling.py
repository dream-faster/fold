"""
Ensembling (Composite Models)
===========================
"""

from fold_wrappers import WrapStatsForecast
from statsforecast.models import ARIMA
from xgboost import XGBRegressor

from fold.composites import Concat, Ensemble
from fold.loop import train_evaluate
from fold.splitters import ExpandingWindowSplitter
from fold.transformations import AddLagsX, AddWindowFeatures, Difference
from fold.utils.dataset import get_preprocessed_dataset

X, y = get_preprocessed_dataset(
    "weather/historical_hourly_la", target_col="temperature", shorten=1000
)

splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.1)
pipeline_tabular = [
    Difference(),
    Concat(
        [
            AddWindowFeatures([("temperature", 14, "mean")]),
            AddLagsX(columns_and_lags=[("temperature", list(range(1, 5)))]),
        ]
    ),
    XGBRegressor(),
]
pipeline_arima = WrapStatsForecast(ARIMA, {"order": (1, 1, 0)}, use_exogenous=False)
ensemble = Ensemble([pipeline_tabular, pipeline_arima])

scorecard, prediction, trained_pipelines = train_evaluate(ensemble, X, y, splitter)
