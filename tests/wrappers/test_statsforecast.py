import numpy as np

from fold.loop import train_backtest
from fold.models.wrappers.statsforecast import WrapStatsForecast
from fold.splitters import ExpandingWindowSplitter, SingleWindowSplitter
from fold.utils.test_utils import (
    run_pipeline_and_check_if_results_close_exogenous,
    run_pipeline_and_check_if_results_close_univariate,
)
from fold.utils.tests import generate_monotonous_data


def test_statsforecast_univariate_naive() -> None:
    from statsforecast.models import Naive

    X, y = generate_monotonous_data(length=70)

    splitter = ExpandingWindowSplitter(initial_train_window=50, step=1)
    pipeline = WrapStatsForecast(model_class=Naive, init_args={})
    pred, _ = train_backtest(pipeline, None, y, splitter)
    assert np.isclose(
        y.squeeze().shift(1)[pred.index][:-1], pred.squeeze()[:-1], atol=0.01
    ).all()


def test_statsforecast_univariate_autoarima() -> None:
    from statsforecast.models import AutoARIMA

    run_pipeline_and_check_if_results_close_univariate(
        model=WrapStatsForecast.from_model(AutoARIMA()),
        splitter=ExpandingWindowSplitter(initial_train_window=50, step=5),
    )


def test_statsforecast_exogenous_autoarima() -> None:
    from statsforecast.models import AutoARIMA

    run_pipeline_and_check_if_results_close_exogenous(
        model=[WrapStatsForecast.from_model(AutoARIMA())],
        splitter=ExpandingWindowSplitter(initial_train_window=50, step=2),
    )


def test_statsforecast_univariate_arima() -> None:
    from statsforecast.models import ARIMA

    run_pipeline_and_check_if_results_close_univariate(
        model=WrapStatsForecast(model_class=ARIMA, init_args={"order": (1, 0, 0)}),
        splitter=ExpandingWindowSplitter(initial_train_window=50, step=5),
    )


def test_automatic_wrapping_statsforecast() -> None:
    from statsforecast.models import ARIMA

    _, y = generate_monotonous_data()
    train_backtest(
        ARIMA(order=(1, 1, 0)),
        None,
        y,
        splitter=SingleWindowSplitter(0.5),
    )


def test_statsforecast_univariate_mstl() -> None:
    from statsforecast.models import MSTL

    run_pipeline_and_check_if_results_close_univariate(
        model=WrapStatsForecast.from_model(MSTL(season_length=10), online_mode=False),
        splitter=ExpandingWindowSplitter(initial_train_window=50, step=10),
    )
