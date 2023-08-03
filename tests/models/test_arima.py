from fold.loop import train_backtest
from fold.models.arima import ARIMA
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_sine_wave_data

# from fold_wrappers import WrapStatsForecast, WrapStatsModels


# from statsforecast.models import ARIMA as StatsForecastARIMA
# from statsmodels.tsa.arima.model import ARIMA as StatsModelARIMA


def test_arima_equivalent() -> None:
    _, y = generate_sine_wave_data(length=70, freq="s")
    splitter = ExpandingWindowSplitter(initial_train_window=50, step=1)

    model = ARIMA(1, 1, 0)
    pred_own_ar, _ = train_backtest(model, None, y, splitter)

    # model = WrapStatsModels(
    #     StatsModelARIMA, init_args={"order": (1, 1, 0), "trend": "n"}, online_mode=False
    # )
    # model = WrapStatsForecast.from_model(
    #     StatsForecastARIMA((1, 1, 0)), online_mode=False
    # )

    # pred_statsforecast_ar, _ = train_backtest(model, None, y, splitter)
    # assert np.isclose(
    #     pred_statsforecast_ar.squeeze(), pred_own_ar.squeeze(), atol=0.05
    # ).all()


def test_arima_speed() -> None:
    _, y = generate_sine_wave_data(length=700, freq="s")

    model = ARIMA(2, 1, 0)
    splitter = ExpandingWindowSplitter(initial_train_window=0.1, step=0.1)
    train_backtest(model, None, y, splitter)
