from fold.loop import train_backtest
from fold.models.ar import AR
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_monotonous_data

# from fold_wrappers import WrapStatsForecast, WrapStatsModels
# from statsmodels.tsa.ar_model import AutoReg as StatsModelAR


def test_ar_1_equivalent() -> None:
    _, y = generate_monotonous_data(length=70, freq="s")
    splitter = ExpandingWindowSplitter(initial_train_window=40, step=1)

    model = AR(1)
    pred_own_ar, _ = train_backtest(model, None, y, splitter)

    # model = WrapStatsModels(
    #     StatsModelAR,
    #     init_args={
    #         "lags": [1],
    #         "trend": "n",
    #     },
    #     online_mode=True,
    # )
    # pred_statsforecast_ar, _ = train_backtest(model, None, y, splitter)
    # assert np.isclose(
    #     pred_statsforecast_ar.squeeze(), pred_own_ar.squeeze(), atol=0.02
    # ).all()


def test_ar_2_equivalent() -> None:
    _, y = generate_monotonous_data(length=70, freq="s")
    splitter = ExpandingWindowSplitter(initial_train_window=40, step=1)

    model = AR(2)
    pred_own_ar, _ = train_backtest(model, None, y, splitter)

    # model = WrapStatsModels(
    #     StatsModelAR,
    #     init_args={
    #         "lags": [2],
    #         "trend": "n",
    #     },
    #     online_mode=True,
    # )
    # pred_statsforecast_ar, _ = train_backtest(model, None, y, splitter)
    # assert np.isclose(
    #     pred_statsforecast_ar.squeeze(), pred_own_ar.squeeze(), atol=0.02
    # ).all()


def test_ar_speed() -> None:
    _, y = generate_monotonous_data(length=7000, freq="s")

    model = AR(2)
    splitter = ExpandingWindowSplitter(initial_train_window=0.1, step=0.1)
    train_backtest(model, None, y, splitter)
