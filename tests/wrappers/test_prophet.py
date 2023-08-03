import numpy as np

from fold.loop import TrainMethod, train_backtest
from fold.models.wrappers.prophet import WrapProphet
from fold.splitters import ExpandingWindowSplitter, SingleWindowSplitter
from fold.utils.tests import generate_monotonous_data, generate_sine_wave_data


def test_prophet() -> None:
    from prophet import Prophet

    X, y = generate_sine_wave_data(cycles=100, length=2400, freq="H")

    splitter = ExpandingWindowSplitter(initial_train_window=0.5, step=0.1)
    transformations = WrapProphet.from_model(Prophet())

    pred, _ = train_backtest(transformations, X, y, splitter)
    assert np.isclose(y.squeeze()[pred.index], pred.squeeze(), atol=0.1).all()


def test_automatic_wrapping_prophet() -> None:
    from prophet import Prophet

    _, y = generate_monotonous_data()
    train_backtest(
        Prophet(),
        None,
        y,
        splitter=SingleWindowSplitter(0.5),
    )


def test_prophet_updates() -> None:
    from prophet import Prophet

    X, y = generate_sine_wave_data(cycles=100, length=2400, freq="H")

    splitter = ExpandingWindowSplitter(initial_train_window=0.8, step=0.1)
    transformations = WrapProphet.from_model(Prophet())

    train_backtest(transformations, X, y, splitter, train_method=TrainMethod.sequential)
    # assert np.isclose(y.squeeze()[pred.index], pred.squeeze(), atol=0.3).all()
    # TODO: this is a flaky test, but don't yet know how to make Prophet more stable with updating
