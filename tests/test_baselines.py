from fold.loop import backtest, train
from fold.models.baseline import BaselineNaiveSeasonal
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.columns import OnlyPredictions
from fold.transformations.test import Test
from fold.utils.tests import generate_sine_wave_data


def test_baseline_naive_seasonal() -> None:
    X, y = generate_sine_wave_data(1000)

    def check_if_not_nan(x):
        assert not x.isna().squeeze().any()

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [
        BaselineNaiveSeasonal(seasonal_length=10),
        Test(fit_func=check_if_not_nan, transform_func=lambda X: X),
        OnlyPredictions(),
    ]
    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert (
        len(pred) == 590
    )  # loop should trim the predictions by 10, as BaselineNaiveSeasonal's predict_in_sample() will return `seasonal_length` NaNs at the beginning of the predictions
    assert (pred.squeeze() == y.shift(10)[pred.index]).all()
