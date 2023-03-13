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

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    transformations = [
        BaselineNaiveSeasonal(seasonal_length=10),
        Test(fit_func=check_if_not_nan, transform_func=lambda X: X),
        OnlyPredictions(),
    ]
    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert (pred.squeeze() == y.shift(10)[pred.index]).all()
    assert (
        len(pred) == 600
    )  # should return non-NaN predictions for the all out-of-sample sets
