import numpy as np

from fold.loop import backtest, train
from fold.splitters import Splitter
from fold.utils.tests import generate_monotonous_data


def run_pipeline_and_check_if_results_close_univariate(
    model, splitter: Splitter, data_length: int = 70
):
    X, y = generate_monotonous_data(length=data_length)

    trained_pipeline = train(model, None, y, splitter)
    pred = backtest(trained_pipeline, None, y, splitter)
    assert np.isclose(y.squeeze()[pred.index], pred.squeeze().values, atol=0.1).all()


def run_pipeline_and_check_if_results_close_exogenous(
    model, splitter: Splitter, data_length: int = 70
):
    X, y = generate_monotonous_data(length=data_length)

    trained_pipeline = train(model, X, y, splitter)
    pred = backtest(trained_pipeline, X, y, splitter)
    assert np.isclose(y.squeeze()[pred.index], pred.squeeze().values, atol=0.1).all()
