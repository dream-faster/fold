import numpy as np
import pytest

from fold.loop.encase import train_backtest
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.difference import Difference
from fold.utils.tests import generate_sine_wave_data


# @pytest.mark.parametrize("lag", [1, 2, 3, 5, 10, 24])
def test_difference(lag: int):
    X, y = generate_sine_wave_data(length=600)
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=100)
    pred, trained_pipelines = train_backtest(Difference(), X, y, splitter)
    assert np.isclose(
        X.squeeze()[pred.index],
        trained_pipelines[0].iloc[0].inverse_transform(pred).squeeze(),
        atol=1e-3,
    ).all()


@pytest.mark.parametrize("lag", [1, 2, 3, 5, 10, 24])
def test_difference_inverse(lag: int):
    X, y = generate_sine_wave_data(length=600)
    difference = Difference(lag)
    difference.fit(X, y)
    diffed = difference.transform(X, in_sample=True).squeeze()
    inverse_diffed = difference.inverse_transform(diffed)
    assert np.isclose(X.squeeze(), inverse_diffed, atol=1e-3).all()

    diffed = difference.transform(X, in_sample=False).squeeze()
    inverse_diffed = difference.inverse_transform(diffed)
    assert np.isclose(X.squeeze(), inverse_diffed, atol=1e-3).all()


# test difference with updates!
