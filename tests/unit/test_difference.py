import numpy as np
import pytest

from fold.loop.encase import train_backtest
from fold.splitters import SingleWindowSplitter
from fold.transformations.difference import Difference
from fold.utils.tests import generate_sine_wave_data


@pytest.mark.parametrize("lag", [1, 2, 3, 5, 10, 24])
def test_difference(lag: int):
    X, y = generate_sine_wave_data(length=100)
    splitter = SingleWindowSplitter(train_window=50)
    pred, trained_pipelines = train_backtest(Difference(), X, y, splitter)
    assert np.isclose(
        X.squeeze()[pred.index],
        trained_pipelines[0].iloc[0].inverse_transform(pred, in_sample=True).squeeze(),
        atol=1e-3,
    ).all()


@pytest.mark.parametrize("lag", [1, 2, 3, 5, 10, 24])
def test_difference_inverse(lag: int):
    X, y = generate_sine_wave_data(length=100)
    X_train, X_test, y_train = (
        X.iloc[:52],
        X.iloc[52:],
        y.iloc[:52],
    )
    difference = Difference(lag)
    difference.fit(X_train, y_train)
    diffed = difference.transform(X_train, in_sample=True).squeeze()
    inverse_diffed = difference.inverse_transform(diffed, in_sample=True)
    assert np.isclose(X_train.squeeze(), inverse_diffed, atol=1e-3).all()

    difference.update(X_train, y_train)
    assert difference.last_values_X.equals(X_train.iloc[-lag:None])
    diffed = difference.transform(X_test, in_sample=False).squeeze()
    inverse_diffed = difference.inverse_transform(diffed, in_sample=False)
    assert np.isclose(X_test.squeeze(), inverse_diffed, atol=1e-3).all()
