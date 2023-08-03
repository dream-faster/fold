import numpy as np

from fold.loop import train_backtest
from fold.models.wrappers.arch import WrapArch
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_sine_wave_data


def test_arch_univariate() -> None:
    _, y = generate_sine_wave_data(length=200)
    y = np.log(y + 2.0).diff().dropna() * 100
    model = WrapArch(init_args=dict(vol="Garch", p=1, o=1, q=1, dist="Normal"))
    splitter = ExpandingWindowSplitter(initial_train_window=0.5, step=0.1)
    pred, _ = train_backtest(model, None, y, splitter)
    # assert np.isclose(y[pred.index] ** 2, pred.squeeze().values, atol=0.1).all()
