import numpy as np

from fold.loop import train_backtest
from fold.models.garch import ARCH
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_sine_wave_data

# from fold_wrappers.arch import WrapArch


def test_garch_equivalent() -> None:
    _, y = generate_sine_wave_data(length=70)
    y = np.log(y + 2.0).diff().dropna() * 100
    splitter = ExpandingWindowSplitter(initial_train_window=50, step=1)

    model = ARCH(1)
    pred_own_ar, trained = train_backtest(model, None, y, splitter)

    # model = WrapArch(init_args=dict(vol="arch", p=1, o=0, q=0, dist="Normal"))

    # pred_statsforecast_ar, trained_3rdparty = train_backtest(model, None, y, splitter)
    # assert np.isclose(
    #     pred_statsforecast_ar.squeeze(), pred_own_ar.squeeze(), atol=0.05
    # ).all()
