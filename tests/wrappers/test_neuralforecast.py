import numpy as np
import pandas as pd

from fold.loop import train_backtest
from fold.models.wrappers.neuralforecast import WrapNeuralForecast
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_monotonous_data, generate_sine_wave_data


def test_neuralforecast_nbeats() -> None:
    from neuralforecast.models import NBEATS

    X, y = generate_sine_wave_data(cycles=50)

    step = 100
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=step)

    transformations = WrapNeuralForecast.from_model(
        NBEATS(
            input_size=step,
            h=step,
            max_steps=50,
        ),
    )
    pred, _ = train_backtest(transformations, X, y, splitter)
    assert np.isclose(y.squeeze()[pred.index], pred.squeeze(), atol=0.1).all()


def test_neuralforecast_nhits() -> None:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS

    X, y = generate_monotonous_data(length=500)

    step = 100
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=step)
    model = NHITS(
        input_size=step,
        h=step,
        max_steps=50,
    )
    transformations = WrapNeuralForecast.from_model(model)
    pred, _ = train_backtest(transformations, X, y, splitter)

    data = pd.DataFrame(
        {"ds": X[:400].index, "y": y[:400].values, "unique_id": 1.0},
    )
    nf = NeuralForecast(models=[model], freq="m")
    nf.fit(data)
    nf_pred = nf.predict()

    assert np.isclose(nf_pred["NHITS"].values, pred.squeeze().values, atol=0.01).all()
    assert np.isclose(y.squeeze()[pred.index], pred.squeeze(), atol=0.01).all()
