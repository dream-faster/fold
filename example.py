from src.drift.loop import infer, train
from src.drift.models import Baseline, BaselineStrategy
from src.drift.splitters import ExpandingWindowSplitter
from src.drift.transformations import NoTransformation
from tests.utils import generate_sine_wave_data


def test_baseline_naive_model() -> None:

    y = generate_sine_wave_data()
    X = y.shift(1)

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [NoTransformation(), Baseline(strategy=BaselineStrategy.naive)]

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = infer(transformations_over_time, X, splitter)
    assert (y[pred.index].shift(1) == pred).sum() == len(pred) - 1


test_baseline_naive_model()
