from fold.loop import backtest, train
from fold.models.random import RandomClassifier
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.columns import OnlyPredictions
from fold.utils.tests import generate_all_zeros


def test_random_classifier() -> None:
    X, y = generate_all_zeros(1000)

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    transformations = [RandomClassifier(all_classes=[0, 1]), OnlyPredictions()]
    transformations_over_time = train(transformations, X, y, splitter)
    pred = backtest(transformations_over_time, X, y, splitter)
    assert pred.squeeze().sum() > 1
