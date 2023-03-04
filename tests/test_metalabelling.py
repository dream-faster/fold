from fold.loop import backtest, train
from fold.models.dummy import DummyClassifier
from fold.models.metalabeling import MetaLabeling
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.columns import OnlyPredictions
from fold.utils.tests import generate_all_zeros


def test_metalabeling() -> None:

    X = generate_all_zeros(1000)
    y = X.squeeze()

    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)
    transformations = [
        MetaLabeling(
            primary=[
                lambda x: x,
                DummyClassifier(
                    predicted_value=1,
                    all_classes=[1, 0],
                    predicted_probabilities=[1.0, 0.0],
                ),
            ],
            meta=[
                lambda x: x,
                DummyClassifier(
                    predicted_value=0.5,
                    all_classes=[1, 0],
                    predicted_probabilities=[0.5, 0.5],
                ),
            ],
            positive_class=1,
        ),
        OnlyPredictions(),
    ]

    transformations_over_time = train(transformations, X, y, splitter)
    _, pred = backtest(transformations_over_time, X, y, splitter)
    assert (pred.squeeze() == 0.5).all()
