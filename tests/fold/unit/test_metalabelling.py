from fold.composites.metalabeling import MetaLabeling
from fold.loop import backtest, train
from fold.models.dummy import DummyClassifier
from fold.splitters import ExpandingWindowSplitter
from fold.transformations.columns import OnlyPredictions
from fold.utils.tests import generate_zeros_and_ones


def test_metalabeling() -> None:
    X, y = generate_zeros_and_ones(1000)

    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
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

    trained_pipelines = train(transformations, X, y, splitter)
    pred = backtest(trained_pipelines, X, y, splitter)
    assert (pred.squeeze() == 0.5).all()
