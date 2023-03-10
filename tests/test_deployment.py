import numpy as np

from fold.loop import infer, train_for_deployment, update
from fold.models.baseline import BaselineNaive
from fold.utils.tests import generate_sine_wave_data


def test_deployment() -> None:
    # the naive model returns X as prediction, so y.shift(1) should be == pred
    X, y = generate_sine_wave_data()
    X_train = X[:900]
    X_test = X[900:]
    y_train = y[:900]
    y_test = y[900:]

    transformations = [BaselineNaive()]
    deployable_transformations = train_for_deployment(transformations, X_train, y_train)

    first_prediction = infer(
        deployable_transformations,
        X_test.iloc[0:1],
    )
    assert first_prediction.squeeze() == y_train.iloc[-1]

    preds = []
    for index in X_test.index:
        X = X_test.loc[index:index]
        y = y_test.loc[index:index]
        preds.append(infer(deployable_transformations, X).squeeze())
        deployable_transformations = update(deployable_transformations, X, y)

    assert (y_test.shift(1).values[1:] == np.array(preds)[1:]).all()
