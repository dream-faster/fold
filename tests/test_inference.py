from drift.loop import infer, train_for_deployment, update
from drift.models import BaselineRegressor
from drift.splitters import ExpandingWindowSplitter
from drift.utils.tests import generate_sine_wave_data


def test_deployment() -> None:

    # the naive model returns X as prediction, so y.shift(1) should be == pred
    data = generate_sine_wave_data()
    X_train = data[:900]
    X_test = data[901:]
    y_train = data["sine"][:900].shift(-1)
    y_test = data["sine"][901:].shift(-1)

    transformations = [BaselineRegressor(strategy=BaselineRegressor.Strategy.naive)]
    deployable_transformations = train_for_deployment(transformations, X_train, y_train)

    # for row_X, row_y in zip(X_test, y_test):

    #     X =
    #     pred = infer(deployable_transformations, row_X)
    #     deployable_transformations = update(deployable_transformations, X_train, y_train)

    # assert (X.squeeze()[pred.index] == pred.squeeze()).all()
