import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from fold.loop import infer, train
from fold.splitters import ExpandingWindowSplitter
from fold.utils.tests import generate_monotonous_data


def test_deployment() -> None:
    X, y = generate_monotonous_data()
    X_train = X[:900]
    X_test = X[900:]
    y_train = y[:900]
    y_test = y[900:]

    transformations = LinearRegression()
    trained_pipelinecard, _, _ = train(
        transformations,
        X_train,
        y_train,
        splitter=ExpandingWindowSplitter(0.2, 0.1),
        for_deployment=True,
    )

    first_prediction = infer(
        trained_pipelinecard,
        pd.concat([X_train.iloc[-1:None], X_test.iloc[0:1]], axis="index"),
    )
    assert np.isclose(first_prediction.squeeze()[-1], y_test.iloc[0], atol=0.001)

    # preds = []
    # for index in X_test.index:
    #     X = X_test.loc[index:index]
    #     y = y_test.loc[index:index]
    #     preds.append(infer(deployable_transformations, X).squeeze())
    #     deployable_transformations = update(deployable_transformations, X, y)

    # assert (y_test.shift(1).values[1:] == np.array(preds)[1:]).all()
