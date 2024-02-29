# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from random import choices

import numpy as np
import pandas as pd
from finml_utils.dataframes import concat_on_columns

from ..base import Artifact, Transformation, fit_noop
from .base import Model


class RandomClassifier(Model):
    """
    A model that predicts random classes and probabilities.

    Parameters
    ----------
    all_classes : list[int]
        All possible classes.
    probability_mean : float
        The mean of the normal distribution used to generate the probabilities.


    Examples
    --------
    ```pycon
    >>> import numpy as np
    >>> from fold.loop import train_backtest
    >>> from fold.splitters import SlidingWindowSplitter
    >>> from fold.models import RandomClassifier
    >>> from fold.utils.tests import generate_sine_wave_data
    >>> X, y  = generate_sine_wave_data()
    >>> splitter = SlidingWindowSplitter(train_window=0.5, step=0.2)
    >>> np.random.seed(42)
    >>> pipeline = RandomClassifier([0,1], [0.5, 0.5])
    >>> preds, trained_pipeline, _, _ = train_backtest(pipeline, X, y, splitter)

    ```
    """

    name = "RandomClassifier"

    def __init__(
        self, all_classes: list[int], probability_mean: list[float] | None = None
    ) -> None:
        self.all_classes = all_classes
        if probability_mean is not None:
            assert len(probability_mean) == len(all_classes)
        self.probability_mean = (
            [1 / len(all_classes) for _ in range(len(all_classes))]
            if probability_mean is None
            else probability_mean
        )
        self.properties = Transformation.Properties(requires_X=False)

    def predict(self, X: pd.DataFrame) -> pd.Series | pd.DataFrame:
        predictions = pd.Series(
            choices(population=self.all_classes, k=len(X)),
            index=X.index,
            name="predictions_RandomClassifier",
        )
        probabilities = concat_on_columns(
            [
                pd.Series(
                    np.random.normal(prob_mean, 0.1, len(X)).clip(0, 1),
                    index=X.index,
                    name=f"probabilities_RandomClassifier_{associated_class}",
                )
                for associated_class, prob_mean in zip(
                    self.all_classes, self.probability_mean, strict=True
                )
            ],
        )
        probabilities = probabilities.div(probabilities.sum(axis=1), axis=0)

        return concat_on_columns([predictions, probabilities])

    fit = fit_noop
    predict_in_sample = predict
    update = fit


class RandomBinaryClassifier(Model):
    """
    A random model that mimics the probability distribution of the target seen during fitting.
    """

    properties = Transformation.Properties(requires_X=False)
    name = "RandomBinaryClassifier"

    def predict(self, X: pd.DataFrame) -> pd.Series | pd.DataFrame:
        return generate_synthetic_predictions_binary(
            self.memory_target, self.memory_sample_weights, X.index
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series | None = None,
        raw_y: pd.Series | None = None,
    ) -> Artifact | None:
        self.memory_target = y
        self.memory_sample_weights = (
            sample_weights
            if sample_weights is not None
            else pd.Series(1, index=y.index)
        )

    predict_in_sample = predict
    update = fit_noop


def generate_synthetic_predictions_binary(
    target: pd.Series,
    sample_weights: pd.Series,
    index: pd.Index,
) -> pd.DataFrame:
    target = target.copy()
    target[target == 0.0] = -1
    prob_mean_class_1 = (target * sample_weights).mean() / 2 + 0.5
    prob_class_1 = np.random.normal(prob_mean_class_1, 0.1, len(index)).clip(0, 1)
    prob_class_0 = 1 - prob_class_1
    return pd.DataFrame(
        {
            "predictions_RandomClassifier": (prob_class_1 > prob_mean_class_1).astype(
                "int"
            ),
            "probabilities_RandomClassifier_0": prob_class_0,
            "probabilities_RandomClassifier_1": prob_class_1,
        },
        index=index,
    )
