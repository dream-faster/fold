from random import choices, random
from typing import List, Union

import numpy as np
import pandas as pd

from ..transformations.base import Transformation
from .base import Model


class RandomClassifier(Model):

    properties = Transformation.Properties(requires_past_X=False)
    name = "RandomClassifier"

    def __init__(self, all_classes: List[int], probability_mean: float = 0.5) -> None:
        self.all_classes = all_classes
        self.probability_mean = probability_mean

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    def predict(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        predictions = pd.Series(
            choices(population=self.all_classes, k=len(X)),
            index=X.index,
            name="predictions_RandomClassifier",
        )
        probabilities = [
            pd.Series(
                np.random.normal(self.probability_mean, 0.1, len(X)).clip(0, 1),
                index=X.index,
                name=f"probabilities_RandomClassifier_{associated_class}",
            )
            for associated_class in self.all_classes
        ]

        return pd.concat([predictions] + probabilities, axis=1)
