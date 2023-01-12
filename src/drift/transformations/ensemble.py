from typing import List, Optional

import numpy as np
import pandas as pd

from .base import Transformation


class Ensemble(Transformation):
    def __init__(self, models: List[Transformation]) -> None:
        self.models = models
        self.name = "Ensemble-" + "-".join([model.name for model in models])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        # to be done in the training loop with get_child_transformations()
        pass

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return np.mean(np.vstack([model.transform(X) for model in self.models]), axis=0)

    def get_child_transformations(self) -> Optional[List[Transformation]]:
        return self.models
