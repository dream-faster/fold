from copy import deepcopy
from typing import List

import pandas as pd

from ..transformations.base import Composite, Transformations


class Ensemble(Composite):
    def __init__(self, models: Transformations) -> None:
        self.models = models
        self.name = "Ensemble-" + "-".join(
            [
                transformation.name if hasattr(transformation, "name") else ""
                for transformation in models
            ]
        )

    def postprocess_result(self, results: List[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(results, axis=1).mean(axis=1).to_frame()

    def get_child_transformations(self) -> Transformations:
        return self.models


class PerColumnEnsemble(Composite):
    def __init__(self, models: Transformations) -> None:
        self.models = models
        self.name = "PerColumnEnsemble-" + "-".join(
            [
                transformation.name if hasattr(transformation, "name") else ""
                for transformation in models
            ]
        )

    def before_fit(self, X: pd.DataFrame) -> None:
        self.models = [deepcopy(self.models) for _ in X.columns]

    def preprocess_X(self, X: pd.DataFrame, index: int) -> pd.DataFrame:
        return X.iloc[:, index].to_frame()

    def postprocess_result(self, results: List[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(results, axis=1).mean(axis=1).to_frame()

    def get_child_transformations(self) -> Transformations:
        return self.models
