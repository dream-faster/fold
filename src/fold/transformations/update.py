from typing import Optional

import pandas as pd

from .base import Transformation, fit_noop


class DontUpdate(Transformation):
    properties = Transformation.Properties()

    def __init__(self, transformation: Transformation) -> None:
        self.transformation = transformation
        self.name = f"DontUpdate-{transformation.name}"

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        self.transformation.fit(X, y, sample_weights)

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return self.transformation.transform(X, in_sample)

    update = fit_noop


class InjectPastDataAtInference(Transformation):
    """
    This transformation is used to inject all past `X` into the wrapped transformation, but only at inference time.
    """

    properties = Transformation.Properties()
    past_X: Optional[pd.DataFrame] = None

    def __init__(self, transformation: Transformation) -> None:
        self.transformation = transformation
        self.name = f"InjectPastDataAtInference-{transformation.name}"
        self.properties.mode = transformation.properties.mode

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        self.transformation.fit(X, y, sample_weights)
        self.past_X = (
            pd.concat([self.past_X, X], axis="index") if self.past_X is not None else X
        )

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        complete_X = (
            pd.concat([self.past_X, X], axis="index") if self.past_X is not None else X
        )
        result = self.transformation.transform(complete_X, in_sample)
        return result.loc[X.index]

    update = fit
