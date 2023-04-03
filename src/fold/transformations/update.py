from typing import Optional

import pandas as pd

from ..base import Transformation, fit_noop


class DontUpdate(Transformation):
    """
    Don't update the wrapped Transformation
    """

    properties = Transformation.Properties(requires_X=False)

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
    This transformation is used to inject `window_size` (if None, all) of past `X` into the wrapped transformation, but only at inference time.
    """

    def __init__(
        self, transformation: Transformation, window_size: Optional[int]
    ) -> None:
        self.transformation = transformation
        self.name = f"InjectPastDataAtInference-{transformation.name}"
        self.properties = Transformation.Properties(
            requires_X=False,
            memory_size=0 if window_size is None else window_size,
            mode=transformation.properties.mode,
        )

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return self.transformation.transform(X, in_sample)

    fit = fit_noop
    update = fit
