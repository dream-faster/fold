import pandas as pd

from .base import Transformation, fit_noop


class Identity(Transformation):
    properties = Transformation.Properties()

    name = "Identity"

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return X

    fit = fit_noop
    update = fit
