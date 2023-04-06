from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from fold.base import InvertibleTransformation, fit_noop


class TakeLog(InvertibleTransformation):
    def __init__(
        self,
        base: Union[int, str],
    ) -> None:
        if base not in ["e", "10", 10, np.e, "2", 2]:
            raise ValueError("base should be either 'e', '10', 10, '2', 2.")
        self.base = base

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        if self.base == "e" or self.base == np.e:
            return pd.DataFrame(np.log(X.values), columns=X.columns)
        elif self.base == "10" or self.base == 10:
            return pd.DataFrame(np.log10(X.values), columns=X.columns)
        elif self.base == "2" or self.base == 2:
            return pd.DataFrame(np.log2(X.values), columns=X.columns)
        else:
            raise ValueError(f"Invalid base: {self.base}")

    def inverse_transform(self, X: pd.Series) -> pd.Series:
        if self.base == "e" or self.base == np.e:
            return pd.Series(np.exp(X.values), index=X.index)
        elif self.base == "10" or self.base == 10:
            return 10**X
        elif self.base == "2" or self.base == 2:
            return 2**X
        else:
            raise ValueError(f"Invalid base: {self.base}")

    fit = fit_noop
    update = fit_noop


class AddConstant(InvertibleTransformation):
    def __init__(
        self,
        constant: Union[int, float, Dict[str, Union[float, int]]],
    ) -> None:
        if not isinstance(constant, (int, float, dict)) and not constant == "auto":
            raise ValueError("C can take only 'auto', integers or floats")

        self.constant = constant

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X + self.constant

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X - self.constant

    fit = fit_noop
    update = fit_noop


class TurnPositive(InvertibleTransformation):
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        self.constant = dict(X.min(axis=0).abs() + 1)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X + self.constant

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X - self.constant

    update = fit_noop
