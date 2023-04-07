from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from fold.base import InvertibleTransformation, fit_noop


class TakeLog(InvertibleTransformation):
    name = "TakeLog"
    properties = InvertibleTransformation.Properties(requires_X=True)

    def __init__(
        self,
        base: Union[int, str] = "e",
    ) -> None:
        if base not in ["e", np.e, "10", 10, "2", 2]:
            raise ValueError("base should be either 'e', np.e, '10', 10, '2', 2.")
        self.base = base

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        if self.base == "e" or self.base == np.e:
            return pd.DataFrame(np.log(X.values), columns=X.columns, index=X.index)
        elif self.base == "10" or self.base == 10:
            return pd.DataFrame(np.log10(X.values), columns=X.columns, index=X.index)
        elif self.base == "2" or self.base == 2:
            return pd.DataFrame(np.log2(X.values), columns=X.columns, index=X.index)
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
    name = "AddConstant"
    properties = InvertibleTransformation.Properties(requires_X=True)

    def __init__(
        self,
        constant: Union[int, float, Dict[str, Union[float, int]]],
    ) -> None:
        if not isinstance(constant, (int, float, dict)):
            raise ValueError(
                "constant can be only integer, float or a dictionary of integers or floats"
            )

        self.constant = constant

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        if isinstance(self.constant, dict):
            transformed_columns = X[list(self.constant.keys())] + pd.Series(
                self.constant
            )
            return pd.concat(
                [X.drop(columns=self.constant.keys()), transformed_columns],
                axis="columns",
            )
        else:
            return X + self.constant

    def inverse_transform(self, X: pd.Series) -> pd.Series:
        constant = self.constant
        if constant is dict:
            constant = next(iter(constant.values()))
        return X - constant

    fit = fit_noop
    update = fit_noop


class TurnPositive(InvertibleTransformation):
    name = "TurnPositive"
    properties = InvertibleTransformation.Properties(requires_X=True)

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        sample_weights: Optional[pd.Series] = None,
    ) -> None:
        min_values = X.min(axis=0)
        self.constant = dict(min_values[min_values <= 0].abs() + 1)

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        transformed_columns = X[list(self.constant.keys())] + pd.Series(self.constant)
        return pd.concat(
            [X.drop(columns=self.constant.keys()), transformed_columns],
            axis="columns",
        )

    def inverse_transform(self, X: pd.Series) -> pd.Series:
        return X - next(iter(self.constant.values()))

    update = fit_noop
