from typing import List, Tuple, Union

import pandas as pd

from fold.utils.checks import is_X_available

from ..utils.list import flatten, transform_range_to_list, wrap_in_list
from .base import Transformation, fit_noop


class AddLagsY(Transformation):
    """
    Adds past values of `y`.
    """

    def __init__(self, lags: Union[List[int], range]) -> None:
        if not isinstance(lags, range) and not isinstance(lags, List):
            raise ValueError("lags must be a range or a List")
        self.lags = sorted(transform_range_to_list(lags))
        self.name = f"AddLagsY-{self.lags}"
        self.properties = Transformation.Properties(
            requires_X=False,
            mode=Transformation.Properties.Mode.online,
            memory_size=max(self.lags),
            _internal_supports_minibatch_backtesting=True,
        )

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        lags = pd.DataFrame([])

        if in_sample:
            for lag in self.lags:
                lags[f"y_lag_{lag}"] = self._state.memory_y.shift(lag)[-len(X) :]
        else:
            past_y = self._state.memory_y.reindex(X.index)
            for lag in self.lags:
                lags[f"y_lag_{lag}"] = past_y.shift(lag)[-len(X) :]

        if is_X_available(X):
            return pd.concat([X, lags], axis="columns")
        else:
            # If X is just an DataFrame with zeros, then just return the lags
            return lags

    fit = fit_noop
    update = fit_noop


class AddLagsX(Transformation):
    """
    Adds past values of `X` for the desired column(s).
    """

    ColumnAndLag = Tuple[str, Union[int, List[int]]]

    def __init__(
        self, columns_and_lags: Union[List[ColumnAndLag], ColumnAndLag]
    ) -> None:
        self.columns_and_lags = wrap_in_list(columns_and_lags)

        def check_and_transform_if_needed(
            column_and_lag: AddLagsX.ColumnAndLag,
        ) -> AddLagsX.ColumnAndLag:
            column, lags = column_and_lag
            if (
                not isinstance(lags, int)
                and not isinstance(lags, List)
                and not isinstance(lags, range)
            ):
                raise ValueError("lags must be an int or a List or a range")
            lags = sorted(
                transform_range_to_list([lags] if isinstance(lags, int) else lags)
            )
            return column, lags

        self.columns_and_lags = list(
            map(check_and_transform_if_needed, self.columns_and_lags)
        )
        self.name = f"AddLagsX-{self.columns_and_lags}"
        self.properties = Transformation.Properties(
            requires_X=True,
            memory_size=max(flatten([l for _, l in self.columns_and_lags])),
        )

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        X_lagged = pd.DataFrame([])
        for column, lags in self.columns_and_lags:
            for lag in lags:
                if column == "all":
                    X_lagged = pd.concat(
                        [X_lagged, X.shift(lag)[-len(X) :].add_suffix(f"_lag_{lag}")],
                        axis="columns",
                    )
                else:
                    X_lagged[f"{column}_lag_{lag}"] = X[column].shift(lag)[-len(X) :]
        return pd.concat([X, X_lagged], axis="columns")

    fit = fit_noop
    update = fit_noop
