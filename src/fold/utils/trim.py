# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from typing import Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

from ..base import Extras

T = TypeVar("T", pd.Series, Optional[pd.Series])


def trim_initial_nans(
    X: pd.DataFrame, y: pd.Series, extras: Extras
) -> Tuple[pd.DataFrame, pd.Series, Extras]:
    # Optimize for speed, if the first value is not NaN, we can save all the subsequent computation
    if not X.iloc[0].isna().any() and (y is None or not np.isnan(y.iloc[0])):
        return X, y, extras
    first_valid_index_X = get_first_valid_index(X)
    first_valid_index_y = get_first_valid_index(y)
    if first_valid_index_X is None or first_valid_index_y is None:
        return (
            pd.DataFrame(),
            pd.Series(dtype="float64"),
            Extras(),
        )
    first_valid_index = max(first_valid_index_X, first_valid_index_y)
    return (
        X.iloc[first_valid_index:],
        y.iloc[first_valid_index:],
        extras.iloc(slice(first_valid_index, None)),
    )


def trim_initial_nans_single(X: pd.DataFrame) -> pd.DataFrame:
    # Optimize for speed, if the first value is not NaN, we can save all the subsequent computation
    if not X.iloc[0].isna().any():
        return X
    first_valid_index = get_first_valid_index(X)
    return X.iloc[first_valid_index:]


def get_first_valid_index(series: Union[pd.Series, pd.DataFrame]) -> int:
    if series.empty:
        return 0
    if isinstance(series, pd.DataFrame):
        return next(
            (
                idx
                for idx, (_, x) in enumerate(series.iterrows())
                if not pd.isna(x).any()
            ),
            None,
        )
    elif isinstance(series, pd.Series):
        return next(
            (idx for idx, (_, x) in enumerate(series.items()) if not pd.isna(x)),
            None,
        )
